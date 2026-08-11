"""
本地智能大脑 - 替代 Dify 的完整处理链路。

架构：
                        ┌──> [设备控制轨] ──> 规则NLU/LLM JSON ──> MQTT 下发
 用户文本 ──> 意图路由 ┤
                        └──> [知识问答轨] ──> ChromaDB 检索 ──> LLM 回答

设备控制轨：
  1. 优先用规则 NLU (nlu_parser) 解析，命中率高且零延迟
  2. 规则未命中时用 LLM (llm_nlu) 兜底，输出结构化 JSON
  3. 通过 execute_command 执行设备控制 + MQTT 下发
  4. 生成自然语言回复

知识问答轨：
  1. 从 ChromaDB 检索 Top-K 相关知识段落
  2. 将检索结果注入 LLM Prompt，生成回答
"""
import asyncio
import json
from typing import Dict, Any, Optional

from router import route_intent, route_intent_with_confidence
from nlu_parser import parse_multi_intent, parse_intent, get_device_risk, set_last_device
from device_controller import execute_command, DEVICE_STATES, state_lock
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, SILICONFLOW_MODEL, LLM_TIMEOUT
from llm_nlu import parse_intent_llm

# Device state formatting (inlined from old dify_client — dependency removed)
_AC_MODE_CN = {
    "cooling": "制冷", "heating": "制热", "dehumidify": "除湿", "fan": "送风", "auto": "自动",
    "cool": "制冷", "heat": "制热", "dry": "除湿",
}
_AC_FAN_CN = {
    "quiet": "静音档", 1: "1档", 2: "2档", 3: "3档", 4: "4档", 5: "5档", "auto": "自动风",
}
_TYPE_CN = {"ac": "空调", "light": "灯", "curtain": "窗帘", "tv": "电视"}


def build_device_states_context() -> str:
    """构建设备当前状态的简短字符串"""
    parts = []
    for dev_id, state in DEVICE_STATES.items():
        room = state.get("room", "")
        dev_type = state.get("type", "")
        display = room + _TYPE_CN.get(dev_type, "设备")
        if dev_type == "curtain":
            status = "已拉开" if state.get("state") == "open" else "已关闭"
            parts.append(f"{display}:{status}")
        elif dev_type == "ac":
            power = "开" if state.get("power") == "on" else "关"
            temp = state.get("temperature", "")
            mode = state.get("mode", "")
            mode_cn = _AC_MODE_CN.get(mode, "")
            status = f"{power}"
            if temp:
                status += f" {temp}°"
            if mode_cn:
                status += f" {mode_cn}"
            fan_speed = state.get("fan_speed", "")
            if fan_speed:
                fan_cn = _AC_FAN_CN.get(fan_speed,
                    str(fan_speed) + "档" if isinstance(fan_speed, int) else str(fan_speed))
                status += f" {fan_cn}"
            parts.append(f"{display}:{status}")
        else:
            power = "开" if state.get("power") == "on" else "关"
            parts.append(f"{display}:{power}")
    return "，".join(parts)


async def process(text: str, session_id: str = "default") -> Dict[str, Any]:
    """
    本地大脑主入口：接收用户文本，路由后分别处理。

    Args:
        text: 用户输入文本（ASR 识别结果或直接文字）
        session_id: 会话ID，用于上下文追踪

    Returns:
        {
            "success": bool,
            "reply": str,          # 回复文本
            "intent": str,         # "device_control" | "knowledge_qa"
            "device_executed": bool,  # 是否执行了设备控制
            "commands": list,      # 执行的命令列表（设备控制时）
            "error": str | None,
        }
    """
    if not text or not text.strip():
        return {"success": False, "reply": "请说出您的指令", "intent": "unknown", "error": "empty input"}

    # 1. 意图路由
    route_result = route_intent_with_confidence(text)
    intent_type = route_result["intent"]
    print(f"[SmartBrain] 意图路由: {intent_type} (confidence={route_result['confidence']})")
    print(f"[SmartBrain] 信号: {route_result['signals']}")

    # 2. 分轨处理
    if intent_type == "device_control":
        return await _handle_device_control(text, session_id)
    else:
        return await _handle_knowledge_qa(text, session_id)


async def _handle_device_control(text: str, session_id: str) -> Dict[str, Any]:
    """
    设备控制轨：规则NLU → LLM兜底 → 执行命令 → 生成回复

    策略：
    1. 先用规则 NLU 解析（parse_multi_intent）
    2. 如果规则未命中（空列表），用 LLM 兜底
    3. 执行解析出的所有意图
    4. 拼接回复文本
    """
    print(f"[SmartBrain] → 设备控制轨")

    # 构建设备状态上下文
    device_context = build_device_states_context()
    print(f"[SmartBrain] 设备状态: {device_context}")

    # 1. 尝试规则 NLU
    intents = parse_multi_intent(text, session_id)
    used_method = "rule"

    # 2. 规则未命中 → LLM 兜底
    if not intents and SILICONFLOW_API_KEY:
        print(f"[SmartBrain] 规则NLU未命中，尝试LLM兜底...")
        llm_result = await asyncio.to_thread(parse_intent_llm, text)
        if llm_result.get("success") and llm_result.get("intent"):
            intents = [llm_result["intent"]]
            used_method = "llm"
            print(f"[SmartBrain] LLM解析成功: {llm_result['intent']}")

    if not intents:
        return {
            "success": False,
            "reply": "抱歉，我没有理解您的设备控制指令，可以再说一遍吗？",
            "intent": "device_control",
            "device_executed": False,
            "commands": [],
            "error": "no intent parsed",
            "method": used_method,
        }

    # 3. 执行所有意图
    results = []
    messages = []
    commands_executed = []

    for intent in intents:
        # 记录最后操作的设备（用于上下文续控）
        device_id = intent.get("device_id", "")
        if session_id and device_id:
            set_last_device(session_id, device_id)

        # 高风险设备消歧
        risk = get_device_risk(device_id)
        if risk == "high_risk" and intent.get("need_clarification"):
            messages.append(f"请问您要操作哪个房间的{intent.get('device_name', '设备')}？")
            continue

        # 执行命令
        result = await execute_command(intent)
        results.append(result)
        commands_executed.append({
            "device_id": intent.get("device_id"),
            "action": intent.get("action"),
            "value": intent.get("value"),
        })

        if result.get("success"):
            msg = result.get("message", "")
            if msg:
                messages.append(msg)
        else:
            err_msg = result.get("message", "执行失败")
            messages.append(err_msg)

    # 4. 拼接回复：优先用 main.py 的 _merge_messages 进行多意图合并
    if not messages:
        reply_text = "指令已执行"
    else:
        try:
            from main import _merge_messages
            reply_text = _merge_messages(messages)
        except Exception:
            reply_text = "，".join(messages)

    return {
        "success": all(r.get("success", False) for r in results) if results else False,
        "reply": reply_text,
        "intent": "device_control",
        "device_executed": len(commands_executed) > 0,
        "commands": commands_executed,
        "method": used_method,
        "error": None,
    }


async def _handle_knowledge_qa(text: str, session_id: str) -> Dict[str, Any]:
    """
    知识问答轨：ChromaDB 检索 → LLM 回答 / 天气查询 → API

    策略：
    1. 检测天气查询 → 调用天气 API 返回实时数据
    2. 从 ChromaDB 检索 Top-2 相关知识段落
    3. 将检索结果注入 LLM Prompt，生成回答
    4. 若 RAG 不可用，降级为纯 LLM 对话
    """
    print(f"[SmartBrain] → 知识问答轨")

    # 检测天气查询（包含天气关键词）
    weather_keywords = {"天气", "气温", "下雨", "刮风", "台风", "降雪", "雷电", "雾霾"}
    has_weather = any(kw in text for kw in weather_keywords)
    if has_weather:
        print(f"[SmartBrain] 检测到天气查询，尝试天气 API...")
        from weather_service import get_weather, extract_city
        city = extract_city(text)
        if city:
            weather_data = await get_weather(city)
            if weather_data:
                reply = weather_data["description"]
                print(f"[SmartBrain] 天气查询成功: {reply}")
                return {
                    "success": True,
                    "reply": reply,
                    "intent": "knowledge_qa",
                    "device_executed": False,
                    "commands": [],
                    "rag_used": False,
                    "error": None,
                }
            else:
                print(f"[SmartBrain] 天气 API 查询失败，降级处理")
        # 有城市名但查询失败 → LLM 尝试回答
        # 无城市名 → LLM 询问城市

    # 延迟导入 RAG 服务（避免启动时加载 ChromaDB）
    rag_context = ""
    rag_sources = []
    try:
        from rag_service import search_knowledge
        results = search_knowledge(text, top_k=4)
        if results:
            rag_context = "\n\n".join([r["content"] for r in results])
            rag_sources = [r.get("source", "") for r in results]
            print(f"[SmartBrain] RAG 检索到 {len(results)} 条相关知识")
    except ImportError:
        print(f"[SmartBrain] RAG 服务未启用，降级为纯 LLM 对话")
    except Exception as e:
        print(f"[SmartBrain] RAG 检索失败: {e}，降级为纯 LLM 对话")

    # 构建 LLM Prompt
    if rag_context:
        # 预处理知识库：将故障代码转为描述性文本，防止 LLM 复述代码字符
        processed_rag = _preprocess_rag_context(rag_context)
        system_prompt = f"""你是一个简洁的智能家居问答助手。

【你的任务】
根据提供的参考资料回答用户的问题，必须覆盖参考资料中的所有不同要点，不要只挑第一条。

【严格回答规则】
1. 用自己的话总结，绝对不要逐字复述参考资料中的任何字符或代码
2. 回答控制在 1~2 句、150 字以内，优先列出最常见的 3~4 个原因
3. 引用故障码时，直接说中文含义（如"通信故障"），不要复述代码原文
4. 保持语言自然流畅，不要出现任何单个字符的连续重复
5. 如果参考资料没有相关信息，请说"抱歉，我暂时无法回答这个问题"
6. 多要点时用分号或顿号连起来，尽量一句话说完

【参考资料（仅供理解，不要复述）】
{processed_rag}

请直接给出回答，不需要任何前缀或前缀语。"""
    else:
        if has_weather:
            system_prompt = """你是一个简洁的天气助手。根据用户的问题给出简短回复。
如果问了某城市天气但查询失败，说明天气服务暂时不可用，建议稍后查询。
如果没有提具体城市，请告诉用户需要说明哪个城市。
不需要任何前缀。"""
        else:
            system_prompt = """你是一个简洁的智能家居助手。请用简洁的中文回答，不超过3句话。
绝对不要出现字符重复。"""

    # 调用 LLM
    reply_text = await _call_llm_async(system_prompt, text)
    # 后处理：清洗退化输出
    if reply_text:
        reply_text = _clean_degenerate_output(reply_text)

    return {
        "success": bool(reply_text),
        "reply": reply_text or "抱歉，我暂时无法回答这个问题。",
        "intent": "knowledge_qa",
        "device_executed": False,
        "commands": [],
        "rag_used": bool(rag_context),
        "rag_sources": rag_sources,
        "error": None if reply_text else "LLM call failed",
    }


def _preprocess_rag_context(context: str) -> str:
    """
    预处理 RAG 检索到的上下文，将故障代码转为描述性文本。
    防止 LLM 在回答时复述代码字符导致退化。
    """
    import re

    processed = context

    # 先处理完整的"字母+数字"代码模式（如 "E1故障"、"F0错误"、"E2码"）
    # 将 "E1故障" -> "通信故障"，"F0错误" -> "室温传感器错误"
    code_replacements = {
        r'\bE1\b': '通信',
        r'\bE2\b': '防冻结',
        r'\bE3\b': '低压',
        r'\bE4\b': '高温',
        r'\bE5\b': '过载',
        r'\bF0\b': '室温传感器',
        r'\bF1\b': '管温传感器',
        r'\bF2\b': '外环传感器',
    }
    for code_pattern, desc in code_replacements.items():
        # 替换 "E1故障" -> "通信故障"、"E1错误" -> "通信错误"
        processed = re.sub(
            rf'{code_pattern}(故障|错误|码|代码)',
            f'{desc}\1',
            processed,
        )

    # 然后清除所有剩余的"字母+数字"代码模式（如独立的 "E1", "E2"）
    # 这会消除 LLM 直接复述代码原文的可能性
    processed = re.sub(r'\b[A-Z]\d+\b', '', processed)

    # 清除可能残留的孤立大写字母序列（如 "E E E E"）
    processed = re.sub(r'\b([A-Z])\s+\1(?:s+\1)+\b', '', processed)

    # 清除可能残留的孤立数字序列
    processed = re.sub(r'\b\d+\s+\d+\s+\d+\b', '', processed)

    # 清除多余空格
    processed = re.sub(r'\s+', ' ', processed)

    return processed.strip()


async def _call_llm_async(system_prompt: str, user_text: str) -> Optional[str]:
    """异步调用 SiliconFlow LLM API"""
    import httpx

    if not SILICONFLOW_API_KEY:
        return None

    url = f"{SILICONFLOW_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SILICONFLOW_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.5,
        "max_tokens": 256,
        "repetition_penalty": 1.15,
        "presence_penalty": 0.6,
        "frequency_penalty": 0.6,
    }

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[SmartBrain] LLM 调用失败: {e}")
        return None


def _clean_degenerate_output(text: str) -> str:
    """
    清理 LLM 退化输出中的连续重复字符。

    典型退化模式："E E E E E E E..." 或 "好好好好好好..."
    或 "E 故障，，，是室内外机通信出现问题， E ，可能是由于"
    处理策略：
    1. 清除所有 "X X X" 型重复（空格分隔的单字符重复）
    2. 清除连续单字符重复（长度>=5）
    3. 清除孤立字符后跟重复标点（如 "E ，，，"）
    4. 最终截断退化点
    """
    import re

    if not text or len(text) < 2:
        return text

    cleaned = text

    # Step 0: 清除孤立大写字母后接逗号/空格退化（如 "E ，，，"、"E ，"）
    # 匹配模式：单个字母（尤其是大写字母）被逗号、空格反复围绕
    cleaned = re.sub(r'\b([A-Z])\s*[，,]\s*(?:\1\s*[，,]\s*)*', '', cleaned)

    # Step 1: 清除空格分隔的单字符重复序列（如 "E E E E" → ""）
    cleaned = re.sub(r'(?:\b\w\b\s+){3,}\b\w\b', '', cleaned)

    # Step 2: 清除无间隔的连续单字符重复（如 "EEEEEE" → ""）
    cleaned = re.sub(r'(.)\1{4,}', '', cleaned)

    # Step 3: 查找退化起点，截断
    # 检查是否存在连续5次以上相同字符（无论是否有空格）
    degenerate_start = -1
    for i in range(len(cleaned) - 4):
        char = cleaned[i]
        if not char.strip():
            continue  # 跳过空格
        count = 0
        for j in range(i, min(i + 15, len(cleaned))):
            if cleaned[j] == char or cleaned[j] == ' ':
                if cleaned[j] == char:
                    count += 1
            else:
                break
        if count >= 5:
            degenerate_start = i
            break

    if degenerate_start >= 0:
        cleaned = cleaned[:degenerate_start + 1]

    # Step 4: 逐字符滑动窗口检查
    result_chars = []
    for i, ch in enumerate(cleaned):
        if i >= 4:
            if ch == cleaned[i-1] == cleaned[i-2] == cleaned[i-3]:
                continue
        result_chars.append(ch)

    result = ''.join(result_chars)

    # Step 5: 清理结束后多余的空格和标点
    result = re.sub(r'\s+', ' ', result).strip()
    result = result.strip('，,。.!?、 ')

    # Step 6: 如果清洗过度，用原文截取自第一个标点
    if len(result) < 5:
        for marker in ['。', '！', '？', '.', '!', '?']:
            idx = text.find(marker)
            if 10 <= idx <= 200:
                return text[:idx + 1]
        return text[:100]

    return result


async def process_with_scene(text: str, session_id: str = "default") -> Dict[str, Any]:
    """
    带场景检测的完整处理流程（供 main.py 调用）。

    在标准 process() 基础上增加：
    - 场景模式检测（睡眠/离家/观影）
    - 设备控制后的 AC 属性校正
    """
    result = await process(text, session_id)

    # 场景检测兜底
    if result["intent"] == "device_control":
        # 延迟导入，避免循环依赖
        from main import detect_scene_from_text, execute_scene
        scene = detect_scene_from_text(text)
        if scene:
            print(f"[SmartBrain] 场景检测命中: {scene['description']}")
            scene_result = await execute_scene(scene)
            scene_msg = scene_result.get("message", "")
            has_new = scene_result.get("has_new_changes", False)
            if scene_result.get("success") and scene_msg:
                # 情况A：原设备控制流程实际没有执行任何指令（如"我要睡觉了"→规则NLU未命中），
                #       此时直接用场景回复覆盖原来的"抱歉未理解"提示
                fallback_phrases = ["抱歉，我没有理解", "抱歉，无法识别", "可以再说一遍吗"]
                origin_fallback = any(p in result.get("reply", "") for p in fallback_phrases)
                if origin_fallback or not result.get("device_executed", False):
                    result["reply"] = scene_msg
                else:
                    # 情况B：原指令已执行 + 场景还有额外动作 → 拼接
                    clean_reply = result["reply"].rstrip("。！？.!?")
                    result["reply"] = f"{clean_reply}，同时{scene_msg}"
                result["success"] = True
                result["device_executed"] = result["device_executed"] or has_new
            elif not result.get("success", True) and not result.get("device_executed", False):
                # 场景也没产出消息，且原来的回复是兜底错误 → 改成更友好的场景消息
                if scene_msg:
                    result["reply"] = scene_msg
                    result["success"] = True

    return result