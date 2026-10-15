"""
本地智能大脑 - 替代 Dify 的完整处理链路。

架构：
                        ┌──> [设备控制轨] ──> 规则NLU/LLM JSON ──> MQTT 下发
 用户文本 ──> 意图路由 ┤
                        └──> [知识问答轨] ──> ChromaDB 检索 ──> LLM 回答

设备控制轨：
  1. 优先用规则 NLU (nlu_parser) 解析，命中率高且零延迟
  2. 规则未命中时用 LLM (llm_nlu) 兜底，输出结构化 JSON
  3. 【歧义检测】泛化设备ID（"空调"→ac）+ 多台同类设备 → 悬挂 pending_intent 并反问
  4. 通过 execute_command 执行设备控制 + MQTT 下发
  5. 生成自然语言回复

知识问答轨：
  1. 从 ChromaDB 检索 Top-K 相关知识段落
  2. 将检索结果注入 LLM Prompt，生成回答
"""
import asyncio
import json
from typing import Dict, Any, Optional, List, Tuple

from router import route_intent, route_intent_with_confidence
from nlu_parser import (parse_multi_intent, parse_intent, get_device_risk, set_last_device,
                        parse_device_query, split_compound_command, asr_phonetic_correct,
                        detect_explicit_room, resolve_device_id, ROOM_CN_TO_SLUG)
from device_controller import execute_command, DEVICE_STATES, state_lock
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, SILICONFLOW_MODEL, LLM_TIMEOUT
from llm_nlu import parse_intent_llm
from dialog_context import (
    GENERIC_DEV_TO_TYPE,
    is_generic_device_id,
    count_devices_of_type,
    get_candidate_rooms_for_type,
    build_clarification_question,
    PendingIntent,
    dialog_manager,
    extract_room_cn,
)

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

    # ★ 第一道全局 ASR 同音校正（路由 + 后续所有子流程都使用校正后文本）
    text = asr_phonetic_correct(text)
    if not text or not text.strip():
        return {"success": False, "reply": "请说出您的指令", "intent": "unknown", "error": "empty after correction"}

    # 1. 意图路由
    route_result = route_intent_with_confidence(text)
    intent_type = route_result["intent"]
    print(f"[SmartBrain] 意图路由: {intent_type} (confidence={route_result['confidence']})")
    print(f"[SmartBrain] 信号: {route_result['signals']}")

    # 2. 分轨处理
    if intent_type == "device_query":
        return await _handle_device_query(text, session_id)
    elif intent_type == "device_control":
        return await _handle_device_control(text, session_id)
    else:
        return await _handle_knowledge_qa(text, session_id)


async def _handle_device_control(text: str, session_id: str) -> Dict[str, Any]:
    """
    设备控制轨：【复合指令预拆分】→ 每个子指令独立 NLU + 独立执行 → 综合播报

    策略：
    1. ★ Verb-First 预拆分（Compound Splitter）：
       "关闭客厅灯打开卧室灯" → ["关闭客厅灯", "打开卧室灯"]
       每个子指令天然是【动作+设备】对，从根上避免"关闭X打开Y"中 X 的动作被 Y 污染
    2. 单指令：保持原有 parse_multi_intent（兼容"所有窗帘/全屋灯"这类泛化引用）
    3. 规则失败 → LLM 兜底（仅单指令）
    4. 每个 intent 独立执行（高风险消歧、状态校验、MQTT 下发）
    5. 综合播报（Combined TTS）：多个动作串成自然口语
    """
    print(f"[SmartBrain] → 设备控制轨")
    # ★ 设备控制轨再校正一次（万一 handle_context_clarification/main.py 入口没调）
    text = asr_phonetic_correct(text)

    # 构建设备状态上下文
    device_context = build_device_states_context()
    print(f"[SmartBrain] 设备状态: {device_context}")

    # ============== Step 1. 复合指令预拆分 ==============
    chunks = split_compound_command(text)
    print(f"[SmartBrain] 复合指令拆分: {chunks}")

    all_intents: List[Dict[str, Any]] = []
    used_method = "rule"

    if len(chunks) >= 2:
        # --- 复合模式：对每个 chunk 独立 parse_intent，彻底隔离动作与设备的绑定 ---
        for idx, chunk in enumerate(chunks):
            r = parse_intent(chunk)
            if r["success"] and r.get("intent"):
                intent = r["intent"]
                intent["_chunk_idx"] = idx
                all_intents.append(intent)
                if session_id and intent.get("device_id"):
                    set_last_device(session_id, intent["device_id"])
            else:
                # 兜底：chunk 可能是泛化词（"所有窗帘都拉上"这种），交给 parse_multi_intent
                fallback = parse_multi_intent(chunk, session_id)
                for i, it in enumerate(fallback):
                    it["_chunk_idx"] = idx
                    it["_chunk_sub"] = i
                    all_intents.append(it)
                    if session_id and it.get("device_id"):
                        set_last_device(session_id, it["device_id"])
    else:
        # --- 单指令模式：保持原有 parse_multi_intent（兼容泛化"所有灯"）---
        all_intents = parse_multi_intent(text, session_id)

        # 规则未命中 → LLM 兜底（仅单指令，复合模式不进 LLM 保证可控）
        if not all_intents and SILICONFLOW_API_KEY:
            print(f"[SmartBrain] 规则NLU未命中，尝试LLM兜底...")
            llm_result = await asyncio.to_thread(parse_intent_llm, text)
            if llm_result.get("success") and llm_result.get("intent"):
                all_intents = [llm_result["intent"]]
                used_method = "llm"
                print(f"[SmartBrain] LLM解析成功: {llm_result['intent']}")

    if not all_intents:
        return {
            "success": False,
            "reply": "抱歉，我没有理解您的设备控制指令，可以再说一遍吗？",
            "intent": "device_control",
            "device_executed": False,
            "commands": [],
            "error": "no intent parsed",
            "method": used_method,
        }

    # ============== Step 1.5. 【同设备多指令合并】 ==============
    #   解决："打开卧室空调调到24度制冷风速一档" 被拆成两条下发（on + set），
    #   导致 _execute_ac 命中 action=="on" 的幂等短路，属性更新被独立丢弃或回复拼接啰嗦。
    #   合并后只对每个 device_id 下发一次 _execute_ac，自然生成"已处于开启状态 + 多项属性"的一句话。
    def _merge_intents_for_same_device(intents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # lazy import：从 device_controller 借温度提取 helper（中文数字+阿拉伯数字二合一）
        try:
            from device_controller import _extract_any_temperature_degree as _extract_temp
        except Exception:
            _extract_temp = lambda s: None  # 兜底失败不报错

        merged: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for intent in intents:
            did = intent.get("device_id", "")
            if not did or did.endswith(".all"):
                # 泛化 .all 或 无 device_id 不合并
                merged[f"__orphan_{id(intent)}"] = intent
                order.append(f"__orphan_{id(intent)}")
                continue
            if did not in merged:
                merged[did] = dict(intent)
                order.append(did)
                base = merged[did]
                # 对 base 也补一遍 raw_text 温度（防止单条 intent 中文数字没解析）
                if (base.get("value") is None or base.get("value_type") != "temperature"):
                    raw = base.get("raw_text") or base.get("original_text") or ""
                    if raw:
                        t = _extract_temp(raw)
                        if t is not None:
                            base["value"], base["value_type"] = t, "temperature"
                            print(f"[SmartBrain][合并补温] base chunk 兜底温度: '{raw}' → {t}°")
                continue
            # ---- 同 device_id：合并 attrs / value / value_type / action / 顶层属性字段 ----
            base = merged[did]
            # (A) action：有明确 on/off 优先，其次 set，再保留原 action
            new_action = intent.get("action") or ""
            if new_action in ("on", "off"):
                base["action"] = new_action
            elif base.get("action") not in ("on", "off") and new_action:
                base["action"] = new_action

            # (B) 先补当前子 intent 自己的 value/value_type（★ 解决"调到二十四度"子句温度丢失）
            if (intent.get("value") is None or intent.get("value_type") != "temperature"):
                sub_raw = intent.get("raw_text") or intent.get("original_text") or ""
                if sub_raw:
                    sub_t = _extract_temp(sub_raw)
                    if sub_t is not None:
                        intent["value"], intent["value_type"] = sub_t, "temperature"
                        print(f"[SmartBrain][合并补温] 子句兜底温度: '{sub_raw}' → {sub_t}°")

            # (C) value / value_type：新的（更具体）覆盖旧的空值或弱值
            if intent.get("value") is not None and (base.get("value") is None):
                base["value"] = intent["value"]
            vt_new = intent.get("value_type") or ""
            vt_base = base.get("value_type") or ""
            if vt_new and vt_new != vt_base:
                # temperature > 其它类型
                if vt_new == "temperature" or not vt_base:
                    base["value_type"] = vt_new
                    if intent.get("value") is not None:
                        base["value"] = intent["value"]

            # (D) 顶层 temperature / mode / fan_mode / fan_speed / swing_mode / swing
            #     → 同步到 attrs，并在合适时覆盖 base 的 value/value_type
            for src_key, dst_key in (("temperature", "temperature"),
                                     ("mode", "mode"),
                                     ("fan_mode", "fan_speed"),
                                     ("fan_speed", "fan_speed"),
                                     ("swing_mode", "swing"),
                                     ("swing", "swing")):
                v = intent.get(src_key)
                if v is None:
                    continue
                base_attrs = base.get("attrs")
                if not isinstance(base_attrs, dict):
                    base_attrs = {}
                    base["attrs"] = base_attrs
                base_attrs.setdefault(dst_key, v)
                if src_key == "temperature" and (base.get("value") is None or base.get("value_type") != "temperature"):
                    if isinstance(v, (int, float)) or (isinstance(v, str) and str(v).strip().isdigit()):
                        base["value"] = int(v) if isinstance(v, (int, float)) else int(str(v).strip())
                        base["value_type"] = "temperature"

            # (E) attrs：update 合并（温度/mode/fan 在 (D) 已 setdefault 过，这里仍用 update 防漏）
            new_attrs = intent.get("attrs") or {}
            if new_attrs:
                base_attrs2 = base.get("attrs")
                if not isinstance(base_attrs2, dict):
                    base_attrs2 = {}
                    base["attrs"] = base_attrs2
                base_attrs2.update(new_attrs)

            # (F) raw_text / original_text / device_name 等保留（取第一个非空即可）
            for k in ("raw_text", "original_text", "device_name", "control_mode", "delta"):
                if not base.get(k) and intent.get(k):
                    base[k] = intent[k]
        return [merged[k] for k in order]

    if len(all_intents) > 1:
        _before = len(all_intents)
        all_intents = _merge_intents_for_same_device(all_intents)
        if len(all_intents) < _before:
            print(f"[SmartBrain] 同设备多指令合并: {_before} → {len(all_intents)} 条，用于 AC/TV 等多属性一次下发")

    # ============== Step 2. 先做【歧义检测】：泛化ID + 多台同类设备 = 反问 ==============
    #   但先执行【需求 4 最高优先级防线】：如果用户文本中明确提到了客厅/卧室（或同音错词：克厅/卧试…），
    #   就强制锁定该房间，把通用设备 ID（air_conditioner/light/curtain/tv）直接 resolve 到对应房间，
    #   严禁反问"请问您是要操作卧室窗帘还是客厅窗帘"。
    explicit_room_for_route = detect_explicit_room(text, asr_corrected=True)
    if explicit_room_for_route:
        for intent in all_intents:
            did = intent.get("device_id", "") or ""
            if is_generic_device_id(did):
                fixed = resolve_device_id(did, explicit_room=explicit_room_for_route)
                if fixed != did:
                    # 顺带把 device_name 改成 "卧室空调/客厅窗帘" 中文，后续拼 reply 更自然
                    room_cn = {v: k for k, v in ROOM_CN_TO_SLUG.items()}.get(explicit_room_for_route, "")
                    type_cn = {"air_conditioner": "空调", "ac": "空调", "light": "灯",
                               "curtain": "窗帘", "tv": "电视"}.get(did, "")
                    print(f"[SmartBrain] ★ 房间锁定: text='{text}' "
                          f"room={explicit_room_for_route}({room_cn}) → device_id {did!r} → {fixed!r}")
                    intent["device_id"] = fixed
                    intent["need_clarification"] = False
                    intent["_explicit_room_slug"] = explicit_room_for_route
                    if room_cn and type_cn:
                        intent["device_name"] = room_cn + type_cn

    for intent in all_intents:
        device_id = intent.get("device_id", "")
        dev_type = GENERIC_DEV_TO_TYPE.get(device_id, "")
        # 命中条件：①泛化设备ID  ②系统中该类型设备 > 1 台  ③用户【没有】明确给出房间
        if is_generic_device_id(device_id) and count_devices_of_type(dev_type) > 1:
            candidates = get_candidate_rooms_for_type(dev_type)
            target_val = _extract_target_val_with_clamp(intent, dev_type)
            action_label = intent.get("action") or _guess_action_label(intent, dev_type)
            # 挂起 pending_intent
            pending = dialog_manager.save_pending(
                session_id=session_id,
                intent=intent,
                original_text=text,
                target_device_type=dev_type,
                target_val=target_val,
                action_label=action_label,
                candidate_rooms_cn=candidates,
            )
            clarification_q = build_clarification_question(pending)
            print(f"[SmartBrain] 检测到设备歧义: type={dev_type}, candidates={candidates}, 反问={clarification_q}")
            # 直接返回澄清问题（不执行任何设备控制）
            return {
                "success": True,
                "reply": clarification_q,
                "intent": "device_control",
                "device_executed": False,
                "commands": [],
                "method": used_method,
                "need_clarification": True,
                "pending_intent": pending.to_slot_dict(),
                "error": None,
            }

    # ============== Step 3. 循环独立执行每个 intent ==============
    results: List[Dict[str, Any]] = []
    commands_executed: List[Dict[str, Any]] = []

    # 结构化执行记录（为综合播报准备）：
    #   {"kind":"done"/"already"/"clarify"/"fail", "device": str, "action_label": str, "msg": str}
    exec_records: List[Dict[str, Any]] = []

    for intent in all_intents:
        device_id = intent.get("device_id", "")
        # 记录上下文（用于后续"调高2度"这类无设备名指令）
        if session_id and device_id:
            set_last_device(session_id, device_id)

        # 高风险设备消歧（但【需求 4】已给明确房间时禁止反问）
        risk = get_device_risk(device_id)
        local_explicit = intent.get("_explicit_room_slug") or explicit_room_for_route \
            or detect_explicit_room(text, asr_corrected=True)
        if risk == "high_risk" and intent.get("need_clarification") and not local_explicit:
            dev_type = GENERIC_DEV_TO_TYPE.get(device_id, "")
            candidates = get_candidate_rooms_for_type(dev_type) if dev_type else []
            target_val = _extract_target_val_with_clamp(intent, dev_type)
            action_label = intent.get("action") or ""
            pending = dialog_manager.save_pending(
                session_id=session_id,
                intent=intent,
                original_text=text,
                target_device_type=dev_type,
                target_val=target_val,
                action_label=action_label,
                candidate_rooms_cn=candidates,
            )
            clarification_q = build_clarification_question(pending)
            exec_records.append({
                "kind": "clarify",
                "device": intent.get("device_name") or "设备",
                "msg": clarification_q,
            })
            continue

        # 真正下发命令（包含状态判断 / turn_off 误判 防护在 execute_command 内部）
        # —— ★ 诊断日志：让用户/调试一眼能看到"下发给 execute_command 的 intent 字段"和"返回的 message"
        did = intent.get("device_id") or ""
        is_ac_intent = "ac" in did or (intent.get("device_name") or "").find("空调") >= 0
        if is_ac_intent:
            print(f"[SmartBrain] ★ 下发 AC intent → "
                  f"device_id={did!r}, action={intent.get('action')!r}, "
                  f"value={intent.get('value')!r}, value_type={intent.get('value_type')!r}, "
                  f"attrs={intent.get('attrs')!r}, raw_text={intent.get('raw_text') or intent.get('original_text')!r}")
        result = await execute_command(intent)
        results.append(result)
        if is_ac_intent:
            _ret_msg = result.get("message", "")
            print(f"[SmartBrain] ★ execute_command[AC] 返回 → "
                  f"success={result.get('success')}, message={_ret_msg!r}")
        commands_executed.append({
            "device_id": intent.get("device_id"),
            "action": intent.get("action"),
            "value": intent.get("value"),
        })

        action_label = intent.get("action") or ""
        dev_name = intent.get("device_name") or device_id or "设备"
        if result.get("success"):
            msg = result.get("message", "")
            # 区分：已是目标状态 vs 实际执行了控制
            already_hit = bool(msg) and (
                "已经" in msg or "已是" in msg or "就是" in msg or "仍然是" in msg
            )
            if already_hit:
                exec_records.append({
                    "kind": "already", "device": dev_name,
                    "action_label": action_label, "msg": msg,
                })
            else:
                exec_records.append({
                    "kind": "done", "device": dev_name,
                    "action_label": action_label, "msg": msg,
                })
        else:
            exec_records.append({
                "kind": "fail", "device": dev_name,
                "action_label": action_label,
                "msg": result.get("message", "执行失败"),
            })

    # ============== Step 4. 综合播报（Combined TTS）==============
    reply_text = _build_compound_reply(exec_records)

    # —— ★ Reply 强制非空（_handle_device_control 出口） ——
    if not isinstance(reply_text, str) or not reply_text.strip():
        reply_text = "操作已完成。"
        print(f"[Reply-Safe] _handle_device_control: reply_text 为空，强制兜底 → {reply_text!r}")

    return {
        "success": all(r.get("success", False) for r in results) if results else True,
        "reply": reply_text,
        "intent": "device_control",
        "device_executed": len(commands_executed) > 0,
        "commands": commands_executed,
        "method": used_method,
        "need_clarification": any(r.get("kind") == "clarify" for r in exec_records),
        "error": None,
    }


def _extract_target_val_with_clamp(intent: Dict[str, Any], dev_type: str) -> Any:
    """
    从 intent 中提取最终目标值，并按物理范围做 clamp（满足需求 1：target_val 必须是已做过边界限制的值）。
    例：空调温度设 50 → 30；AC 风扇档位设 9 → 3。
    """
    raw_value = intent.get("value")
    if raw_value is None and intent.get("attrs"):
        raw_value = intent["attrs"].get("temperature") or intent["attrs"].get("fan_speed")
    if raw_value is None:
        return None
    # 空调温度 clamp
    if dev_type == "ac":
        value_type = intent.get("value_type", "")
        # 温度类
        if value_type == "temperature" or isinstance(raw_value, (int, float)):
            attr_temp = (intent.get("attrs") or {}).get("temperature")
            candidate = raw_value if (isinstance(raw_value, (int, float))) else attr_temp
            if isinstance(candidate, (int, float)):
                return int(max(16, min(30, candidate)))
    # 灯亮度 clamp
    if dev_type == "light" and isinstance(raw_value, (int, float)):
        return int(max(0, min(100, raw_value)))
    return raw_value


def _guess_action_label(intent: Dict[str, Any], dev_type: str) -> str:
    """根据 intent + 设备类型推断 action_label，用于 build_clarification_question 选择友好动词"""
    action = (intent.get("action") or "").strip().lower()
    if action:
        return action
    value_type = intent.get("value_type", "") or ""
    if value_type == "temperature" or dev_type == "ac":
        val = intent.get("value")
        attrs = intent.get("attrs") or {}
        if val is not None or attrs.get("temperature") is not None:
            return "set_temperature"
        if attrs.get("fan_speed") is not None:
            return "set_fan_mode"
    return "set"


def _action_to_cn(action_label: str, dev_name: str) -> str:
    """把 action("on"/"off"/"increase"...) + 设备名 -> 中文动宾短语（关闭客厅灯 / 开启卧室灯）"""
    a = (action_label or "").strip().lower()
    # 特殊设备类型的动作语义
    is_curtain = "窗帘" in dev_name
    is_light = "灯" in dev_name
    is_ac = "空调" in dev_name
    is_tv = "电视" in dev_name
    if a == "off" or a == "turn_off":
        if is_curtain:
            return f"拉上{dev_name}"
        if is_light:
            return f"关闭{dev_name}"
        return f"关闭{dev_name}"
    if a == "on" or a == "turn_on":
        if is_curtain:
            return f"拉开{dev_name}"
        if is_light:
            return f"开启{dev_name}"
        return f"开启{dev_name}"
    if a in ("close_curtain",):
        return f"拉上{dev_name}"
    if a in ("open_curtain",):
        return f"拉开{dev_name}"
    if a == "increase":
        return f"调高{dev_name}"
    if a == "decrease":
        return f"调低{dev_name}"
    if a == "set":
        return f"设置{dev_name}"
    if a == "toggle":
        return f"切换{dev_name}"
    if a == "cool":
        return f"开启{dev_name}制冷"
    if a == "heat":
        return f"开启{dev_name}制热"
    return f"操作{dev_name}"


def _msg_is_detailed(msg: str, device_name: str, action_label_cn: str) -> bool:
    """
    判断 device_controller 返回的 msg 是否是"详细播报"（即已经包含温度/模式/风速/越界说明/已开+属性等）。
    详细播报 → 必须在 _build_compound_reply 中原样优先使用，禁止回退为"已为您设置XX"泛化句。
    """
    if not msg:
        return False
    # 1) 长度防御：泛化句"已为您开启卧室空调/已为您设置客厅空调"一般 11~13 字；详细播报 >= 16 字
    if len(msg) >= 16:
        return True
    # 2) 关键词命中（任何一个即认为是详细播报）——越界说明、属性确认、增减分支、中文数字度
    DETAIL_HINTS = (
        "温度范围", "调至最高", "调至最低", "温度调至", "模式切换", "模式为",
        "风速调至", "风速为", "升高至", "降低至", "已处于开启状态", "已为您将",
        "设置为：", "制冷", "制热", "除湿", "送风", "自动", "档", "度，", "度。",
    )
    if any(h in msg for h in DETAIL_HINTS):
        return True
    # 3) 不等于泛化句模板（兜底检查）
    generic_templates = (
        f"已为您开启{device_name}。",
        f"已为您关闭{device_name}。",
        f"已为您{action_label_cn}{device_name}。",
        f"已为您将{device_name}设置好。",
    )
    return msg not in generic_templates


def _build_compound_reply(records: List[Dict[str, Any]]) -> str:
    """
    【★ 修复版】综合播报：**优先用 device_controller 返回的详细 msg**，
    仅当 msg 空/泛化时才回退自拼。这才是越界文案 / 已开+属性更新文案能送达用户的关键。

    策略分层：
    1) 把 done 记录分成 2 组：
       - detailed_dones：msg 已经是详细播报（直接原样保留整句）
       - generic_dones：  msg 太泛化，需要用 _action_to_cn 自拼
    2) 单条 done + msg 详细 → 直接整句返回原 msg（原 msg 已带句号，不二次拼接）
    3) 多条 done（含 detailed 混 generic）→ detailed 原句 + "，同时" + generic 自拼
    4) already / clarify / fail 附加逻辑保持不变，拼接在其后
    """
    if not records:
        msg = "操作已完成。"
        print(f"[Reply-Safe] _build_compound_reply: 空记录兜底 → {msg!r}")
        return msg

    dones_all = [r for r in records if r.get("kind") == "done"]
    alreadys: List[str] = [r.get("msg") or "" for r in records if r.get("kind") == "already" and r.get("msg")]
    clarifies: List[str] = [r.get("msg") or "" for r in records if r.get("kind") == "clarify" and r.get("msg")]
    fails: List[str] = [r.get("msg") or "" for r in records if r.get("kind") == "fail" and r.get("msg")]

    detailed_msgs: List[str] = []
    generic_actions: List[str] = []

    for r in dones_all:
        dev_name = r.get("device") or "设备"
        act_cn = _action_to_cn(r.get("action_label", ""), dev_name)
        raw_msg = (r.get("msg") or "").strip()
        if _msg_is_detailed(raw_msg, dev_name, act_cn):
            detailed_msgs.append(raw_msg.rstrip("。！？.!?"))
        else:
            generic_actions.append(act_cn)

    parts: List[str] = []

    # (A) done 部分
    # 特殊优化：只有 1 条 done，且是详细播报 → 整句直接用（device_controller 已做过标点处理和友好文案）
    if len(detailed_msgs) == 1 and not generic_actions and not alreadys:
        single = detailed_msgs[0]
        if not (single.endswith("。") or single.endswith("？")):
            single += "。"
        reply = single
        if clarifies or fails:
            tail_parts = []
            if clarifies:
                tail_parts.append("；".join(clarifies))
            if fails:
                tail_parts.append("；".join(fails))
            reply = reply.rstrip("。！？.!?") + "。" + "；".join(tail_parts)
            if not (reply.endswith("。") or reply.endswith("？")):
                reply += "。"
        return reply

    # 否则（多 detailed / 有 generic / 有 already）正常拼接
    if detailed_msgs:
        # 详细播报之间用"，同时"连接
        parts.append("，同时".join(detailed_msgs))
    if generic_actions:
        if len(generic_actions) == 1:
            parts.append(f"已为您{generic_actions[0]}")
        elif len(generic_actions) == 2:
            parts.append(f"已为您{generic_actions[0]}，并{generic_actions[1]}")
        else:
            head = "、".join(generic_actions[:-1])
            parts.append(f"已为您{head}，并{generic_actions[-1]}")

    # (B) already（已是最终状态）
    if len(alreadys) == 1:
        parts.append(alreadys[0].strip("。"))
    elif len(alreadys) >= 2:
        parts.append("，".join(a.strip("。") for a in alreadys))

    # (C) clarify / fail
    if clarifies:
        parts.append("；".join(clarifies))
    if fails:
        parts.append("；".join(fails))

    if not parts:
        msg = "操作已完成。"
        print(f"[Reply-Safe] _build_compound_reply: 空 parts 兜底 → {msg!r}")
        return msg

    reply = "。".join(p.strip() for p in parts if p and p.strip())
    if not (reply.endswith("。") or reply.endswith("？")):
        reply += "。"

    # —— ★ 终极防线：任何情况下 reply 非空且非空白 ——
    if not reply or not reply.strip():
        reply = "操作已完成。"
        print(f"[Reply-Safe] _build_compound_reply: 拼接后仍空，强制兜底 → {reply!r}")
    return reply


async def _handle_device_query(text: str, session_id: str) -> Dict[str, Any]:
    """
    【设备查询轨】：只读 DEVICE_STATES，无 MQTT（除非 conditional 条件命中且需要控制）。

    3 类查询：
      - single_power 单设备开关状态
      - list_on      列表/聚合查询（所有亮/开的灯或指定类型设备）
      - attribute    属性数值查询（温度/风速/模式 等，直接读内存值，不走 RAG）
    可选 conditional 条件控制："看看卧室空调开着没，没开就打开"。
    """
    print(f"[SmartBrain] → 设备查询轨（仅读状态，禁止默认发MQTT）")

    # 构建设备状态上下文（Debug 用）
    device_context = build_device_states_context()
    print(f"[SmartBrain] 设备状态: {device_context}")

    # 1) 解析查询结构
    query = parse_device_query(text, session_id)
    print(f"[SmartBrain] 查询结构: {query}")

    kind = query["query_kind"]
    commands_executed = []
    reply_text = ""
    success = True
    device_executed = False

    # ------------------------------------------------------------------
    # 3-A-1. 单设备开关状态
    # ------------------------------------------------------------------
    if kind == "single_power":
        dev_id = query["target_device_id"]
        dev_name = query["target_device_name"] or dev_id or "设备"
        if not dev_id or dev_id not in DEVICE_STATES:
            reply_text = f"抱歉，我没有找到您说的{dev_name}，可以换个说法吗？"
            success = False
        else:
            state = DEVICE_STATES[dev_id]
            power = state.get("power", "off")
            # 记录上下文（用于后续"调高2度"这类无设备名指令）
            if session_id and dev_id:
                set_last_device(session_id, dev_id)

            # ---- 条件控制分支 B ----
            cond = query.get("conditional")
            if cond:
                expected = cond["expected_power"]
                then_act = cond["then_action"]
                if power == expected:
                    # 条件命中：需要执行 then_action
                    dev_type = state.get("type", "")
                    action = "on" if then_act == "on" else "off"  # 与 device_controller 内置 action 语义对齐
                    cmd_intent = {
                        "device_id": dev_id,
                        "device_name": dev_name,
                        "action": action,
                        "value": None,
                        "value_type": None,
                        "attrs": {},
                    }
                    exec_res = await execute_command(cmd_intent)
                    device_executed = exec_res.get("success", False)
                    commands_executed.append({
                        "device_id": dev_id,
                        "action": action,
                        "value": None,
                    })
                    on_cn = "开启状态" if then_act == "on" else "关闭状态"
                    was_cn = "关闭状态" if then_act == "on" else "开启状态"
                    do_cn = "打开" if then_act == "on" else "关闭"
                    if exec_res.get("success"):
                        reply_text = f"{dev_name}原为{was_cn}，已为您{do_cn}"
                    else:
                        reply_text = f"{dev_name}原为{was_cn}，尝试{do_cn}失败，请稍后再试"
                else:
                    # 条件不命中：设备当前已经处于用户"最终希望"的状态
                    already_cn = "开启状态" if then_act == "on" else "关闭状态"
                    reply_text = f"{dev_name}已经是{already_cn}"
                return {
                    "success": success,
                    "reply": reply_text,
                    "intent": "device_query",
                    "device_executed": device_executed,
                    "commands": commands_executed,
                    "error": None,
                }

            # ---- 纯查询分支 A ----
            if power == "on":
                reply_text = f"{dev_name}目前是开启状态"
            else:
                reply_text = f"{dev_name}目前是关闭状态"
            return {
                "success": True,
                "reply": reply_text,
                "intent": "device_query",
                "device_executed": False,
                "commands": [],
                "error": None,
            }

    # ------------------------------------------------------------------
    # 3-A-2. 列表/聚合查询（哪些灯亮着 / 哪些设备开着）
    # ------------------------------------------------------------------
    if kind == "list_on":
        dev_type = query["target_device_type"] or "all"
        target_room = query["target_room"]  # "" 表示全屋
        # 房间名双模式兼容（中/英文均可），避免空字符串导致设备名丢失（如"客厅灯"->"灯"）
        _ROOM_CN_BI = {
            "bedroom": "卧室", "living_room": "客厅", "kitchen": "厨房",
            "bathroom": "卫生间", "study": "书房", "dining": "餐厅", "balcony": "阳台",
            "卧室": "卧室", "客厅": "客厅", "厨房": "厨房", "卫生间": "卫生间",
            "书房": "书房", "餐厅": "餐厅", "阳台": "阳台",
        }
        _TYPE_CN_JOIN = {"light": "灯", "ac": "空调", "tv": "电视",
                         "curtain": "窗帘", "speaker": "音响", "all": "设备"}
        # 遍历 DEVICE_STATES，找出符合 type 与 room 的设备
        matched: List[Tuple[str, Dict[str, Any]]] = []
        for dev_id, state in DEVICE_STATES.items():
            s_type = state.get("type", "")
            s_room_cn = _ROOM_CN_BI.get(state.get("room", ""), state.get("room", ""))
            # 类型过滤
            if dev_type != "all" and s_type != dev_type:
                continue
            # 房间过滤（用户指定了房间才过滤）
            if target_room and s_room_cn != target_room:
                continue
            matched.append((dev_id, state))

        if not matched:
            scope = target_room or "家里"
            t_cn = _TYPE_CN_JOIN.get(dev_type, "设备")
            reply_text = f"当前{scope}没有安装{t_cn}"
            return {
                "success": True,
                "reply": reply_text,
                "intent": "device_query",
                "device_executed": False,
                "commands": [],
                "error": None,
            }

        # 分别收集 on / off 的【完整中文名称】（卧室灯、客厅灯，不是短语）
        on_names: List[str] = []
        off_names: List[str] = []
        for dev_id, state in matched:
            s_type = state.get("type", "")
            s_room_cn = _ROOM_CN_BI.get(state.get("room", ""), state.get("room", ""))
            t_cn = _TYPE_CN_JOIN.get(s_type, "设备")
            name = s_room_cn + t_cn  # 例如："卧室灯" / "客厅空调"
            if s_type == "curtain":
                if state.get("state") == "open":
                    on_names.append(name)
                else:
                    off_names.append(name)
            else:
                if state.get("power") == "on":
                    on_names.append(name)
                else:
                    off_names.append(name)

        # =============== 文本动态组装（正向查询，重点回答开启的设备）===============
        scope = target_room or "家里"
        t_cn = _TYPE_CN_JOIN.get(dev_type, "设备")
        total = len(matched)
        opened = len(on_names)
        closed = len(off_names)

        # 正向话术描述：对灯说"亮着/亮"，对其他设备说"开着"
        verb_on = "亮着" if dev_type == "light" else "开着"
        verb_all_on = "都亮着" if dev_type == "light" else "都开着"
        adj_all_off = "都是关闭的"  # 统一用"关闭的"

        if opened == total and total > 0:
            # 情况 C：全部开启
            reply_text = f"当前{scope}所有的{t_cn}{verb_all_on}"
        elif opened == 0 and closed > 0:
            # 情况 B：全部关闭
            reply_text = f"当前{scope}所有的{t_cn}{adj_all_off}"
        else:
            # 情况 A：部分开启 —— 重点回答【正亮着/开着】的设备，严禁再把关闭设备拼入
            # 多个设备用顿号分隔，末尾加"正亮着。"/"正开着。"
            devices_str = "、".join(on_names)
            reply_text = f"目前{devices_str}正{verb_on}"

        return {
            "success": True,
            "reply": reply_text,
            "intent": "device_query",
            "device_executed": False,
            "commands": [],
            "error": None,
        }

    # ------------------------------------------------------------------
    # 3-A-3. 属性数值查询（直接读 DEVICE_STATES，严禁走 RAG）
    # ------------------------------------------------------------------
    if kind == "attribute":
        dev_id = query["target_device_id"]
        dev_name = query["target_device_name"] or dev_id or "设备"
        field = query["target_attribute"]
        attr_cn = query["target_attribute_cn"] or field or "属性"

        if not dev_id or dev_id not in DEVICE_STATES:
            reply_text = f"抱歉，我没有找到您说的{dev_name}，无法查询{attr_cn}"
            return {
                "success": False,
                "reply": reply_text,
                "intent": "device_query",
                "device_executed": False,
                "commands": [],
                "error": "device not found",
            }
        if not field:
            reply_text = f"抱歉，我无法理解您要查询{dev_name}的什么属性，可以换个说法吗？"
            return {
                "success": False,
                "reply": reply_text,
                "intent": "device_query",
                "device_executed": False,
                "commands": [],
                "error": "attribute unknown",
            }

        # 记录上下文
        if session_id and dev_id:
            set_last_device(session_id, dev_id)

        state = DEVICE_STATES[dev_id]
        raw_val = state.get(field, None)
        if raw_val is None or raw_val == "":
            reply_text = f"{dev_name}目前没有记录{attr_cn}"
            return {
                "success": True,
                "reply": reply_text,
                "intent": "device_query",
                "device_executed": False,
                "commands": [],
                "error": None,
            }

        # 中文口语化格式化
        dev_type = state.get("type", "")
        if field == "temperature":
            reply_text = f"{dev_name}当前设定温度是 {raw_val} 度"
        elif field == "fan_speed":
            fan_cn = _AC_FAN_CN.get(raw_val, str(raw_val) + ("档" if isinstance(raw_val, int) else ""))
            reply_text = f"{dev_name}当前{attr_cn}为{fan_cn}"
        elif field == "mode":
            mode_cn = _AC_MODE_CN.get(raw_val, str(raw_val))
            reply_text = f"{dev_name}当前运行在{mode_cn}模式"
        elif field == "swing_mode":
            if raw_val in ("on", "swing", True, 1):
                reply_text = f"{dev_name}当前扫风是开启的"
            else:
                reply_text = f"{dev_name}当前扫风是关闭的"
        elif field == "brightness":
            reply_text = f"{dev_name}当前{attr_cn}是 {raw_val}%"
        elif field == "volume":
            reply_text = f"{dev_name}当前{attr_cn}是 {raw_val}"
        elif field == "channel":
            reply_text = f"{dev_name}当前频道是 {raw_val}"
        else:
            reply_text = f"{dev_name}当前{attr_cn}是 {raw_val}"

        return {
            "success": True,
            "reply": reply_text,
            "intent": "device_query",
            "device_executed": False,
            "commands": [],
            "error": None,
        }

    # 兜底
    reply_text = "抱歉，我没有理解您的查询请求，可以换个说法吗？"
    return {
        "success": False,
        "reply": reply_text,
        "intent": "device_query",
        "device_executed": False,
        "commands": [],
        "error": "query kind unknown",
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
    - ★ 澄清反问保护：need_clarification=True 时禁止场景消息覆盖/拼接反问文案
    """
    result = await process(text, session_id)

    # ========== 澄清反问 高优先级保护：如果已经要用户选房间，绝对不能用场景消息覆盖 ==========
    if result.get("need_clarification"):
        return result

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

    # —— ★ process_with_scene 最终出口：全局 Reply 强制非空 ——
    r = result.get("reply")
    if not isinstance(r, str) or not r.strip():
        result["reply"] = "操作已完成。"
        print(f"[Reply-Safe] process_with_scene: 最终 reply 为空/空白，兜底 → {result['reply']!r}")
    return result