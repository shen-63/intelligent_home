import json
import re
import requests
from typing import Dict, Any, Optional

from config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    SILICONFLOW_MODEL,
    LLM_TIMEOUT,
    LLM_MAX_RETRIES,
)
from nlu_parser import DEVICE_MAP, ACTION_PATTERNS, VALUE_PATTERNS, get_device_risk

# 构建 LLM 的 system prompt，严格约束输出格式
_DEVICE_TABLE = "\n".join(
    f'  - "{raw}" → "{dev_id}"'
    for raw, dev_id in DEVICE_MAP
)

_ACTION_TABLE = "\n".join(
    f'  - 匹配 "{p[1:-1]}..." → "{act}"'
    for p, act in ACTION_PATTERNS
)

SYSTEM_PROMPT = f"""你是一个智能家居语音助手的意图解析器。你的任务是将用户的自然语言指令解析为严格的 JSON 格式。

## 已知设备映射表
用户可能提到的设备及对应的标准 device_id：
{_DEVICE_TABLE}

## 动作映射规则
从用户文本中识别以下动作之一：
{_ACTION_TABLE}
  - 如果没有匹配，action 为 "unknown"

## 数值提取
- 温度："26度" → value=26, value_type="temperature"
- 亮度百分比："80%" / "最亮" / "最暗" → value=80/100/10, value_type="brightness"
- 窗帘状态："全开" / "全关" → value="全开"/"全关", value_type="curtain_state"

## 环境感知模式（重要 - 处理模糊环境指令）
当用户表达环境感受并包含房间名称时，必须优先匹配对应房间的设备：
- "客厅有点热/太热了/很闷" → device_id="living_room_ac", action="on", 模式=cool
- "卧室有点冷/太冷了" → device_id="bedroom_ac", action="on", 模式=heat
- "客厅有点暗/太黑了" → device_id="living_room_light", action="on"
- "卧室太亮了/晃眼" → device_id="bedroom_light", action="off"
- 关键规则：用户提到的房间名（客厅/卧室/大厅/房间）必须映射到对应房间的设备，不要默认选卧室

## 输出格式（严格 JSON，无其他文字）
{{
  "success": true/false,
  "msg": "解析成功" 或 错误描述,
  "device_id": "设备ID",
  "device_name": "原始设备名称",
  "action": "on/off/increase/decrease/set/unknown",
  "value": 数值或null,
  "value_type": "temperature/brightness/curtain_state/null",
  "attrs": {{"mode": "cool/heat/..."}} 或 {{}},
  "confidence": 0.0-1.0,
  "need_clarification": true/false
}}

## 规则
1. 只输出 JSON，不要任何解释或 markdown。
2. 如果无法识别设备或动作，success=false，其余字段尽可能填写。
3. device_id 必须严格使用映射表中的英文 ID，不要自己编造。
4. 对于环境感知指令（如"客厅有点热"），必须将模式信息放入 attrs.mode 字段。

## 消歧策略（重要 — 按设备风险分级）
当用户没有明确说房间时，根据设备类型采用不同策略：

### 低风险设备 (空调/灯/电视) → 自动消歧
- "打开空调" → 默认卧室：device_id="bedroom_ac", need_clarification=false
- "开灯" → 默认卧室：device_id="bedroom_light", need_clarification=false
- 选择卧室的理由：人们在卧室停留时间最长，是最常用的生活空间

### 高风险设备 (窗帘) → 必须询问用户
- "拉开窗帘" → device_id="curtain", need_clarification=true
- 原因：窗帘涉及隐私，操作错房间体验很差

### 极端模糊 → 必须询问
- "把那个调一下" / "弄一下" / "那个" → need_clarification=true
- 只有在用户明确说了设备和动作时才能判断，否则一律 need_clarification=true
"""


def parse_intent_llm(text: str) -> Dict[str, Any]:
    """
    使用硅基流动大模型解析意图。
    输出与 nlu_parser.parse_intent() 完全相同的 schema。

    Returns:
        {
            "success": bool,
            "msg": str,
            "method": "llm",
            "confidence": float,
            "need_clarification": bool,
            "intent": {...} | None
        }
    """
    if not SILICONFLOW_API_KEY:
        return {
            "success": False,
            "msg": "未配置 SILICONFLOW_API_KEY，无法调用 LLM",
            "method": "llm",
            "confidence": 0.0,
            "need_clarification": False,
            "intent": None,
        }

    raw_output = _call_llm(text)
    if raw_output is None:
        return {
            "success": False,
            "msg": "LLM 调用失败，请检查 API Key 或网络",
            "method": "llm",
            "confidence": 0.0,
            "need_clarification": False,
            "intent": None,
        }

    parsed = _parse_llm_output(raw_output, text)
    parsed["method"] = "llm"
    return parsed


def _call_llm(text: str) -> Optional[str]:
    """调用硅基流动 API，返回原始文本，失败返回 None"""
    url = f"{SILICONFLOW_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SILICONFLOW_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                continue
            print(f"[LLM] HTTP error (attempt {attempt}): {e}")
        except requests.exceptions.RequestException as e:
            print(f"[LLM] Request error (attempt {attempt}): {e}")
        except (KeyError, IndexError, ValueError) as e:
            print(f"[LLM] Parse error: {e}")
            return None
    return None


def _parse_llm_output(raw: str, original_text: str) -> Dict[str, Any]:
    """将 LLM 输出解析为标准意图 schema，对 LLM 输出做容错处理"""
    data = None

    # 尝试直接解析，容忍 LLM 输出的额外字段
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

    if data is None:
        return _fallback_no_parse(raw, original_text)

    # 从多个可能的字段名中提取值（容错 LLM 输出变体）
    device_id = data.get("device_id", "") or data.get("device", "")
    device_name = data.get("device_name", device_id) or data.get("device", device_id)
    action = data.get("action", "unknown")
    if not action:
        action = data.get("intent", "unknown")
    value = data.get("value")
    if value is None:
        # 尝试从其他字段提取数值
        for key in ("temperature", "target_temperature", "brightness", "level"):
            if data.get(key) is not None:
                value = data[key]
                break
    value_type = data.get("value_type")
    confidence = float(data.get("confidence", data.get("conf", 0.5)))
    need_clarification = bool(data.get("need_clarification", data.get("need_clarify", False)))

    # 关键容错：如果有有效的 device_id 和 action，即使 LLM 说 success=false 也视为成功
    known_devices = [d[1] for d in DEVICE_MAP]
    if device_id and device_id in known_devices and action and action != "unknown":
        success = True
        msg = data.get("msg", "LLM 解析成功（容错模式）")
    else:
        success = bool(data.get("success", False))
        msg = data.get("msg", "")
        if not success:
            msg = msg or f"LLM 未能识别有效设备，device_id={device_id}"

    # 强制分级消歧：高风险设备（窗帘）即使 LLM 自动消歧，也必须询问用户
    if success and not need_clarification:
        curtain_ids = ("curtain", "bedroom_curtain", "living_room_curtain")
        if device_id in curtain_ids:
            device_id = "curtain"
            device_name = "窗帘"
            need_clarification = True
            print(f"[LLM] 强制高风险设备消歧: {device_id} → need_clarification=true")

    intent = None
    if success and device_id and action != "unknown":
        intent = {
            "raw_text": original_text,
            "device_id": device_id,
            "device_name": device_name,
            "action": action,
            "value": value,
            "value_type": value_type,
            "attrs": data.get("attrs", {}) or {},
        }

    return {
        "success": success,
        "msg": msg,
        "method": "llm",
        "confidence": confidence,
        "need_clarification": need_clarification and success,
        "intent": intent,
    }


def _fallback_no_parse(raw: str, original_text: str) -> Dict[str, Any]:
    """LLM 输出完全无法解析时的降级处理"""
    return {
        "success": False,
        "msg": f"LLM 输出格式异常，原始输出前200字符: {raw[:200]}",
        "method": "llm",
        "confidence": 0.0,
        "need_clarification": False,
        "intent": None,
    }
