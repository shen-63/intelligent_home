from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import asyncio
import re

# 设备状态防竞态锁（asyncio.Lock，配合全链路 async/await）
state_lock = asyncio.Lock()

DEVICE_STATES: Dict[str, Dict[str, Any]] = {
    "bedroom_ac":         {"type": "ac",      "room": "卧室", "power": "off", "temperature": 26, "mode": "cooling", "swing": "off", "fan_speed": 2},
    "living_room_ac":     {"type": "ac",      "room": "客厅", "power": "off", "temperature": 26, "mode": "cooling", "swing": "off", "fan_speed": 2},
    "bedroom_light":      {"type": "light",   "room": "卧室", "power": "off", "brightness": 80},
    "living_room_light":  {"type": "light",   "room": "客厅", "power": "off", "brightness": 100},
    "bedroom_curtain":    {"type": "curtain", "room": "卧室", "state": "closed"},
    "living_room_curtain":{"type": "curtain", "room": "客厅", "state": "closed"},
    "bedroom_tv":         {"type": "tv",      "room": "卧室", "power": "off", "channel": 1},
    "living_room_tv":     {"type": "tv",      "room": "客厅", "power": "off", "channel": 1},
}

AC_MODE_NAMES: Dict[str, str] = {
    "cooling": "制冷", "heating": "制热", "dehumidify": "除湿", "fan": "送风", "auto": "自动",
    "cool": "制冷", "heat": "制热", "dry": "除湿",
}
AC_SWING_NAMES: Dict[str, str] = {
    "vertical": "上下扫风", "horizontal": "左右扫风", "off": "关闭扫风",
    "up_down": "上下扫风", "left_right": "左右扫风", "stop": "关闭扫风",
}

# 属性值规范化映射（统一转换为前端能识别的标准格式）
AC_MODE_NORM: Dict[str, str] = {
    "cool": "cooling", "heat": "heating", "dry": "dehumidify",
    "cooling": "cooling", "heating": "heating", "dehumidify": "dehumidify",
    "fan": "fan", "auto": "auto",
}
AC_SWING_NORM: Dict[str, str] = {
    "on": "horizontal", "left_right": "horizontal", "horizontal": "horizontal",
    "up_down": "vertical", "vertical": "vertical", "off": "off", "stop": "off",
}
AC_FAN_NAMES: Dict[str, str] = {
    "quiet": "静音档", 1: "1档", 2: "2档", 3: "3档", 4: "4档", 5: "5档", "auto": "自动风",
}

DEFAULT_ROOM_MAP: Dict[str, str] = {
    "air_conditioner": "bedroom_ac",
    "light": "bedroom_light",
    "curtain": "bedroom_curtain",
    "tv": "bedroom_tv",
}

# 设备类型 -> 中文显示名
DEVICE_TYPE_NAMES: Dict[str, str] = {
    "ac": "空调",
    "light": "灯",
    "curtain": "窗帘",
    "tv": "电视",
}

# 中文房间名 -> MQTT Topic 使用的英文 slug
ROOM_ENG_MAP: Dict[str, str] = {"卧室": "bedroom", "客厅": "living_room"}
# 英文 slug -> 中文名（给 impl 上层兜底拼装回复文案用）
ROOM_SLUG_TO_CN: Dict[str, str] = {v: k for k, v in ROOM_ENG_MAP.items()}


def _build_confirmation(intent: Dict[str, Any], fallback_reason: str = "") -> str:
    """
    【需求 3】确认文案强制非空兜底（只有在 impl 抛异常/返回空 message 时才用）。

    会先从 intent.device_id / DEVICE_STATES 中取出"卧室空调/客厅窗帘"这种中文显示名，
    再按 intent.action 拼出精准 confirmation：
      - on  / turn_on  → "已为您开启{location}{device_name}。"
      - off / turn_off → "已为您关闭{location}{device_name}。"
      - set / set_temperature 且 value_type=temperature → "已为您把{display}温度调至 {value} 度。"
      - set 其它 → "已为您设置好{display}。"
    当所有信息都缺失时，才最后返回 "操作已完成。"。
    """
    try:
        dev_id = ""
        action = ""
        value = None
        value_type = None
        if isinstance(intent, dict):
            dev_id = str(intent.get("device_id") or "")
            action = str(intent.get("action") or "")
            value = intent.get("value")
            value_type = intent.get("value_type")

        # 1) 解析中文显示名 display（优先 DEVICE_STATES.room）
        display = ""
        if dev_id and dev_id in DEVICE_STATES:
            st = DEVICE_STATES[dev_id]
            room_cn = st.get("room") or ""
            type_cn = DEVICE_TYPE_NAMES.get(st.get("type") or "", "")
            if room_cn and type_cn:
                display = room_cn + type_cn
            elif isinstance(intent, dict) and isinstance(intent.get("device_name"), str):
                display = intent["device_name"]
        if not display and isinstance(intent, dict) and isinstance(intent.get("device_name"), str):
            display = intent["device_name"]
        if not display:
            # 最后一招：从 device_id 中拆 bedroom/living_room
            parts = dev_id.replace(".all", "").split("_")
            if parts:
                slug = parts[0]
                type_short = "_".join(parts[1:]) if len(parts) > 1 else ""
                room_cn = ROOM_SLUG_TO_CN.get(slug, slug)
                type_cn = DEVICE_TYPE_NAMES.get(type_short, type_short)
                if room_cn and type_cn:
                    display = room_cn + type_cn
                elif type_cn:
                    display = type_cn

        # 2) 动作归一化（和 _execute_command_impl 保持一致）
        if action == "turn_on":
            action = "on"
        elif action == "turn_off":
            action = "off"
        elif action in ("set_temperature", "set_value", "adjust", "adjust_ac",
                        "set_fan_mode", "set_hvac_mode", "set_swing_mode", "set_ac"):
            action = "set"
        elif action in ("open_curtain",):
            action = "on"
        elif action in ("close_curtain",):
            action = "off"

        # 3) 按 display + action 拼
        if display:
            if action in ("on", "open_curtain"):
                return f"已为您开启{display}。"
            if action in ("off", "close_curtain"):
                return f"已为您关闭{display}。"
            if action == "set":
                if value_type == "temperature" and isinstance(value, (int, float)):
                    return f"已为您将{display}温度调至 {int(value)} 度。"
                if value is not None:
                    return f"已为您将{display}设置为 {value}。"
                return f"已为您设置好{display}。"
            if action == "increase":
                return f"已为您调高{display}。"
            if action == "decrease":
                return f"已为您调低{display}。"
            # action 未知但有设备名 → 不说"操作已完成"，而是给个更友好的完成提示
            return f"{display}操作已完成。"

        # 4) 最后一道：真的什么都取不到
        print(f"[Reply-Safe] _build_confirmation: 取不到设备名/动作，原因={fallback_reason!r}，降级为'操作已完成。'")
        return "操作已完成。"
    except Exception as e:
        print(f"[Reply-Safe] _build_confirmation 异常: {e}，最终兜底为'操作已完成。'")
        return "操作已完成。"

# ======================================================================
# 【★ 未注册设备检测 + 显式房间优先解析】
# 解决：输入"打开书房灯"模糊回退成卧室灯的Bug。
# 原则：用户显式提到"X房/厅 + Y(灯/空调/窗帘/电视)"时，若系统无此具体设备，
#       直接报"未找到"，绝对不默认回退到其他房间。
# ======================================================================

# 【用户可能提到的所有空间词】→ 规范化的 room_cn key（与 DEVICE_STATES["room"] 对齐）
#   注意：即使某个房间当前未注册设备，也要识别出这个词，避免把"书房灯"当"灯"回退卧室。
_EXPLICIT_ROOM_KEYWORDS: List[Tuple[str, str]] = [
    ("主卧", "卧室"), ("次卧", "卧室"), ("卧室", "卧室"), ("房间", "卧室"), ("屋里", "卧室"), ("里面", "卧室"),
    ("客厅", "客厅"), ("大厅", "客厅"), ("起居室", "客厅"), ("厅里", "客厅"),
    ("书房", "书房"), ("书斋", "书房"),
    ("厨房", "厨房"), ("后厨", "厨房"), ("饭厅", "餐厅"), ("餐厅", "餐厅"),
    ("卫生间", "卫生间"), ("厕所", "卫生间"), ("洗手间", "卫生间"), ("浴室", "卫生间"), ("卫浴", "卫生间"),
    ("阳台", "阳台"), ("露台", "阳台"),
    ("儿童房", "儿童房"), ("小孩房", "儿童房"), ("婴儿房", "儿童房"),
    ("书房", "书房"), ("阁楼", "阁楼"), ("地下室", "地下室"), ("玄关", "玄关"), ("走廊", "走廊"),
    ("家里", ""), ("全屋", ""), ("整个屋子", ""), ("所有房间", ""),
]

# 设备类型中文别称 -> dev_type key
_EXPLICIT_TYPE_KEYWORDS: List[Tuple[str, str]] = [
    ("空调", "ac"), ("冷气", "ac"), ("冷气机", "ac"), ("中央空调", "ac"),
    ("灯", "light"), ("灯光", "light"), ("电灯", "light"), ("照明灯", "light"),
    ("窗帘", "curtain"), ("窗纱", "curtain"), ("帘子", "curtain"),
    ("电视", "tv"), ("电视机", "tv"), ("彩电", "tv"),
]

def _extract_explicit_room_and_type(raw_text: str) -> Optional[Tuple[str, str, str, str]]:
    """
    从用户原始文本中提取【显式】房间+设备类型的组合。
    返回 (room_cn_key, room_original_word, dev_type, type_original_word)；若未同时提取到两者返回 None。
    例："打开书房灯" → ("书房", "书房", "light", "灯")
        "厨房空调调到24度" → ("厨房", "厨房", "ac", "空调")
        "打开灯" → None（没提房间）
    """
    if not raw_text:
        return None
    room_hit: Optional[Tuple[int, str, str]] = None  # (pos, room_cn, original)
    type_hit: Optional[Tuple[int, str, str]] = None  # (pos, dev_type, original)
    for original, room_cn in _EXPLICIT_ROOM_KEYWORDS:
        pos = raw_text.find(original)
        if pos >= 0 and (room_hit is None or pos < room_hit[0]):
            room_hit = (pos, room_cn, original)
    for original, dt in _EXPLICIT_TYPE_KEYWORDS:
        pos = raw_text.find(original)
        if pos >= 0 and (type_hit is None or pos < type_hit[0]):
            type_hit = (pos, dt, original)
    if room_hit and type_hit and room_hit[1]:  # room_cn 不为空
        return room_hit[1], room_hit[2], type_hit[1], type_hit[2]
    return None


def _device_exists_with_room_and_type(room_cn: str, dev_type: str) -> bool:
    """检查 DEVICE_STATES 中是否存在 room=room_cn 且 type=dev_type 的设备"""
    for dev in DEVICE_STATES.values():
        if dev.get("room") == room_cn and dev.get("type") == dev_type:
            return True
    return False


def _resolve_device_id(intent: Dict[str, Any]) -> str:
    dev_id = intent["device_id"]
    if dev_id in DEVICE_STATES:
        return dev_id
    return DEFAULT_ROOM_MAP.get(dev_id, dev_id)


def _display_name(state: Dict[str, Any], dev_name: str) -> str:
    """生成无重复的设备显示名，如 '卧室空调' 而非 '卧室卧室空调'"""
    type_name = DEVICE_TYPE_NAMES.get(state["type"], dev_name)
    if dev_name and type_name in dev_name:
        return dev_name
    return f"{state['room']}{type_name}"


# =============================================================
# MQTT 辅助函数
# =============================================================

def get_mqtt_meta(device_id: str) -> Dict[str, Any]:
    """
    根据 device_id（如 bedroom_ac、bedroom_light、light.all）
    导出 MQTT Topic 所需的英文元数据。

    Returns {
        "room": "bedroom" | "living_room" | None,
        "type": "ac" | "light" | "curtain" | "tv" | None,
        "multi_type": "ac" | "light" | "curtain" | "tv" | None,  # 批量操作时的设备类型
    }
    """
    from config import MQTT_TOPIC_PREFIX

    # 1) 精确设备：如 bedroom_ac -> type=ac, room=bedroom
    if device_id in DEVICE_STATES:
        info = DEVICE_STATES[device_id]
        cn_room = info.get("room", "")
        room = ROOM_ENG_MAP.get(cn_room, cn_room)
        return {
            "room": room,
            "type": info.get("type"),
            "multi_type": None,
        }

    # 2) HA entity 风格：climate.bedroom_ac / light.bedroom_light
    if "." in device_id and not device_id.startswith("all_"):
        _, tail = device_id.split(".", 1)
        if tail in DEVICE_STATES:
            info = DEVICE_STATES[tail]
            cn_room = info.get("room", "")
            room = ROOM_ENG_MAP.get(cn_room, cn_room)
            return {"room": room, "type": info.get("type"), "multi_type": None}

    # 3) 批量目标：light.all / climate.all / 所有灯
    from main import ALL_TARGETS_MAP  # 延迟导入，避免循环
    if device_id in ALL_TARGETS_MAP:
        mt = ALL_TARGETS_MAP[device_id]
        return {"room": None, "type": None, "multi_type": mt}
    if device_id.endswith(".all"):
        short = device_id[:-4]  # light.all -> light
        type_map = {"climate": "ac", "cover": "curtain", "media_player": "tv",
                    "light": "light", "ac": "ac", "curtain": "curtain", "tv": "tv"}
        return {"room": None, "type": None, "multi_type": type_map.get(short)}

    # 兜底：从 device_id 中猜
    low = device_id.lower()
    if "bedroom" in low:
        room = "bedroom"
    elif "living" in low or "living_room" in low:
        room = "living_room"
    else:
        room = None
    dt: Optional[str] = None
    if "ac" in low or "climate" in low:
        dt = "ac"
    elif "light" in low or "lamp" in low:
        dt = "light"
    elif "curtain" in low or "cover" in low:
        dt = "curtain"
    elif "tv" in low or "media" in low:
        dt = "tv"
    return {"room": room, "type": dt, "multi_type": None}


async def apply_mqtt_state_update(device_id: str, new_state: Dict[str, Any]) -> bool:
    """
    当 MQTT broker 推送 home/{room}/{type}/state 反馈时，
    异步安全地（带 state_lock）把字段写回 DEVICE_STATES[device_id]。
    """
    async with state_lock:
        # 兼容 mqtt_client 反推的 entity_id 格式 climate.bedroom_ac
        dev_id = device_id
        if "." in device_id:
            _, tail = device_id.split(".", 1)
            if tail in DEVICE_STATES:
                dev_id = tail
        if dev_id not in DEVICE_STATES:
            return False
        state = DEVICE_STATES[dev_id]
        if not isinstance(new_state, dict):
            return False

        # 规范化别名：power（AC/Light/TV 通用） 也兼容 "state": "ON"（MQTT 风格）
        if "power" in new_state:
            v = new_state["power"]
            if isinstance(v, str):
                state["power"] = "on" if v.lower() in ("on", "true", "1") else "off"
            else:
                state["power"] = "on" if bool(v) else "off"
        if "state" in new_state and state.get("type") == "curtain":
            vs = str(new_state["state"]).lower()
            state["state"] = "open" if vs in ("open", "1", "on") else "closed"
        elif "state" in new_state and state.get("type") != "curtain":
            vs = str(new_state["state"]).lower()
            if vs in ("on", "off", "open", "closed"):
                state["power"] = "on" if vs in ("on", "open") else "off"
        if "temperature" in new_state:
            try:
                state["temperature"] = int(new_state["temperature"])
            except Exception:
                pass
        if "mode" in new_state or "hvac_mode" in new_state:
            mode = new_state.get("mode") or new_state.get("hvac_mode")
            mode_map = {"cool": "cooling", "heat": "heating", "dry": "dehumidify"}
            m = mode_map.get(str(mode).lower(), mode)
            if m:
                state["mode"] = m
        if "fan_speed" in new_state or "fan" in new_state:
            fan = new_state.get("fan_speed") or new_state.get("fan")
            if fan is not None:
                state["fan_speed"] = fan
        if "swing" in new_state or "swing_mode" in new_state:
            swing = new_state.get("swing") or new_state.get("swing_mode")
            s = AC_SWING_NORM.get(str(swing).lower(), swing)
            if s:
                state["swing"] = s
        if "brightness" in new_state and state.get("type") == "light":
            try:
                state["brightness"] = int(new_state["brightness"])
            except Exception:
                pass
        if "channel" in new_state and state.get("type") == "tv":
            try:
                state["channel"] = int(new_state["channel"])
            except Exception:
                pass

        return True


_CN_NUM_TABLE = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100,
}

def _parse_chinese_number(text: str) -> Optional[int]:
    """极简中文数字解析：只覆盖 1~100 范围内空调温度常用组合（二十四=24 / 十八=18 / 一百=100）。"""
    s = text.strip()
    if not s:
        return None
    # 纯阿拉伯（保险兜底）
    if s.isdigit():
        return int(s)
    total = 0
    current = 0
    # 处理"二十四"这种十位数：二 + 十 + 四 → 2*10 + 4
    for ch in s:
        if ch not in _CN_NUM_TABLE:
            return None
        v = _CN_NUM_TABLE[ch]
        if v == 10:
            # 十：前面有数字（一~九）则乘，否则默认 10
            current = 1 if current == 0 else current
            total += current * 10
            current = 0
        elif v == 100:
            current = 1 if current == 0 else current
            total += current * 100
            current = 0
        else:
            current = v
    total += current
    return total if total > 0 else None


def _extract_any_temperature_degree(text: str) -> Optional[int]:
    """
    从任意文本中提取"X度/℃/°C/°"的温度值，X 同时支持阿拉伯数字和中文数字。
    返回 None 表示未找到。只取匹配到的第一个温度值。
    """
    if not text:
        return None
    # 1) 阿拉伯数字优先（最快命中 50/5/24）
    m = re.search(r'(\d+)\s*(?:度|℃|°C|°)', text)
    if m:
        return int(m.group(1))
    # 2) 中文数字："(一到十的组合)度"
    m2 = re.search(r'([零〇一二两三四五六七八九十百]+)\s*(?:度|℃|°C|°)', text)
    if m2:
        return _parse_chinese_number(m2.group(1))
    return None


async def _execute_ac(state: Dict[str, Any], display: str, action: str,
                      value: Any, value_type: str, attrs: Dict[str, Any],
                      dev_id: str,
                      control_mode: str = "absolute",
                      delta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """空调多属性控制：支持【绝对量设置】和【增量调节（调高/调小）】两种模式。

    - absolute 模式：行为不变（打开/关闭/设置温度、模式、扫风、风力）
    - delta 模式：优先处理，读取当前值 → 加 delta → clamp 边界 → 写状态 → 生成形如
                  "已将卧室空调温度调高 2 度，当前为 27 度" 的回复。
    """
    changes: List[str] = []

    # ==================== 💥 增量控制（最高优先级） ====================
    if control_mode == "delta" and delta and isinstance(delta, dict):
        attr_field = delta.get("attr_field")
        attr_cn = delta.get("attr_cn") or attr_field
        delta_value = int(delta.get("delta_value", 0))
        if delta_value == 0:
            return {
                "success": False, "code": 400,
                "message": f"增量值为 0，未对{display}做任何调整",
                "resolved_device_id": dev_id, "state_snapshot": state.copy(),
            }

        # 先保证空调是开的（增量调节说明用户在使用，不开没意义）
        power_was_off = (state.get("power") != "on")
        if power_was_off:
            state["power"] = "on"
            state["mode"] = state.get("mode") or "cooling"

        # ---- 根据属性字段做增量 ----
        if attr_field == "temperature":
            lo, hi = 16, 30
            current = int(state.get("temperature", 26))
            raw_new = current + delta_value
            new_val = max(lo, min(hi, raw_new))
            actual_delta = new_val - current
            state["temperature"] = new_val
            # 生成回复文本
            if actual_delta > 0:
                changes.append(f"温度{('调高 ' + str(actual_delta) + ' 度') if actual_delta != 1 else '调高 1 度'}")
            elif actual_delta < 0:
                changes.append(f"温度{('调低 ' + str(abs(actual_delta)) + ' 度') if actual_delta != -1 else '调低 1 度'}")
            else:
                changes.append(f"温度已达{'最高' if delta_value > 0 else '最低'}值（{new_val} 度），无法继续{'调高' if delta_value > 0 else '调低'}")
            changes.append(f"当前为 {new_val} 度")
            # 把 value/value_type 补上，方便后续调试看
            value = new_val
            value_type = "temperature"

        elif attr_field == "fan_speed":
            lo, hi = 1, 5
            current_raw = state.get("fan_speed", 2)
            # fan_speed 可能是 "quiet"/"auto" 这种非数字值 → 取基础档位 2（或 quiet→1, auto→2）
            fan_equiv = {"quiet": 1, "auto": 2}
            base = current_raw if isinstance(current_raw, int) else fan_equiv.get(str(current_raw).lower(), 2)
            raw_new = base + delta_value
            new_val = max(lo, min(hi, raw_new))
            actual_delta = new_val - base
            state["fan_speed"] = new_val
            fan_cn_new = AC_FAN_NAMES.get(new_val, f"{new_val}档")
            if actual_delta > 0:
                changes.append(f"风力{('调大 ' + str(actual_delta) + ' 档') if actual_delta != 1 else '调大 1 档'}")
            elif actual_delta < 0:
                changes.append(f"风力{('调小 ' + str(abs(actual_delta)) + ' 档') if actual_delta != -1 else '调小 1 档'}")
            else:
                changes.append(f"风力已达{'最高' if delta_value > 0 else '最低'}档（{fan_cn_new}），无法继续{'调大' if delta_value > 0 else '调小'}")
            changes.append(f"当前为{fan_cn_new}")

        else:
            return {
                "success": False, "code": 400,
                "message": f"暂不支持对{display}的{attr_cn}做增量调节",
                "resolved_device_id": dev_id, "state_snapshot": state.copy(),
            }

        # 拼接最终消息
        if power_was_off:
            prefix = f"已为您开启{display}"
            if changes:
                prefix += "，" + "，".join(changes)
        else:
            prefix = f"已将{display}" + "，".join(changes)
        return {
            "success": True, "code": 200,
            "message": prefix,
            "resolved_device_id": dev_id,
            "state_snapshot": state.copy(),
        }

    # 语义触发的模式指令：cool/heat 等同于 on + 设置模式
    if action in ("cool", "heat", "auto", "fan", "dry"):
        mode_map = {"cool": "cooling", "heat": "heating", "auto": "auto", "fan": "fan", "dry": "dry"}
        mode_name = mode_map.get(action, action)
        if state.get("power") == "on":
            if state.get("mode") == mode_name:
                return {
                    "success": True, "code": 200,
                    "message": f"{display}已经在{AC_MODE_NAMES.get(mode_name, action)}模式了",
                    "resolved_device_id": dev_id,
                    "state_snapshot": state.copy(),
                }
            state["mode"] = mode_name
            return {
                "success": True, "code": 200,
                "message": f"已为您将{display}切换为{AC_MODE_NAMES.get(mode_name, action)}模式",
                "resolved_device_id": dev_id,
                "state_snapshot": state.copy(),
            }
        state["power"] = "on"
        state["mode"] = mode_name
        return {
            "success": True, "code": 200,
            "message": f"已为您开启{display}，并切换为{AC_MODE_NAMES.get(mode_name, action)}模式",
            "resolved_device_id": dev_id,
            "state_snapshot": state.copy(),
        }

    # ================================================================
    # 【★ attrs.temperature 兼容层】：在 _execute_command_impl 层已经处理了
    #   顶层字段 / raw_text 中文数字，这里做最后一道 attrs.temperature → value/value_type
    #   的同步，保证 100% 进入 clamp 分支。
    # ================================================================
    if (value is None or value_type != "temperature") and isinstance(attrs, dict):
        attr_temp = attrs.get("temperature")
        if attr_temp is not None:
            candidate = None
            if isinstance(attr_temp, (int, float)):
                candidate = int(attr_temp)
            elif isinstance(attr_temp, str) and attr_temp.strip().isdigit():
                candidate = int(attr_temp.strip())
            if candidate is not None:
                value, value_type = candidate, "temperature"
                try:
                    attrs.pop("temperature", None)
                except Exception:
                    pass
                print(f"[_execute_ac][{dev_id}] attrs.temperature 兼容成功 → {candidate}°")

    # ================================================================
    # 解耦开关幂等 与 属性调节（需求 2）：
    #   引入 power_idempotent 前缀：当 action=="on" 但设备已开时，仅记录这个
    #   状态，不做短路 return；让后续温度/模式/风力照常处理。
    #   值为 None / "already_on" / "turned_on" / "turned_off"
    # ================================================================
    power_note: Optional[str] = None   # None / "already_on" / "turned_on" / "turned_off"

    # ---------- 开关控制（严格按：有属性时不短路 return） ----------
    if action == "on":
        if state.get("power") == "on":
            if not value_type and not attrs:
                # 纯开启，无任何属性 → 幂等提示直接返回
                return {
                    "success": True, "code": 200,
                    "message": f"{display}已经是开启状态了。",
                    "resolved_device_id": dev_id,
                    "state_snapshot": state.copy(),
                }
            # 有属性调节 → 记录"已处于开启状态"前缀，继续执行属性更新，不 return
            power_note = "already_on"
        else:
            state["power"] = "on"
            power_note = "turned_on"
    elif action == "off":
        if state.get("power") == "off" and not value_type and not attrs:
            return {
                "success": True, "code": 200,
                "message": f"{display}已经是关闭状态了哦。",
                "resolved_device_id": dev_id,
                "state_snapshot": state.copy(),
            }
        state["power"] = "off"
        power_note = "turned_off"

    # ================================================================
    # 温度处理（需求 1：严格按文案格式 + clamp 后必须说明原因和实际设定值）
    # ================================================================
    temp_clamped_full_msg = None
    temp_normal_change = None   # 非越界时，保存"温度调至 24 度"这种短语
    TEMP_LO, TEMP_HI = 16, 30

    if value_type == "temperature" and value is not None:
        clamped_reason = None  # "high" / "low" / None
        raw_value_before_clamp = value  # 保留原始值用于日志
        if value > TEMP_HI:
            clamped_reason = "high"
            value = TEMP_HI
        elif value < TEMP_LO:
            clamped_reason = "low"
            value = TEMP_LO
        print(f"[_execute_ac][{dev_id}] 温度 clamp → 原始值={raw_value_before_clamp}, 限制后={value}, 越界类型={clamped_reason!r}, 当前电源状态={state.get('power')}")

        auto_turned_on = False
        if state.get("power") != "on":
            state["power"] = "on"
            state["mode"] = state.get("mode") or "cooling"
            auto_turned_on = True
            # 如果之前没记过 power_note（action 是 set 而不是 on），补一下
            if power_note is None:
                power_note = "turned_on"
        state["temperature"] = value

        # ---- 严格按需求 1 格式输出越界提示 ----
        if clamped_reason == "high":
            # 格式："空调温度范围为 16-30 度，已为您将客厅空调调至最高 30 度。"
            if auto_turned_on:
                temp_clamped_full_msg = (
                    f"空调温度范围为 {TEMP_LO}-{TEMP_HI} 度，"
                    f"已为您开启{display}并调至最高 {TEMP_HI} 度。"
                )
            else:
                temp_clamped_full_msg = (
                    f"空调温度范围为 {TEMP_LO}-{TEMP_HI} 度，"
                    f"已为您将{display}调至最高 {TEMP_HI} 度。"
                )
        elif clamped_reason == "low":
            if auto_turned_on:
                temp_clamped_full_msg = (
                    f"空调温度范围为 {TEMP_LO}-{TEMP_HI} 度，"
                    f"已为您开启{display}并调至最低 {TEMP_LO} 度。"
                )
            else:
                temp_clamped_full_msg = (
                    f"空调温度范围为 {TEMP_LO}-{TEMP_HI} 度，"
                    f"已为您将{display}调至最低 {TEMP_LO} 度。"
                )
        else:
            # 正常范围：明确说明"温度调至 X 度"，绝不只说"已设置空调"
            temp_normal_change = f"温度调至 {value} 度"
            changes.append(temp_normal_change)

    # 属性值规范化映射（统一转换为前端能识别的标准格式）
    _fan_norm = {
        "quiet": "quiet", "auto": "auto",
        "1档": 1, "2档": 2, "3档": 3, "4档": 4, "5档": 5,
        "一档": 1, "二档": 2, "三档": 3, "四档": 4, "五档": 5,
        "一挡": 1, "二挡": 2, "三挡": 3, "四挡": 4, "五挡": 5,
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
        "low": 1, "medium": 2, "mid": 2, "high": 3,
        "低档": 1, "低速": 1, "低风": 1, "小风": 1, "小档": 1,
        "中档": 2, "中速": 2, "中风": 2,
        "高档": 3, "高速": 3, "高风": 3, "大风": 3, "大档": 3,
        "静音档": "quiet", "静音风": "quiet", "静音": "quiet", "最低风": "quiet",
        "自动风": "auto", "自动档": "auto",
    }

    # 模式（注意：AC_MODE_NAMES 的 value 已自带"模式"结尾，如 cooling→"制冷模式"）
    mode = attrs.get("mode")
    if mode:
        normalized_mode = AC_MODE_NORM.get(mode, mode)
        state["mode"] = normalized_mode
        mode_cn = AC_MODE_NAMES.get(normalized_mode, mode)
        # 统一前缀"切换为XX模式"，语义更自然、不重复
        changes.append(f"切换为{mode_cn}")

    # 扫风
    swing = attrs.get("swing")
    if swing:
        normalized_swing = AC_SWING_NORM.get(swing, swing)
        state["swing"] = normalized_swing
        swing_cn = AC_SWING_NAMES.get(normalized_swing, swing)
        changes.append(swing_cn)

    # 风力
    fan_speed = attrs.get("fan_speed")
    if fan_speed is not None:
        current_fan = state.get("fan_speed")
        if fan_speed in _fan_norm:
            normalized_fan = _fan_norm[fan_speed]
        else:
            s = str(fan_speed).strip()
            num_map = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}
            def _extract_delta(text):
                m2 = re.search(r'(一|二|两|三|四|五|[1-5])\s*(档|挡)', text)
                if m2:
                    t = m2.group(1)
                    if t in num_map:
                        return num_map[t]
                    try:
                        return int(t)
                    except Exception:
                        return 1
                return 1
            m = re.match(r'^\s*([+-])\s*(\d+)\s*(档|挡)?\s*$', s)
            if m and isinstance(current_fan, int):
                _d = int(m.group(2)) * (1 if m.group(1) == '+' else -1)
                base = current_fan if isinstance(current_fan, int) else 2
                normalized_fan = max(1, min(5, base + _d))
            elif any(re.search(p, s) for p in [r'调大', r'调高', r'加大', r'增强', r'大一档', r'大.*档', r'升.*档']) and isinstance(current_fan, int):
                d = _extract_delta(s)
                normalized_fan = max(1, min(5, current_fan + d))
            elif any(re.search(p, s) for p in [r'调小', r'调低', r'减小', r'减弱', r'小一档', r'小.*档', r'降.*档']) and isinstance(current_fan, int):
                d = _extract_delta(s)
                normalized_fan = max(1, min(5, current_fan - d))
            elif re.match(r'^\d+$', s) and 1 <= int(s) <= 5:
                normalized_fan = int(s)
            else:
                normalized_fan = _fan_norm.get(fan_speed, fan_speed)
        state["fan_speed"] = normalized_fan
        fan_cn = AC_FAN_NAMES.get(normalized_fan, str(normalized_fan) if isinstance(normalized_fan, int) else str(normalized_fan))
        changes.append(f"风速调至{fan_cn}")

    # 默认模式（on 且未显式指定 mode 时补 cooling）
    if action == "on" and "mode" not in (attrs or {}) and power_note in ("turned_on", "already_on"):
        state["mode"] = state.get("mode", "cooling")

    # 只要 action 是 set 或 属性调节，确保电源是开的
    if action != "on" and action != "off" and state.get("power") != "on":
        state["power"] = "on"
        if power_note is None:
            power_note = "turned_on"

    # ================================================================
    # 回复文案组装（重点：已开+有属性 → 用"已处于开启状态，已为您..."）
    # ================================================================
    has_any_change = bool(changes) or temp_clamped_full_msg is not None or power_note is not None

    if not has_any_change:
        # 连状态 + 属性 + 开关 都没碰（理论上只会在 action=set 但啥都没传时触发）
        # → 按 intent+display 动态拼精准文案，绝不返回空 message 触发外层兜底
        safe_msg = _build_confirmation(
            {"device_id": dev_id, "device_name": display, "action": action,
             "value": value, "value_type": value_type},
            fallback_reason="AC has_any_change=False",
        )
        print(f"[_execute_ac][{dev_id}] 最终回复(无变更分支) → {safe_msg!r}")
        return {
            "success": True, "code": 200,
            "message": safe_msg,
            "resolved_device_id": dev_id,
            "state_snapshot": state.copy(),
        }
    else:
        # ---- 温度越界句式优先（已有完整主谓宾，其它属性追加其后） ----
        if temp_clamped_full_msg is not None:
            other_adjusts = [c for c in changes if c != temp_normal_change]
            if other_adjusts:
                msg = temp_clamped_full_msg.rstrip("。！？.!?") + "，" + "，".join(other_adjusts) + "。"
            else:
                msg = temp_clamped_full_msg
            print(f"[_execute_ac][{dev_id}] 最终回复(越界分支) → {msg!r}")
            return {
                "success": True, "code": 200,
                "message": msg,
                "resolved_device_id": dev_id,
                "state_snapshot": state.copy(),
            }

        # ---- 正常情况：按 power_note + 属性 组合文案 ----
        # 先区分出"温度/模式/风速"等属性变更（changes 现在不会包含"已开启/已关闭"了）
        attr_changes = list(changes)

        if power_note == "already_on":
            # 【需求 2 文案】设备已开，仅对属性做确认
            if attr_changes:
                attr_text = "，".join(attr_changes)
                msg = f"{display}已处于开启状态，已为您将{attr_text}。"
            else:
                msg = f"{display}已处于开启状态。"
        elif power_note == "turned_on":
            prefix = f"已为您开启{display}"
            if attr_changes:
                msg = prefix + "，" + "，".join(attr_changes) + "。"
            else:
                msg = prefix + "。"
        elif power_note == "turned_off":
            prefix = f"已为您关闭{display}"
            if attr_changes:
                msg = prefix + "，" + "，".join(attr_changes) + "。"
            else:
                msg = prefix + "。"
        else:
            # 纯属性变更（没有涉及开关）→ "已为您将卧室空调温度调至 24 度，模式切换为制冷。"
            if attr_changes:
                attr_text = "，".join(attr_changes)
                msg = f"已为您将{display}{attr_text}。"
            else:
                msg = f"已为您将{display}设置好。"

        print(f"[_execute_ac][{dev_id}] 最终回复(正常分支 power_note={power_note!r}) → {msg!r}")
        return {
            "success": True, "code": 200,
            "message": msg,
            "resolved_device_id": dev_id,
            "state_snapshot": state.copy(),
        }

    # increase/decrease on AC temperature
    if action in ("increase", "decrease"):
        step = 1 if action == "increase" else -1
        cur = state.get("temperature", 26)
        state["temperature"] = max(TEMP_LO, min(TEMP_HI, cur + step))
        state["power"] = "on"
        msg = f"已将{display}温度{'升高' if step > 0 else '降低'}至 {state['temperature']} 度。"
        print(f"[_execute_ac][{dev_id}] 最终回复(增减分支) → {msg!r}")
        return {
            "success": True, "code": 200,
            "message": msg,
            "resolved_device_id": dev_id,
            "state_snapshot": state.copy(),
        }

    # 空调 set 动作兜底：【删除原来的空话，改为动态读 state 播报关键属性值】
    if action == "set":
        # 从当前 state 读取有意义的字段，避免"已设置好"这种空泛回复
        _parts = []
        if isinstance(state.get("temperature"), (int, float)):
            _parts.append(f"温度为 {state['temperature']} 度")
        if state.get("mode"):
            _parts.append(f"模式为{AC_MODE_NAMES.get(state['mode'], state['mode'])}")
        fan_raw = state.get("fan_speed")
        if fan_raw is not None:
            fan_cn = AC_FAN_NAMES.get(fan_raw, (str(fan_raw) + "档") if isinstance(fan_raw, int) else str(fan_raw))
            _parts.append(f"风速为{fan_cn}")
        if _parts:
            msg = f"已为您将{display}设置为：" + "，".join(_parts) + "。"
        else:
            msg = f"已为您将{display}设置好。"
        print(f"[_execute_ac][{dev_id}] 最终回复(set 兜底分支) → {msg!r}")
        return {
            "success": True, "code": 200,
            "message": msg,
            "resolved_device_id": dev_id,
            "state_snapshot": state.copy(),
        }

    msg_500 = f"抱歉，暂时无法理解对{display}的操作"
    print(f"[_execute_ac][{dev_id}] 最终回复(500 兜底) → {msg_500!r}")
    return {
        "success": False, "code": 500,
        "message": msg_500,
        "resolved_device_id": dev_id,
        "state_snapshot": state.copy(),
    }


async def _execute_tv(state: Dict[str, Any], display: str, action: str,
                      attrs: Dict[str, Any], dev_id: str,
                      control_mode: str = "absolute",
                      delta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """电视控制：开关。delta 模式下先转绝对量处理（暂未实现音量/频道增量，回退为通用提示）"""
    # 对 TV 的增量控制暂时不支持（DEVICE_STATES 里没有 volume 字段），给出清晰提示
    if control_mode == "delta" and delta:
        attr_cn = delta.get("attr_cn") or "参数"
        return {
            "success": False, "code": 400,
            "message": f"抱歉，暂不支持对{display}的{attr_cn}做增量调节",
            "resolved_device_id": dev_id,
            "state_snapshot": state.copy(),
        }
    if action == "on":
        if state.get("power") == "on":
            return {
                "success": True, "code": 200,
                "message": f"{display}已经是开启状态了。",
                "resolved_device_id": dev_id,
                "state_snapshot": state.copy(),
            }
        state["power"] = "on"
        return {
            "success": True, "code": 200,
            "message": f"已为您开启{display}",
            "resolved_device_id": dev_id,
            "state_snapshot": state.copy(),
        }
    elif action == "off":
        if state.get("power") == "off":
            return {
                "success": True, "code": 200,
                "message": f"{display}已经是关闭状态了哦。",
                "resolved_device_id": dev_id,
                "state_snapshot": state.copy(),
            }
        state["power"] = "off"
        return {
            "success": True, "code": 200,
            "message": f"已为您关闭{display}",
            "resolved_device_id": dev_id,
            "state_snapshot": state.copy(),
        }

    return {
        "success": False, "code": 500,
        "message": f"抱歉，暂时无法理解对{display}的操作",
        "resolved_device_id": dev_id,
        "state_snapshot": state.copy(),
    }


async def _publish_changed_mqtt(
    before_snap: Dict[str, Dict[str, Any]],
    intent: Dict[str, Any],
) -> None:
    """对比 before_snap 与当前 DEVICE_STATES，diff 出被修改的设备，逐个 publish MQTT。"""
    try:
        from mqtt_client import get_mqtt_client
    except ImportError:
        return
    mqtt = get_mqtt_client()

    action = intent.get("action", "set")
    value = intent.get("value")
    attrs = intent.get("attrs") or {}
    hvac_mode = attrs.get("hvac_mode", "") or attrs.get("mode", "")
    fan_mode_val = attrs.get("fan_speed", "") or attrs.get("fan_mode", "")
    swing_mode_val = attrs.get("swing_mode", "") or attrs.get("swing", "")

    alias_map = {
        "turn_on": "turn_on", "on": "turn_on",
        "turn_off": "turn_off", "off": "turn_off",
        "toggle": "toggle",
        "set_fan_mode": "set", "set_hvac_mode": "set", "set_swing_mode": "set",
        "set_temperature": "set", "set_value": "set", "set_ac": "set", "set": "set",
        "adjust": "set", "adjust_ac": "set",
    }
    norm_action = alias_map.get(action, action if action in ("turn_on", "turn_off", "toggle", "set") else "set")

    import copy as _c
    async with state_lock:
        changed_devs = []
        for dev_id, cur in DEVICE_STATES.items():
            prev = before_snap.get(dev_id, {})
            if cur != prev:
                changed_devs.append(dev_id)

    for dev_id in changed_devs:
        meta = get_mqtt_meta(dev_id)
        room = meta.get("room")
        dtype = meta.get("type")
        if not room or not dtype:
            continue
        extra: Dict[str, Any] = {}
        async with state_lock:
            s = _c.deepcopy(DEVICE_STATES.get(dev_id, {}))

        # ================================================================
        # 【★ 关键修复】MQTT payload 的属性值，必须以 clamp 后的 DEVICE_STATES 为准！
        #   旧逻辑：if intent.value 非空 → 用 intent.value（原始 5/50 未 clamp，导致越界下发）
        #   新逻辑：先填 intent.attrs 中那些 state 没覆盖的（非空调普通参数），
        #          然后对 AC 用 state 值 100% 覆盖 temperature/mode/fan_speed/swing_mode。
        # ================================================================

        # 非 AC 场景：允许用 intent.value / intent.attrs（但仍优先 state 里的明确值）
        if s.get("type") != "ac":
            if value is not None and value != "":
                extra["temperature"] = value
            if fan_mode_val and fan_mode_val != "":
                extra["fan_mode"] = fan_mode_val
                extra["fan_speed"] = fan_mode_val
            if hvac_mode:
                extra["mode"] = hvac_mode
            if swing_mode_val and swing_mode_val != "":
                extra["swing_mode"] = swing_mode_val
        # 空调场景：先填 intent.attrs 中 non-control 属性（如 fan_mode 同义名兼容），
        #           然后强制用 state 值覆盖 temperature/mode/fan_speed/swing_mode。
        else:
            # 兼容 fan_mode/fan_speed/swing_mode/swing 同义名命名差异的透传（值仍以 state 为准）
            if fan_mode_val and fan_mode_val != "":
                extra["fan_mode"] = fan_mode_val
            if swing_mode_val and swing_mode_val != "":
                pass  # 下面用 state["swing"] 统一覆盖
            # ★ 一律以 clamp / 属性合并后的 state 为最终下发值
            t = s.get("temperature")
            if t is not None:
                extra["temperature"] = t
            m = s.get("mode")
            if m:
                extra["mode"] = m
            fs = s.get("fan_speed")
            if fs is not None:
                extra["fan_speed"] = fs
                extra["fan_mode"] = fs  # 保持与旧协议双字段兼容
            sw = s.get("swing") or swing_mode_val
            if sw:
                extra["swing_mode"] = sw

        # 电源/窗帘状态（保持原逻辑）
        if s.get("type") == "curtain":
            extra["power"] = s.get("state", "closed")
        else:
            extra["power"] = s.get("power", "off")
        mqtt.publish_command(
            device_id=dev_id,
            room=room,
            dev_type=dtype,
            action=norm_action,
            extra=extra if extra else None,
        )


async def execute_command(intent: Dict[str, Any]) -> Dict[str, Any]:
    """执行设备控制命令（包装 MQTT 发布）。

    1) 保存 DEVICE_STATES 快照
    2) 调用 _execute_command_impl（修改 DEVICE_STATES）
    3) diff 变化的设备，逐个 publish MQTT

    特殊短路：隐式意图的"虚拟设备"（如无开启媒体设备核查）不涉及 DEVICE_STATES 修改，
    也不会 publish MQTT，直接在 impl 层返回友好提示。
    """
    # ---------- 隐式虚拟意图：不碰状态，不发 MQTT ----------
    if intent.get("no_running_media"):
        return await _execute_command_impl(intent)

    import copy as _copy
    async with state_lock:
        before_snap = {k: _copy.deepcopy(v) for k, v in DEVICE_STATES.items()}
    result = await _execute_command_impl(intent)
    if result.get("success", False):
        try:
            await _publish_changed_mqtt(before_snap, intent)
        except Exception as _mqtt_e:
            # MQTT 失败不吞掉"操作已成功"的文案，但要日志打出来
            print(f"[MQTT] publish 失败（不影响用户回复文案）: {_mqtt_e}")

    # —— ★ execute_command 最终出口：message 强制非空（【需求 3】：正常 turn_off/set_temperature 分支必须有精确文案，"操作已完成"只能最后兜底）——
    msg = result.get("message") if isinstance(result, dict) else None
    if not isinstance(msg, str) or not msg.strip():
        # 先根据 intent.device_id + action 拼出 "已为您关闭客厅空调。"/"已为您开启…"/"已为您设置好…"
        safe_msg = _build_confirmation(intent, fallback_reason=f"impl 返回 message={msg!r}")
        if not safe_msg or not safe_msg.strip():
            safe_msg = "操作已完成。"
        if isinstance(result, dict):
            result["message"] = safe_msg
        else:
            result = {"success": True, "code": 200, "message": safe_msg}
        print(f"[Reply-Safe] execute_command: impl 返回 message 为空 → 按 intent 拼装确认文案 → {safe_msg!r}")
    return result


async def _execute_command_impl(intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据 NLU 导出的结构化意图执行设备控制。

    Args:
        intent: {
            "device_id": str,
            "device_name": str,
            "action": str,
            "value": Any,
            "value_type": str,
            "control_mode": "absolute" | "delta",  # 新增
            "delta": None | {                        # 新增
                "attr_field": str, "attr_cn": str,
                "delta_value": int, "bound_type": str, "is_qualitative": bool
            },
        }

    Returns:
        {
            "success": bool,
            "code": int,
            "message": str,
            "resolved_device_id": str,
            "state_snapshot": dict
        }
    """
    dev_id = _resolve_device_id(intent)
    # intent.device_name 不是必传：缺的时候按 intent.device_id/DEVICE_STATES 反推中文显示名
    dev_name = intent.get("device_name")
    if not dev_name:
        if dev_id and dev_id in DEVICE_STATES:
            st = DEVICE_STATES[dev_id]
            room_cn = st.get("room") or ""
            type_cn = DEVICE_TYPE_NAMES.get(st.get("type") or "", "")
            if room_cn and type_cn:
                dev_name = room_cn + type_cn
        if not dev_name and dev_id:
            # 从 device_id 拆 bedroom_ac → 卧室空调
            parts = dev_id.replace(".all", "").split("_")
            if parts:
                slug = parts[0]
                type_short = "_".join(parts[1:]) if len(parts) > 1 else ""
                room_cn = ROOM_SLUG_TO_CN.get(slug, slug)
                type_cn = DEVICE_TYPE_NAMES.get(type_short, type_short)
                if room_cn and type_cn:
                    dev_name = room_cn + type_cn
                elif type_cn:
                    dev_name = type_cn
                else:
                    dev_name = dev_id
            else:
                dev_name = dev_id
        if not dev_name:
            dev_name = "设备"
    action = intent.get("action") or ""
    value = intent.get("value")
    value_type = intent.get("value_type")
    attrs = intent.get("attrs") or {}
    control_mode = intent.get("control_mode", "absolute")
    delta = intent.get("delta")

    # ============ 【★ 动作别名归一化（最前置！）】 ============
    #   必须在 AC/Light/TV 等任何类型分支之前做，确保所有分支看到的 action = on/off/set/...
    #   否则 Light 分支（780行）因 action="turn_off" 不匹配而返回 500 错误。
    if action == "turn_on":
        action = "on"
    elif action == "turn_off":
        action = "off"
    elif action in ("set_fan_mode", "set_hvac_mode", "set_swing_mode", "set_temperature", "set_value", "set_ac", "adjust", "adjust_ac"):
        action = "set"
    elif action == "open_curtain":
        action = "on"
    elif action == "close_curtain":
        action = "off"

    # ============ 【★ 显式房间+设备 存在性拦截（禁止模糊回退）】 ============
    #   核心 Bug：用户说"打开书房灯"时，_match_device 先匹配到泛化"灯"→ light，
    #           resolve_device_id 默认回退 bedroom → 被误识别成"卧室灯已开"。
    #   修复原则：显式提到"XX房/厅 + 设备"组合的，若系统中无该房间的对应类型设备，
    #           直接返回"未找到XX设备"，绝不再回退到其他房间。
    if not dev_id.endswith(".all"):
        raw_text = intent.get("raw_text") or intent.get("original_text") or ""
        explicit = _extract_explicit_room_and_type(raw_text)
        if explicit:
            r_cn, r_orig, d_type, t_orig = explicit
            if not _device_exists_with_room_and_type(r_cn, d_type):
                friendly_display = f"{r_cn}{t_orig}"
                return {
                    "success": False,
                    "code": 404,
                    "message": f"未找到{friendly_display}，请检查设备名称。",
                    "resolved_device_id": dev_id,
                    "state_snapshot": {},
                }

    # ---- 隐式意图：噪音感知但"没有正在播放的媒体设备" ----
    # （在 .all 扩展与 DEVICE_STATES 查找之前拦截，避免报 "找不到设备"）
    if intent.get("no_running_media"):
        target_room = intent.get("no_running_room") or ""
        if target_room:
            message = f"已为您核查，当前{target_room}没有播放设备"
        else:
            message = "已为您核查，当前房间没有播放设备"
        return {
            "success": True,
            "code": 200,
            "message": message,
            "resolved_device_id": dev_id,
            "state_snapshot": {},
        }

    # ---- 多设备扩展：curtain.all / light.all 等泛指目标 ----
    if dev_id.endswith(".all"):
        short_type = dev_id[:-4]  # curtain.all -> curtain
        type_labels = {"curtain": "窗帘", "light": "灯", "ac": "空调", "tv": "电视"}
        verb_on = {"curtain": "拉开", "light": "打开", "ac": "开启", "tv": "开启"}
        verb_off = {"curtain": "关闭", "light": "关闭", "ac": "关闭", "tv": "关闭"}
        verb_set = {"curtain": "设置", "light": "设置", "ac": "设置", "tv": "设置"}
        type_label = type_labels.get(short_type, short_type)
        results = []
        async with state_lock:
            targets = [k for k, v in DEVICE_STATES.items() if v["type"] == short_type]
        for target_id in targets:
            sub_intent = {
                "device_id": target_id,
                "device_name": type_label,
                "action": action,
                "value": value,
                "value_type": value_type,
                "attrs": attrs,
                "control_mode": control_mode,
                "delta": delta,
            }
            sub_result = await _execute_command_impl(sub_intent)
            results.append(sub_result)
        all_ok = all(r.get("success", False) for r in results)

        # 生成简洁的汇总消息
        if action == "on":
            verb = verb_on.get(short_type, "打开")
            message = f"已为您打开所有{type_label}"
        elif action == "off":
            verb = verb_off.get(short_type, "关闭")
            message = f"已为您关闭所有{type_label}"
        else:
            verb = verb_set.get(short_type, "设置")
            message = f"已为您{verb}所有{type_label}"

        return {
            "success": all_ok,
            "code": 200 if all_ok else 207,
            "message": message,
            "resolved_device_id": dev_id,
            "state_snapshot": {},
        }

    if dev_id not in DEVICE_STATES:
        return {
            "success": False,
            "code": 404,
            "message": f"找不到设备「{dev_name}」",
            "resolved_device_id": dev_id,
            "state_snapshot": {},
        }

    # ================================================================
    # 【★ 调用设备特定函数前 → 统一 INTENT 预处理（解决中文数字温度 / 顶层字段布局不一致）】
    #   - 补全 value / value_type（从 raw_text 提取中文数字温度、从 intent 顶层
    #     temperature/mode/fan_speed 同步到 attrs/value 等标准字段）
    #   - 对 ac 类型 100% 走 PRINT 诊断日志（DEBUG 开关在 _execute_ac）
    # ================================================================
    if dev_id in DEVICE_STATES and DEVICE_STATES[dev_id]["type"] == "ac":
        # (A) intent 顶层 temperature / mode / fan_speed / swing_mode → 同步到标准字段
        t_top = intent.get("temperature")
        if t_top is not None and (value is None or value_type != "temperature"):
            if isinstance(t_top, (int, float)) or (isinstance(t_top, str) and str(t_top).strip().isdigit()):
                value = int(t_top) if isinstance(t_top, (int, float)) else int(str(t_top).strip())
                value_type = "temperature"
        for src_key, dst_key in (("mode", "mode"), ("fan_mode", "fan_speed"),
                                 ("fan_speed", "fan_speed"), ("swing_mode", "swing"),
                                 ("swing", "swing")):
            v = intent.get(src_key)
            if v is not None:
                if not isinstance(attrs, dict):
                    attrs = {}
                attrs.setdefault(dst_key, v)

        # (B) raw_text 终极防线：中文数字温度提取（解决"二十四度"→24 丢失）
        if value is None or value_type != "temperature":
            raw = (intent.get("raw_text") or intent.get("original_text") or "").strip()
            if raw:
                cn_temp = _extract_any_temperature_degree(raw)
                if cn_temp is not None:
                    value, value_type = cn_temp, "temperature"
                    print(f"[AC-Preprocess][{dev_id}] ★ raw_text 兜底提取温度成功: '{raw}' → {value}°")

        # (C) PRINT 预处理后最终传入 _execute_ac 的参数（便于用户日志里一眼看穿）
        print(f"[AC-Preprocess][{dev_id}] 最终下发 → action={action!r} | value={value!r} | value_type={value_type!r} | attrs={attrs!r}")

    async with state_lock:
        state = DEVICE_STATES[dev_id]
        dev_type = state["type"]
        display = _display_name(state, dev_name)

        # ============ 空调多属性控制 ============
        if dev_type == "ac":
            return await _execute_ac(state, display, action, value, value_type, attrs, dev_id,
                                     control_mode=control_mode, delta=delta)

        # ============ 电视多属性控制 ============
        if dev_type == "tv":
            return await _execute_tv(state, display, action, attrs, dev_id,
                                     control_mode=control_mode, delta=delta)

        # 灯光开关
        if dev_type == "light":
            if action == "on":
                if state.get("power") == "on":
                    return {
                        "success": True, "code": 200,
                        "message": f"{display}已经是开启状态了。",
                        "resolved_device_id": dev_id,
                        "state_snapshot": state.copy(),
                    }
                state["power"] = "on"
                return {
                    "success": True, "code": 200,
                    "message": f"已为您开启{display}",
                    "resolved_device_id": dev_id,
                    "state_snapshot": state.copy(),
                }
            elif action == "off":
                if state.get("power") == "off":
                    return {
                        "success": True, "code": 200,
                        "message": f"{display}已经是关闭状态了哦。",
                        "resolved_device_id": dev_id,
                        "state_snapshot": state.copy(),
                    }
                state["power"] = "off"
                return {
                    "success": True, "code": 200,
                    "message": f"已为您关闭{display}",
                    "resolved_device_id": dev_id,
                    "state_snapshot": state.copy(),
                }
            return {
                "success": False, "code": 500,
                "message": f"抱歉，暂时无法理解对{display}的操作",
                "resolved_device_id": dev_id,
                "state_snapshot": state.copy(),
            }

        # 窗帘状态（保留基于 value 的全开/全关，兼容 open/close 英文值）
        if dev_type == "curtain" and value_type == "curtain_state" and value is not None:
            _curtain_open_set = {"全开", "open", "打开", "拉开", "全拉"}
            _curtain_close_set = {"全关", "close", "关闭", "拉上"}
            state["state"] = "open" if value in _curtain_open_set else "closed"
            return {
                "success": True, "code": 200,
                "message": f"已将{display}{'拉开' if state['state'] == 'open' else '关闭'}",
                "resolved_device_id": dev_id,
                "state_snapshot": state.copy(),
            }

        # ============ 2. 通用增减操作 ============

        if action in ("increase", "decrease"):
            return {
                "success": False, "code": 400,
                "message": f"{display} 不支持增减操作",
                "resolved_device_id": dev_id,
                "state_snapshot": state.copy(),
            }

        # ============ 3. set 动作 ============

        if action == "set" and value is not None and value_type:
            return {
                "success": True, "code": 200,
                "message": f"已将{display}设置为 {value}",
                "resolved_device_id": dev_id,
                "state_snapshot": state.copy(),
            }

        # ============ 4. 窗帘开关（open/close 语义）============

        if dev_type == "curtain":
            if action == "on":
                if state.get("state") == "open":
                    return {
                        "success": True, "code": 200,
                        "message": f"{display}已经是拉开状态了。",
                        "resolved_device_id": dev_id,
                        "state_snapshot": state.copy(),
                    }
                state["state"] = "open"
                return {
                    "success": True, "code": 200,
                    "message": f"已将{display}拉开",
                    "resolved_device_id": dev_id,
                    "state_snapshot": state.copy(),
                }
            elif action == "off":
                if state.get("state") == "closed":
                    return {
                        "success": True, "code": 200,
                        "message": f"{display}已经是关闭状态了哦。",
                        "resolved_device_id": dev_id,
                        "state_snapshot": state.copy(),
                    }
                state["state"] = "closed"
                return {
                    "success": True, "code": 200,
                    "message": f"已将{display}关闭",
                    "resolved_device_id": dev_id,
                    "state_snapshot": state.copy(),
                }
            else:
                return {
                    "success": False, "code": 500,
                    "message": f"抱歉，无法识别窗帘操作",
                    "resolved_device_id": dev_id,
                    "state_snapshot": state.copy(),
                }

        # ============ 5. 通用开关（AC / Light / TV）============

        # 别名归一化（兼容 turn_on / turn_off，避免不同模块传 action 名不一致）
        if action == "turn_on":
            action = "on"
        elif action == "turn_off":
            action = "off"

        if action == "on":
            if state.get("power") == "on":
                return {
                    "success": True, "code": 200,
                    "message": f"{display}已经是开启状态了。",
                    "resolved_device_id": dev_id,
                    "state_snapshot": state.copy(),
                }
            state["power"] = "on"
            return {
                "success": True, "code": 200,
                "message": f"已为您开启{display}",
                "resolved_device_id": dev_id,
                "state_snapshot": state.copy(),
            }

        if action == "off":
            if state.get("power") == "off":
                return {
                    "success": True, "code": 200,
                    "message": f"{display}已经是关闭状态了哦。",
                    "resolved_device_id": dev_id,
                    "state_snapshot": state.copy(),
                }
            state["power"] = "off"
            return {
                "success": True, "code": 200,
                "message": f"已为您关闭{display}",
                "resolved_device_id": dev_id,
                "state_snapshot": state.copy(),
            }

        return {
            "success": False, "code": 500,
            "message": f"抱歉，暂时无法理解对{display}的操作",
            "resolved_device_id": dev_id,
            "state_snapshot": state.copy(),
        }
