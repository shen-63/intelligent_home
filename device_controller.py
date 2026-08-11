from typing import Dict, Any, Optional, List
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


async def _execute_ac(state: Dict[str, Any], display: str, action: str,
                      value: Any, value_type: str, attrs: Dict[str, Any],
                      dev_id: str) -> Dict[str, Any]:
    """空调多属性控制：一次指令可同时设置温度、模式、扫风、风力"""
    changes: List[str] = []

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

    # 开关控制
    if action == "on":
        if state.get("power") == "on" and not value_type and not attrs:
            return {
                "success": True, "code": 200,
                "message": f"{display}已经开着了",
                "resolved_device_id": dev_id,
                "state_snapshot": state.copy(),
            }
        state["power"] = "on"
        changes.append("已开启")
    elif action == "off":
        if state.get("power") == "off" and not value_type and not attrs:
            return {
                "success": True, "code": 200,
                "message": f"{display}已经关了",
                "resolved_device_id": dev_id,
                "state_snapshot": state.copy(),
            }
        state["power"] = "off"
        changes.append("已关闭")

    # 温度
    if value_type == "temperature" and value is not None:
        if not (16 <= value <= 30):
            return {
                "success": False, "code": 400,
                "message": "温度需在 16-30 度之间",
                "resolved_device_id": dev_id,
                "state_snapshot": state.copy(),
            }
        state["temperature"] = value
        changes.append(f"温度调至 {value} 度")

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

    # 模式
    mode = attrs.get("mode")
    if mode:
        normalized_mode = AC_MODE_NORM.get(mode, mode)
        state["mode"] = normalized_mode
        mode_cn = AC_MODE_NAMES.get(normalized_mode, mode)
        changes.append(f"切换为{mode_cn}模式")

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
        # 支持相对调整：如 "+1"、"-1档"、"调大两档"
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
                delta = int(m.group(2)) * (1 if m.group(1) == '+' else -1)
                base = current_fan if isinstance(current_fan, int) else 2
                normalized_fan = max(1, min(5, base + delta))
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
        changes.append(f"风力设为{fan_cn}")

    # 如果有任何属性变更或开关操作
    if changes:
        # 打开空调时默认设为制冷模式
        if action == "on" and "mode" not in attrs:
            state["mode"] = state.get("mode", "cooling")
        # 只要有属性变更或开启操作，都视为成功
        if action != "on" and action != "off" and state.get("power") != "on":
            state["power"] = "on"

        # 分离开关操作和属性变更
        power_change = None
        other_changes = []
        for c in changes:
            if c in ("已开启", "已关闭"):
                power_change = c
            else:
                other_changes.append(c)

        if power_change:
            prefix = "开启" if power_change == "已开启" else "关闭"
            msg = f"已为您{prefix}{display}"
            if other_changes:
                msg += "，" + "，".join(other_changes)
        else:
            # 纯属性调整（无显式开关），使用"将"字句
            change_text = "，".join(changes)
            msg = f"已为您将{display}{change_text}"

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
        state["temperature"] = max(16, min(30, cur + step))
        state["power"] = "on"
        return {
            "success": True, "code": 200,
            "message": f"已将{display}温度{'升高' if step > 0 else '降低'}至 {state['temperature']} 度",
            "resolved_device_id": dev_id,
            "state_snapshot": state.copy(),
        }

    # 空调 set 动作
    if action == "set":
        return {
            "success": True, "code": 200,
            "message": f"已将{display}设置好",
            "resolved_device_id": dev_id,
            "state_snapshot": state.copy(),
        }

    return {
        "success": False, "code": 500,
        "message": f"抱歉，暂时无法理解对{display}的操作",
        "resolved_device_id": dev_id,
        "state_snapshot": state.copy(),
    }


async def _execute_tv(state: Dict[str, Any], display: str, action: str,
                      attrs: Dict[str, Any], dev_id: str) -> Dict[str, Any]:
    """电视控制：开关"""
    if action == "on":
        if state.get("power") == "on":
            return {
                "success": True, "code": 200,
                "message": f"{display}已经开着了",
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
                "message": f"{display}已经关了",
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
        if value is not None and value != "":
            extra["temperature"] = value
        if fan_mode_val and fan_mode_val != "":
            extra["fan_mode"] = fan_mode_val
            extra["fan_speed"] = fan_mode_val
        if hvac_mode:
            extra["mode"] = hvac_mode
        if swing_mode_val and swing_mode_val != "":
            extra["swing_mode"] = swing_mode_val
        async with state_lock:
            s = _c.deepcopy(DEVICE_STATES.get(dev_id, {}))
        if s.get("type") == "curtain":
            extra["power"] = s.get("state", "closed")
        else:
            extra["power"] = s.get("power", "off")
        if s.get("type") == "ac":
            if "temperature" not in extra:
                t = s.get("temperature")
                if t is not None:
                    extra["temperature"] = t
            m = s.get("mode")
            if m and "mode" not in extra:
                extra["mode"] = m
            fs = s.get("fan_speed")
            if fs is not None and "fan_mode" not in extra and "fan_speed" not in extra:
                extra["fan_speed"] = fs
            sw = s.get("swing")
            if sw and "swing_mode" not in extra:
                extra["swing_mode"] = sw
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
    """
    import copy as _copy
    async with state_lock:
        before_snap = {k: _copy.deepcopy(v) for k, v in DEVICE_STATES.items()}
    result = await _execute_command_impl(intent)
    if result.get("success", False):
        await _publish_changed_mqtt(before_snap, intent)
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
            "value_type": str
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
    dev_name = intent["device_name"]
    action = intent["action"]
    value = intent.get("value")
    value_type = intent.get("value_type")
    attrs = intent.get("attrs") or {}

    # ---- 多设备扩展：curtain.all / light.all 等泛指目标 ----
    if dev_id.endswith(".all"):
        short_type = dev_id[:-4]  # curtain.all -> curtain
        results = []
        messages = []
        async with state_lock:
            targets = [k for k, v in DEVICE_STATES.items() if v["type"] == short_type]
        for target_id in targets:
            sub_intent = {
                "device_id": target_id,
                "device_name": intent.get("device_name", ""),
                "action": action,
                "value": value,
                "value_type": value_type,
                "attrs": attrs,
            }
            sub_result = await _execute_command_impl(sub_intent)
            results.append(sub_result)
            if sub_result.get("message"):
                messages.append(sub_result["message"])
        all_ok = all(r.get("success", False) for r in results)
        return {
            "success": all_ok,
            "code": 200 if all_ok else 207,
            "message": "，".join(messages) if messages else f"已处理所有{short_type}设备",
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

    async with state_lock:
        state = DEVICE_STATES[dev_id]
        dev_type = state["type"]
        display = _display_name(state, dev_name)

        # ============ 空调多属性控制 ============
        if dev_type == "ac":
            return await _execute_ac(state, display, action, value, value_type, attrs, dev_id)

        # ============ 电视多属性控制 ============
        if dev_type == "tv":
            return await _execute_tv(state, display, action, attrs, dev_id)

        # 灯光开关
        if dev_type == "light":
            if action == "on":
                if state.get("power") == "on":
                    return {
                        "success": True, "code": 200,
                        "message": f"{display}已经开着了",
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
                        "message": f"{display}已经关了",
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
                        "message": f"{display}已经拉开了",
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
                        "message": f"{display}已经关闭了",
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

        if action == "on":
            if state.get("power") == "on":
                return {
                    "success": True, "code": 200,
                    "message": f"{display}已经开着了",
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
                    "message": f"{display}已经关了",
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
