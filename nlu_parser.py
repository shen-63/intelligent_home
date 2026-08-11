import re
from typing import Dict, Any, Optional, List, Tuple

from device_controller import DEVICE_STATES

# 会话上下文存储 - 追踪每个会话最后操作的设备
_last_device_context: Dict[str, str] = {}  # session_id -> device_id

# 取消指令关键词
CANCEL_KEYWORDS = ["算了", "不要了", "先不开", "不用了", "算了吧", "算了", "不要", "不用", "取消"]


def set_last_device(session_id: str, device_id: str):
    """记录会话最后操作的设备"""
    _last_device_context[session_id] = device_id


def get_last_device(session_id: str) -> Optional[str]:
    """获取会话最后操作的设备"""
    return _last_device_context.get(session_id)


def clear_last_device(session_id: str):
    """清除会话最后操作的设备"""
    if session_id in _last_device_context:
        del _last_device_context[session_id]


def is_cancel_command(text: str) -> bool:
    """判断是否是取消指令"""
    text = text.strip()
    for kw in CANCEL_KEYWORDS:
        if kw in text:
            return True
    return False


# 设备词典：原始词 -> (设备ID, 是否需要房间限定)
# 按 key 长度倒序排列，确保 "卧室空调" 优先于 "空调" 匹配
DEVICE_MAP: List[Tuple[str, str]] = [
    ("卧室空调", "bedroom_ac"),
    ("客厅空调", "living_room_ac"),
    ("卧室灯", "bedroom_light"),
    ("客厅灯", "living_room_light"),
    ("卧室窗帘", "bedroom_curtain"),
    ("客厅窗帘", "living_room_curtain"),
    ("卧室电视", "bedroom_tv"),
    ("客厅电视", "living_room_tv"),
    ("空调", "air_conditioner"),
    ("灯", "light"),
    ("窗帘", "curtain"),
    ("电视", "tv"),
]

# 通用设备ID -> 候选具体设备ID列表（用于房间消歧）
DEVICE_RESOLUTION: Dict[str, List[str]] = {
    "air_conditioner": ["bedroom_ac", "living_room_ac"],
    "light": ["bedroom_light", "living_room_light"],
    "curtain": ["bedroom_curtain", "living_room_curtain"],
    "tv": ["bedroom_tv", "living_room_tv"],
}

# 设备风险分级（影响消歧策略）
# low_risk: 自动消歧（卧室优先），操作错成本低
# high_risk: 必须询问用户，操作错体验差（隐私/安全）
DEVICE_RISK_LEVELS: Dict[str, str] = {
    "air_conditioner": "low_risk",
    "light": "low_risk",
    "tv": "low_risk",
    "curtain": "high_risk",
}


def get_device_risk(device_id: str) -> str:
    """获取设备的风险等级"""
    return DEVICE_RISK_LEVELS.get(device_id, "low_risk")

# 动作正则匹配规则（顺序即优先级，先匹配先生效）
ACTION_PATTERNS: List[Tuple[str, str]] = [
    (r"(打开|开启|启动|开一下|开|亮)", "on"),
    (r"(关闭|关掉|关上|关一下|关|灭)", "off"),
    (r"(调高|升高|加大|调大|增强)", "increase"),
    (r"(调低|降低|减小|调小|减弱)", "decrease"),
    (r"(调到|调至|设为|设置成|变成)", "set"),
    # 语义触发：根据环境/状态推断动作
    (r"(?:很|太|特别|有点|好|超|巨)(?:黑|暗|黑了|暗了)", "on"),  # 很黑/太暗 → 开灯
    (r"(?:很|太|特别|有点|好|超|巨)(?:亮|晃眼|刺眼)", "off"),  # 太亮 → 关灯/调暗
    (r"(?:很|太|特别|有点|好|超|巨)(?:热|闷|潮热)", "cool"),   # 很热 → 开制冷
    (r"(?:很|太|特别|有点|好|超|巨)(?:冷|冻|凉)", "heat"),     # 很冷 → 开制热
    (r"(?:没|未)(?:开|启动)", "on"),                  # 没开 → 打开
]

# 数值提取规则
VALUE_PATTERNS = [
    (r"(\d+)\s*(?:度|℃|°C)", "temperature"),
    (r"(\d+)\s*(?:%|百分之)", "brightness"),
    (r"(最亮|最暗|最暗|最亮)", "brightness"),
    (r"(全开|全关)", "curtain_state"),
]

# 空调模式匹配
AC_MODE_MAP: List[Tuple[str, str]] = [
    (r"(制冷|降温|凉快)", "cooling"),
    (r"(制热|取暖|加热)", "heating"),
    (r"(除湿|抽湿)", "dehumidify"),
    (r"(送风|通风)", "fan"),
    (r"(自动模式|自动)", "auto"),
]

# 空调扫风模式
AC_SWING_MAP: List[Tuple[str, str]] = [
    (r"(上下扫风|上下摆风|上下)", "vertical"),
    (r"(左右扫风|左右摆风|左右)", "horizontal"),
    (r"(扫风|摆风)", "vertical"),
    (r"(关闭扫风|停止扫风)", "off"),
]

# 空调风力
AC_FAN_MAP: List[Tuple[str, Any]] = [
    (r"(静音档|静音风|静音)", "quiet"),
    (r"(最低风)", "quiet"),
    (r"(1\s*档|一档)", 1),
    (r"(最小风|最小)", 1),
    (r"(2\s*档|二档|低风)", 2),
    (r"(3\s*档|三档|中风)", 3),
    (r"(4\s*档|四档|高风)", 4),
    (r"(5\s*档|五档|最强风|最大风)", 5),
    (r"(?:调到|调至|设为)?最低", 1),
    (r"(?:调到|调至|设为)?最高", 5),
    (r"(?:调到|调至|设为)?最大", 5),
    (r"(自动风|自动档|自动)", "auto"),
]


def _extract_ac_attrs(text: str) -> Dict[str, Any]:
    """提取空调专属属性：模式、扫风、风力"""
    attrs: Dict[str, Any] = {}

    # 模式
    for pattern, mode in AC_MODE_MAP:
        if re.search(pattern, text):
            attrs["mode"] = mode
            break

    # 扫风
    for pattern, swing in AC_SWING_MAP:
        if re.search(pattern, text):
            attrs["swing"] = swing
            break

    # 风力
    for pattern, fan in AC_FAN_MAP:
        if re.search(pattern, text):
            attrs["fan_speed"] = fan
            break

    return attrs


def _match_device(text: str) -> Tuple[Optional[str], Optional[str]]:
    """从文本中匹配设备，返回 (设备ID, 原始匹配词)，匹配不到返回 (None, None)"""
    for raw_name, dev_id in DEVICE_MAP:
        if raw_name in text:
            return dev_id, raw_name
    return None, None


def _match_action(text: str) -> str:
    """从文本中匹配动作，返回动作字符串"""
    for pattern, act in ACTION_PATTERNS:
        if re.search(pattern, text):
            return act
    return "unknown"


def _match_value(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """从文本中提取数值和数值类型，返回 (数值, 类型)"""
    for pattern, val_type in VALUE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            if val_type == "temperature":
                return int(match.group(1)), "temperature"
            elif val_type == "brightness":
                if match.group(1) in ("最亮", "最暗"):
                    return 100 if match.group(1) == "最亮" else 10, "brightness"
                return int(match.group(1)), "brightness"
            elif val_type == " curtain_state":
                return match.group(1), "curtain_state"
    return None, None


# 多意图拆分连接词
SPLIT_PATTERNS: List[str] = [
    r"[，,]",           # 中英文逗号
    r"[、]",             # 顿号
    r"然后",
    r"接着",
    r"顺便",
    r"还有",
    r"另外",
    r"同时",
    r"一并",
    r"都",
]


# 多设备泛化引用：将"所有窗帘"、"全部灯"等映射为 .all 后缀的虚拟设备ID
_ALL_DEVICE_TYPE_MAP: List[Tuple[str, str, str]] = [
    # (匹配模式, 泛化设备ID, 中文显示名)
    (r"(?:所有|全部|一切|所有的)(?:窗帘)", "curtain.all", "所有窗帘"),
    (r"(?:所有|全部|一切|所有的)(?:灯)", "light.all", "所有灯"),
    (r"(?:所有|全部|一切|所有的)(?:空调)", "ac.all", "所有空调"),
    (r"(?:所有|全部|一切|所有的)(?:电视)", "tv.all", "所有电视"),
    (r"(?:全屋|全部)(?:窗帘)", "curtain.all", "所有窗帘"),
    (r"(?:全屋|全部)(?:灯)", "light.all", "所有灯"),
]

# 设备类型 → .all 泛化ID 映射（用于前缀"所有"泛化传播）
_DEVICE_TYPE_TO_ALL: Dict[str, str] = {
    "curtain": "curtain.all",
    "light": "light.all",
    "ac": "ac.all",
    "tv": "tv.all",
}

# 通用泛化量词（前缀中出现时，后续所有设备都泛化）
_ALL_PREFIX_PATTERN = re.compile(r"(?:所有|全部|一切|全屋|统统)")


def _detect_all_device_refs(text: str) -> List[Tuple[int, int, str, str]]:
    """检测文本中的多设备泛化引用，返回 [(start, end, device_id, raw_name), ...]"""
    refs: List[Tuple[int, int, str, str]] = []
    for pattern, dev_id, display in _ALL_DEVICE_TYPE_MAP:
        for m in re.finditer(pattern, text):
            refs.append((m.start(), m.end(), dev_id, display))
    # 去重：短名称被长名称包含时只保留长名称
    filtered: List[Tuple[int, int, str, str]] = []
    for s, e, did, name in sorted(refs, key=lambda x: x[0]):
        overlaps = False
        for fs, fe, _, _ in filtered:
            if s >= fs and e <= fe:
                overlaps = True
                break
        if not overlaps:
            filtered.append((s, e, did, name))
    return filtered


def parse_multi_intent(text: str, session_id: str = "") -> List[Dict[str, Any]]:
    """
    将包含多个设备指令的文本拆分为独立子意图。
    支持上下文追踪：当文本中没有设备词时，使用上一次操作的设备。
    支持取消指令：当文本是取消命令时，自动对上一个设备执行关闭操作。
    支持语义触发：隐式意图（"客厅好黑"→开灯）优先于上下文兜底。
    支持多设备泛化引用："所有窗帘和电视"→ curtain.all + tv.all。

    例如：
    - "把客厅空调关闭，把卧室电视关闭"
      → [("客厅空调", "off"), ("卧室电视", "off")]
    - 用户之前操作了卧室空调，现在说"调到24度"
      → [("bedroom_ac", "set", temperature=24)]
    - "客厅好黑" → 语义触发
      → [("living_room_light", "on")]
    - "打开所有窗帘和电视"
      → [("curtain.all", "on"), ("tv.all", "on")]
    """
    text = text.strip()

    # 只过滤纯咨询类语句（有疑问词 + 没有动作词）
    question_markers = [
        r'怎么办$', r'怎么(解决|处理|回事)', r'为什么', r'啥原因', r'什么原因',
        r'请教一下', r'问一下', r'请问', r'如何', r'为何', r'咋样',
        r'的原因', r'不(制冷|制热|工作|亮|开|关)',
        r'怎么样', r'怎么(样)', r'是什么意思',
    ]
    action_markers = [
        r'帮我', r'给我', r'请(把|将)', r'就.*(打开|开|关|调节|设置|切换)',
        r'如果.*就', r'没.*就', r'不.*就', r'没开.*开', r'如果.*没',
        r'打开$', r'关上$', r'关掉$', r'关闭$', r'开启$', r'调节$', r'设置$',
    ]

    has_question = any(re.search(p, text) for p in question_markers)
    has_action = any(re.search(p, text) for p in action_markers)

    # 如果有疑问但没有明确的动作词，返回空（不执行设备控制，交给 Dify 知识问答处理）
    if has_question and not has_action:
        return []

    # 检查是否是取消指令
    if is_cancel_command(text) and session_id:
        last_dev = get_last_device(session_id)
        if last_dev:
            device_name_map = {
                "bedroom_ac": "卧室空调", "living_room_ac": "客厅空调",
                "bedroom_light": "卧室灯", "living_room_light": "客厅灯",
                "bedroom_curtain": "卧室窗帘", "living_room_curtain": "客厅窗帘",
                "bedroom_tv": "卧室电视", "living_room_tv": "客厅电视",
            }
            return [{
                "device_id": last_dev,
                "device_name": device_name_map.get(last_dev, last_dev),
                "action": "off",
                "value": None,
                "value_type": None,
                "attrs": {},
            }]

    # ========== 语义触发：优先级最高，在上下文兜底之前 ==========
    # "很黑"/"太暗"/"好黑" → 灯；"很热" → 空调(cool)；"很冷" → 空调(heat)
    semantic_device_map = [
        # 光照 → 灯
        (r"(?:很|太|特别|有点|好|超|巨)(?:黑|暗|黑了|暗了)", "light", "on"),
        (r"(?:没|未)(?:开|亮)", "light", "on"),
        (r"(?:很|太|特别|有点|好|超|巨)(?:亮|晃眼|刺眼)", "light", "off"),
        # 温度 → 空调
        (r"(?:很|太|特别|有点|好|超|巨)(?:热|闷|潮热)", "ac", "on", "cool"),
        (r"(?:很|太|特别|有点|好|超|巨)(?:冷|冻|凉)", "ac", "on", "heat"),
        # 补充模式词
        (r"(?:没|不)(?:制冷|制热)", "ac", "on", "cool"),
        (r"(?:空调|冷气).*(?:没|不|未).*(?:工作|启动|反应|亮)", "ac", "on", "cool"),
        (r"(?:空调|冷气).*(?:需要|想要|想|要)(?:开|启动|打开)", "ac", "on", "cool"),
    ]

    for pattern, dev_type, act, *rest in semantic_device_map:
        if re.search(pattern, text):
            candidates = [(k, v) for k, v in DEVICE_STATES.items() if v.get("type") == dev_type]
            if candidates:
                mentioned_bedroom = bool(re.search(r'卧室|房间|屋里|里面', text))
                mentioned_livingroom = bool(re.search(r'客厅|大厅|外面', text))

                if mentioned_bedroom:
                    room_devices = [(k, v) for k, v in candidates if v.get("room") == "卧室"]
                    target = room_devices[0] if room_devices else candidates[0]
                elif mentioned_livingroom:
                    room_devices = [(k, v) for k, v in candidates if v.get("room") == "客厅"]
                    target = room_devices[0] if room_devices else candidates[0]
                else:
                    living_room = [(k, v) for k, v in candidates if v.get("room") == "客厅"]
                    target = living_room[0] if living_room else candidates[0]

                dev_id = target[0]
                dev_state = target[1]
                room = dev_state.get("room", "")
                type_names = {"ac": "空调", "light": "灯", "tv": "电视", "curtain": "窗帘"}
                display = room + type_names.get(dev_type, "设备")

                intent: Dict[str, Any] = {
                    "device_id": dev_id,
                    "device_name": display,
                    "action": act,
                    "value": None,
                    "value_type": None,
                    "attrs": {},
                }
                if rest:
                    intent["attrs"]["mode"] = rest[0]
                print(f"[NLU] 语义触发: '{text}' → {display} ({act})")
                if session_id:
                    set_last_device(session_id, dev_id)
                return [intent]

    # ========== 找到所有设备出现的位置 ==========
    device_positions: List[Tuple[int, int, str, str]] = []

    # 1. 多设备泛化引用（"所有窗帘"、"全部灯"等）
    all_refs = _detect_all_device_refs(text)
    device_positions.extend(all_refs)

    # 2. 具体设备名（去重：短名称被长名称或泛化引用包含时只保留长名称）
    for raw_name, dev_id in DEVICE_MAP:
        start = 0
        while True:
            pos = text.find(raw_name, start)
            if pos == -1:
                break
            end = pos + len(raw_name)
            overlaps = False
            for s, e, _, _ in device_positions:
                if pos >= s and end <= e:
                    overlaps = True
                    break
            if not overlaps:
                device_positions.append((pos, end, dev_id, raw_name))
            start = pos + len(raw_name)

    # 如果没有找到设备，检查是否是上下文指令（如"调到24度"）
    if not device_positions and session_id:
        last_dev = get_last_device(session_id)
        if last_dev:
            result = parse_intent_with_device(text, last_dev)
            if result["success"] and result["intent"]:
                return [result["intent"]]
            device_name_map = {
                "bedroom_ac": "卧室空调", "living_room_ac": "客厅空调",
                "bedroom_light": "卧室灯", "living_room_light": "客厅灯",
                "bedroom_curtain": "卧室窗帘", "living_room_curtain": "客厅窗帘",
                "bedroom_tv": "卧室电视", "living_room_tv": "客厅电视",
            }
            dev_name = device_name_map.get(last_dev, "")
            if dev_name:
                combined_text = f"{dev_name} {text}"
                result = parse_intent(combined_text)
                if result["success"] and result["intent"]:
                    return [result["intent"]]

    intents: List[Dict[str, Any]] = []

    if not device_positions:
        return []

    # 按位置排序
    device_positions.sort(key=lambda x: x[0])

    # 前缀泛化传播：如果文本包含"所有/全部/全屋"等量词，
    # 且存在多个设备，将非泛化的通用设备ID（如 tv, curtain）也泛化到 .all
    # 注意：不处理具体房间设备（如 living_room_tv），避免过度泛化
    if len(device_positions) > 1:
        if _ALL_PREFIX_PATTERN.search(text):
            propagated: List[Tuple[int, int, str, str]] = []
            for pos, end, dev_id, raw_name in device_positions:
                if not dev_id.endswith(".all"):
                    if dev_id in _DEVICE_TYPE_TO_ALL:
                        all_id = _DEVICE_TYPE_TO_ALL[dev_id]
                        propagated.append((pos, end, all_id, f"所有{raw_name}"))
                        continue
                propagated.append((pos, end, dev_id, raw_name))
            device_positions = propagated

    if len(device_positions) == 1:
        pos, end_pos, dev_id, raw_name = device_positions[0]
        # 泛化引用（.all 后缀）直接用整句解析，让 _match_action 提取动作
        if dev_id.endswith(".all"):
            result = parse_intent(text)
            if result["success"] and result["intent"]:
                intent = result["intent"]
                # 强制替换为泛化设备ID
                intent["device_id"] = dev_id
                intent["device_name"] = raw_name
                if session_id:
                    set_last_device(session_id, dev_id)
                return [intent]
            return []

        # 单设备，直接用整句解析
        result = parse_intent(text)
        if result["success"] and result["intent"]:
            intent = result["intent"]
            if session_id:
                set_last_device(session_id, intent.get("device_id", ""))
            return [intent]
        return []

    # 多设备：按位置切分文本片段
    # 提取首设备之前的前缀文本（含公共动作，如"打开所有窗帘和电视"的"打开"）
    first_pos = device_positions[0][0]
    prefix_text = text[:first_pos]
    prefix_action = _match_action(prefix_text) if prefix_text else "unknown"
    if prefix_action == "unknown":
        prefix_action = None

    prev_action = prefix_action  # 初始化为前缀中的动作
    for i, (pos, end_pos, dev_id, raw_name) in enumerate(device_positions):
        if i + 1 < len(device_positions):
            next_pos = device_positions[i + 1][0]
            sub_text = text[pos:next_pos]
        else:
            sub_text = text[pos:]

        # 去掉开头的连接词
        sub_text = re.sub(r"^[，,、\s]*(?:然后|接着|顺便|还有|另外|同时|一并|都)?[，,、\s]*", "", sub_text)
        # 去掉结尾的连接词
        sub_text = re.sub(r"[，,、\s]*(?:然后|接着|顺便|还有|另外|同时|一并|都)[，,、\s]*$", "", sub_text)

        if not sub_text.strip():
            sub_text = raw_name

        result = parse_intent(sub_text)
        if result["success"] and result["intent"]:
            # 如果是泛化引用，强制替换 device_id
            if dev_id.endswith(".all"):
                result["intent"]["device_id"] = dev_id
                result["intent"]["device_name"] = raw_name
            intents.append(result["intent"])
            prev_action = result["intent"].get("action")
        elif prev_action:
            # 子片段无动作，复用上一个/前缀的动作
            fallback_intent = {
                "device_id": dev_id,
                "device_name": raw_name,
                "action": prev_action,
                "value": None,
                "value_type": None,
                "attrs": {},
            }
            intents.append(fallback_intent)

    # 记录最后操作的设备
    if intents and session_id:
        set_last_device(session_id, intents[-1].get("device_id", ""))

    return intents


def parse_intent_with_device(text: str, device_id: str) -> Dict[str, Any]:
    """用指定设备解析指令"""
    device_name_map = {
        "bedroom_ac": "卧室空调", "living_room_ac": "客厅空调",
        "bedroom_light": "卧室灯", "living_room_light": "客厅灯",
        "bedroom_curtain": "卧室窗帘", "living_room_curtain": "客厅窗帘",
        "bedroom_tv": "卧室电视", "living_room_tv": "客厅电视",
    }
    dev_name = device_name_map.get(device_id, "")
    if not dev_name:
        return parse_intent(text)
    combined_text = f"{dev_name} {text}"
    return parse_intent(combined_text)


def parse_intent(text: str) -> Dict[str, Any]:
    """
    基于规则的意图解析。
    输出 schema 稳定，后续替换为 LLM 实现时保持此接口不变。

    Returns:
        {
            "success": bool,
            "msg": str,
            "method": str,        # "rule" 或将来的 "llm"
            "confidence": float,  # 0.0 ~ 1.0
            "need_clarification": bool,  # 是否需要用户确认
            "intent": {
                "raw_text": str,
                "device_id": str,
                "device_name": str,
                "action": str,
                "value": Any,
                "value_type": str
            } | None
        }
    """
    text = text.strip()
    target_device_id, target_device_raw = _match_device(text)
    action = _match_action(text)
    value, value_type = _match_value(text)

    # 未识别到设备
    if not target_device_id:
        return {
            "success": False,
            "msg": f"未能识别到已知设备，原始文本：'{text}'",
            "method": "rule",
            "confidence": 0.0,
            "need_clarification": False,
            "intent": None,
        }

    # 通用设备（无房间限定）需要消歧
    need_clarification = target_device_id in DEVICE_RESOLUTION

    # 提取空调专属属性
    ac_attrs: Dict[str, Any] = {}
    if target_device_id.endswith("_ac") or target_device_id == "air_conditioner":
        ac_attrs = _extract_ac_attrs(text)

    # 构建 intent 基础结构
    base_intent = {
        "raw_text": text,
        "device_id": target_device_id,
        "device_name": target_device_raw,
        "action": action,
        "value": value,
        "value_type": value_type,
        "attrs": ac_attrs,
    }

    # 未识别到动作，但有额外属性
    if action == "unknown":
        if ac_attrs:
            # 有空调属性也算有效意图
            base_intent["action"] = "set"
        else:
            return {
                "success": False,
                "msg": f"识别到设备「{target_device_raw}」但未识别到操作意图",
                "method": "rule",
                "confidence": 0.3,
                "need_clarification": need_clarification,
                "intent": base_intent,
            }

    confidence = 0.6 if need_clarification else 0.9

    return {
        "success": True,
        "msg": "意图解析成功",
        "method": "rule",
        "confidence": confidence,
        "need_clarification": need_clarification,
        "intent": base_intent,
    }


def resolve_device_id(device_id: str, prefer_room: str = "bedroom") -> str:
    """
    将通用设备ID解析为具体设备ID。
    当存在多个房间候选时，默认选卧室。
    后续可接入上下文记忆或用户偏好。
    """
    if device_id in DEVICE_RESOLUTION:
        candidates = DEVICE_RESOLUTION[device_id]
        for c in candidates:
            if prefer_room in c:
                return c
        return candidates[0]
    return device_id
