"""
意图路由模块 - 极轻量的意图判断，决定请求走设备控制轨还是知识问答轨。

设计原则：
- 纯关键词/正则匹配，零 API 调用，耗时 <0.1ms
- 命中率目标 95%+，未命中时默认走知识问答（安全降级）
- 复用 nlu_parser 已有的设备/动作词典，保持一致性
"""
import re
from typing import Literal

from nlu_parser import DEVICE_MAP, ACTION_PATTERNS

IntentType = Literal["device_control", "knowledge_qa", "device_query"]

# ==================== 状态查询特征词（最高优先级：查询绝对不是控制，严禁发 MQTT）====================
# Type 1：单设备开关状态问句（句尾疑问/反诘）
_DEVICE_QUERY_POWER_PATTERNS = [
    r'开着(?:嘛|吗|么|呐|没)',            # 开着嘛 / 开着吗 / 开着没
    r'关(?:了|着)(?:嘛|吗|么|没|呐)',      # 关了吗 / 关着没 / 关了没
    r'开(?:了|过)(?:嘛|吗|么|没|呐)',      # 开了吗 / 开了没
    r'是不是(?:开|关|亮|灭)着(?:的|呀)?',
    r'有没(?:有|开|关)',
    r'(?:现在|目前|这会儿|此刻).*(?:开|关|亮|灭).*吗',
    # 条件查询常见开头（强信号：用户想先知道状态）
    r'^看看.{0,6}(?:开|关|亮)',
    r'^查(?:查|一下|看).{0,6}(?:开|关|亮|温度|度|档位|模式|风速)',
]

# Type 2：列表/聚合状态问句（哪些灯亮着 / 哪些设备开着 等）
_DEVICE_QUERY_LIST_PATTERNS = [
    r'有哪些(?:灯|设备|家电|电器|空调)(?:亮着|开着|关着|在开|在关|运行)?',
    r'(?:哪些|什么|啥)(?:灯|设备|家电|空调)(?:现在|目前)?(?:亮着|开着|关着|在开|在关|开的|关的)',
    r'(?:灯|设备|家电).*有哪些(?:亮|开|关)',
    r'家里.*(?:灯|设备).*(?:开着|亮着)',
    r'(?:家里|屋|房间).*什么.*(?:灯|设备).*(?:开|关)',
]

# Type 3：属性数值问句（多少度 / 几度 / 几档 / 什么模式 / 风速多少 等）
_DEVICE_QUERY_ATTR_PATTERNS = [
    r'(?:是|有|现在|目前).*(?:多少|几).*(?:度|档|级|度设定)',
    r'(?:是|有|现在|目前).*什么(?:模式|风向|扫风|风)',
    r'(?:温度|风速|风力|风量|亮度|音量|声量|频道).*(?:是|多少|几|多大|多高)',
    r'(?:现在|目前|这会儿).*(?:温度|风速|风力|档位|模式)',
    r'(?:是|为).*(?:几度|几档|多少度|多少档)',
    r'多少度$', r'几度$', r'几档$',
    r'什么模式$',
]

# ==================== 设备控制关键词 ====================
# 从 nlu_parser 的 DEVICE_MAP 中提取所有中文设备名
_DEVICE_KEYWORDS = [raw_name for raw_name, _ in DEVICE_MAP]

# 场景关键词（触发场景模式，属于设备控制）
_SCENE_KEYWORDS = [
    "睡觉", "晚安", "睡眠", "睡了", "休息", "要睡", "入眠",
    "出门", "离家", "外出", "上班", "离开", "全关", "全部关闭",
    "看电影", "电影模式", "观影",
]

# 动作词（用户想要操作设备）
_ACTION_KEYWORDS = [
    "打开", "开启", "启动", "关掉", "关闭", "关上", "关",
    "调到", "调至", "设为", "设置", "调节", "调整", "切换",
    "拉开", "拉上", "开", "灭", "亮",
    "调高", "调低", "升高", "降低", "加大", "减小", "增强", "减弱",
    "扫风", "摆风", "风力", "风速", "风档",
]

# 数值关键词（温度、档位等，通常伴随设备控制）
_VALUE_KEYWORDS = [
    r'\d+\s*度', r'\d+\s*档', r'\d+\s*%',
    "最亮", "最暗", "全开", "全关", "静音",
    "制冷", "制热", "除湿", "送风", "自动模式",
    "上下扫风", "左右扫风", "一档", "二档", "三档", "四档", "五档",
]

# 空调/灯属性专属关键词（"风力1档"、"制冷"、"扫风"等无设备名但肯定是设备控制）
_AC_ATTRIBUTE_KEYWORDS = [
    "风力", "风速", "风档", "档位",
    "制冷", "制热", "除湿", "送风", "模式",
    "扫风", "摆风", "风向",
]

# ==================== 相对控制（增量控制）信号 ====================
# 描述"调高/调低"这类相对变化的词，通常伴随属性（温度、风速、亮度、音量等）
_DELTA_ACTION_KEYWORDS = [
    "调高", "调低", "升高", "降低",
    "加大", "减小", "调大", "调小",
    "增强", "减弱",
    "大一点", "小一点", "大一些", "小一些",
    "高一点", "低一点", "高一些", "低一些",
    "亮一点", "暗一点", "亮一些", "暗一些",
    "大一档", "小一档", "升一档", "降一档",
    "往上调", "往下调", "向上调", "向下调",
]

# 可做"增量调节"的属性关键词（一旦与相对词同时出现，必是设备控制）
_DELTA_ATTRIBUTE_KEYWORDS = [
    # 空调/风扇类
    "温度", "风速", "风力", "风量", "风档", "档位",
    # 灯类
    "亮度", "明度",
    # 音频/TV类
    "音量", "声音", "声量",
    # 通用
    "度数", "温度值",
]

# 属性 -> 强绑定设备类型（避免"风速"被误识别为"灯"）
_ATTR_DEVICE_TYPE_BINDING: dict = {
    # 空调专属（任何匹配即设备类型=ac）
    "风速": "ac", "风力": "ac", "风量": "ac", "风档": "ac",
    "扫风": "ac", "摆风": "ac", "风向": "ac",
    "温度": "ac", "制冷": "ac", "制热": "ac", "除湿": "ac", "送风": "ac",
    # 灯专属
    "亮度": "light", "明度": "light",
    # TV专属
    "音量": "tv", "声音": "tv", "声量": "tv", "频道": "tv",
}

# 可被"属性推断设备"的泛化/具体设备ID白名单
_ROOM_KEYWORDS = ["卧室", "主卧", "房间", "屋内", "里面", "客厅", "大厅", "起居室"]

# 环境感知词（"很热"、"太暗"、"太吵"等隐式触发设备控制）
# 注意：此列表与 nlu_parser 的语义触发器保持一致
_ENV_KEYWORDS = [
    r'(?:很|太|特别|有点|好|超|巨)(?:热|闷|冷|冻|凉|黑|暗|亮|晃眼|刺眼|吵|响|闹|嘈杂)',
    r'(?:声音|音量|声响).*(?:太|很|过|特别|有点)?(?:大|高|吵|响)',
    r'(?:没|未)(?:开|亮|启动)',
    r'(?:需要|想要|想|要)(?:开|启动|打开)',
    r'(?:空调|灯|电视|窗帘).*(?:没|未|不)(?:工作|开|启动|反应|亮)',
]

# ==================== 天气查询关键词 ====================
_WEATHER_KEYWORDS = ["天气", "气温", "温度", "下雨", "刮风", "台风", "降雪", "雷电", "雾霾"]

# ==================== 知识问答关键词 ====================
# 疑问句式（明确的知识问答意图）
_QUESTION_PATTERNS = [
    r'怎么\s*(?:办|解决|处理|回事|搞)',
    r'怎么(?:样|样)',  # 今天天气怎么样
    r'为什么',
    r'啥原因',
    r'什么原因',
    r'请教一下',
    r'问一下',
    r'请问',
    r'如何\s*(?:解决|处理|操作|设置)',
    r'为何',
    r'咋办',
    r'是什么',
    r'是什么意思',  # E1故障码是什么意思
    r'怎么读',
    r'什么叫',
    r'区别',
    r'原理',
    r'多少',  # 多少钱/多少度
    r'多久',  # 多久清洗一次
]

# 知识领域关键词（家电故障、保养等）
_KNOWLEDGE_DOMAIN_KEYWORDS = [
    "故障", "错误", "报错", "代码", "保养", "清洁", "清洗",
    "维护", "寿命", "更换", "滤网", "制冷剂", "氟",
    "不制冷", "不制热", "不工作", "不启动", "漏水",
    "异响", "噪音", "发霉", "异味",
    "说明书", "保修", "售后",
]


def _is_device_query(text: str) -> bool:
    """内部辅助：判断文本是否属于【纯状态查询】，命中则任何情况下绝不发 MQTT。"""
    for p in _DEVICE_QUERY_POWER_PATTERNS:
        if re.search(p, text):
            return True
    for p in _DEVICE_QUERY_LIST_PATTERNS:
        if re.search(p, text):
            return True
    for p in _DEVICE_QUERY_ATTR_PATTERNS:
        if re.search(p, text):
            return True
    return False


def route_intent(user_text: str) -> IntentType:
    """
    意图路由：判断请求走【设备查询(device_query)】/【设备控制(device_control)】/【知识问答(knowledge_qa)】。

    策略（优先级严格从高到低）：
    0. 🚦【绝对最高：纯状态查询】→ device_query（只读内存状态，绝不发 MQTT）
       - 覆盖 3 类：单设备开关问句 / 聚合列表问句 / 属性数值问句
    1. 💥相对控制指令（调高/调低 + 属性词）→ 强制 device_control
    2. 属性强绑定命中（温度/风速/制冷/制热/亮度/音量）→ 强制 device_control
    3. 优先检测知识问答信号（疑问句式 + 知识领域 → knowledge_qa）
    4. 检测设备控制信号（设备词/动作词/场景词/数值/环境感知 → 设备控制）
    5. 默认：非疑问句一律走设备控制，疑问句走知识问答

    Returns:
        "device_query" | "device_control" | "knowledge_qa"
    """
    if not user_text or not user_text.strip():
        return "knowledge_qa"

    text = user_text.strip()

    # ==================== 【0. 绝对最高优先级：纯设备状态查询】 ====================
    #    （在增量控制/属性绑定 之前拦截！避免"开着吗/多少度"命中 has_action 后被误路由）
    if _is_device_query(text):
        return "device_query"

    # ==================== 【1. 增量/相对控制】 ====================
    # 相对词（调高、调小、大一点 等）
    has_delta = any(kw in text for kw in _DELTA_ACTION_KEYWORDS)
    # 可增量调节的属性词（温度、风速、亮度、音量 等）
    has_delta_attr = any(kw in text for kw in _DELTA_ATTRIBUTE_KEYWORDS)
    # 属性-设备强绑定（比如出现"风速"→设备是空调；出现"亮度"→是灯）
    bound_type_list = [t for kw, t in _ATTR_DEVICE_TYPE_BINDING.items() if kw in text]
    has_attr_binding = len(bound_type_list) > 0

    # ✅ 规则1：相对词 + 属性词 → 100% 是设备控制（例如 "卧室温度调高2度"）
    #          严禁进入 RAG 聊天，否则会答"调高温度可以省电..."
    if has_delta and has_delta_attr:
        return "device_control"
    # ✅ 规则2：相对词 + 有属性绑定 → 也是设备控制（例 "风速调大一些"，即使未写"温度"二字）
    if has_delta and has_attr_binding:
        return "device_control"
    # ✅ 规则3：属性绑定命中（例如含"温度/风速/制冷/制热"）且不是明确疑问句 → 设备控制
    #          避免"卧室风速调大一档"被误判成卧室灯
    if has_attr_binding and not any(re.search(p, text) for p in _QUESTION_PATTERNS):
        return "device_control"

    # ---- 检测设备控制信号 ----
    has_device = any(kw in text for kw in _DEVICE_KEYWORDS)
    has_action = any(kw in text for kw in _ACTION_KEYWORDS)
    has_scene = any(kw in text for kw in _SCENE_KEYWORDS)
    has_value = any(re.search(p, text) for p in _VALUE_KEYWORDS)
    has_env = any(re.search(p, text) for p in _ENV_KEYWORDS)
    has_ac_attr = any(kw in text for kw in _AC_ATTRIBUTE_KEYWORDS)
    # 空调属性+档位 也视为"有值"，如"风力低档"、"静音模式"
    has_value = has_value or (has_ac_attr or has_attr_binding) and (
        re.search(r'[低中高大小一二三12345]{1,2}档', text) or "静音" in text or "模式" in text)

    # ---- 检测知识问答信号 ----
    has_question = any(re.search(p, text) for p in _QUESTION_PATTERNS)
    has_knowledge_domain = any(kw in text for kw in _KNOWLEDGE_DOMAIN_KEYWORDS)

    # ---- 天气查询：有天气词且不是"室温调节"场景才走知识问答 ----
    #   （注意："温度"二字在 _WEATHER_KEYWORDS 里，但"卧室温度调高2度"显然不是查天气）
    has_weather = any(kw in text for kw in _WEATHER_KEYWORDS)
    if has_weather and not has_device and not has_attr_binding and not has_delta and not has_delta_attr:
        return "knowledge_qa"

    # 优先检测知识问答：有疑问词 + 知识领域关键词 → 知识问答（即使提到设备名）
    # 如"空调不制冷怎么办"、"E1故障是什么"
    if has_question and has_knowledge_domain:
        return "knowledge_qa"

    # 强设备控制信号：有设备词 + 有动作词（且无疑问词）
    if has_device and has_action and not has_question:
        return "device_control"

    # 场景触发（如"我要睡觉了"、"出门了"）
    if has_scene:
        return "device_control"

    # 环境感知触发（如"太热了"、"很黑"）
    has_room = any(kw in text for kw in _ROOM_KEYWORDS)

    # 环境词 + 房间名 → 高置信度设备控制（如"客厅有点热"）
    if has_env and has_room:
        return "device_control"

    # 环境感知触发（如"太热了"、"很黑"）
    if has_env:
        return "device_control"

    # 有设备词 + 有数值（如"空调 26度"）
    if has_device and has_value:
        return "device_control"

    # 有动作词 + 有数值（如"调到26度"）
    if has_action and has_value:
        return "device_control"

    # 有空调属性关键词（如"风力1档"、"制冷模式"、"左右扫风"）→ 设备控制
    if has_ac_attr and not has_question:
        return "device_control"

    # 有数值（数字/档位）且不是疑问句 → 设备控制（等待上下文继承设备名）
    if has_value and not has_question:
        return "device_control"

    # ---- 检测知识问答信号（补充检测）----
    # 明确的疑问句 → 知识问答
    if has_question and not has_action:
        return "knowledge_qa"

    # 知识领域关键词 + 无设备词 → 知识问答
    if has_knowledge_domain and not has_device:
        return "knowledge_qa"

    # ---- 模糊地带：有动作词但无设备词 ----
    # 如"开一下"、"关了" — 可能是上下文续控，归为设备控制
    if has_action and not has_question:
        return "device_control"

    # ---- 默认降级：非疑问句一律走设备控制，疑问句才走知识问答 ----
    if not has_question:
        return "device_control"
    return "knowledge_qa"


def route_intent_with_confidence(user_text: str) -> dict:
    """
    带置信度的意图路由，用于调试和日志。

    Returns:
        {
            "intent": "device_control" | "knowledge_qa",
            "confidence": float,  # 0.0 ~ 1.0
            "signals": dict,     # 检测到的信号明细
        }
    """
    if not user_text or not user_text.strip():
        return {"intent": "knowledge_qa", "confidence": 0.0, "signals": {}}

    text = user_text.strip()

    # 与 route_intent 完全一致的信号抽取
    has_delta = any(kw in text for kw in _DELTA_ACTION_KEYWORDS)
    has_delta_attr = any(kw in text for kw in _DELTA_ATTRIBUTE_KEYWORDS)
    bound_type_list = [t for kw, t in _ATTR_DEVICE_TYPE_BINDING.items() if kw in text]
    has_attr_binding = len(bound_type_list) > 0
    has_question = any(re.search(p, text) for p in _QUESTION_PATTERNS)
    # 【★ 新增】状态查询信号（最高优先级的 True/False）
    has_device_query = _is_device_query(text)

    signals = {
        "has_device": any(kw in text for kw in _DEVICE_KEYWORDS),
        "has_action": any(kw in text for kw in _ACTION_KEYWORDS),
        "has_scene": any(kw in text for kw in _SCENE_KEYWORDS),
        "has_value": any(re.search(p, text) for p in _VALUE_KEYWORDS),
        "has_env": any(re.search(p, text) for p in _ENV_KEYWORDS),
        "has_room": any(kw in text for kw in _ROOM_KEYWORDS),
        "has_question": has_question,
        "has_knowledge_domain": any(kw in text for kw in _KNOWLEDGE_DOMAIN_KEYWORDS),
        "has_weather": any(kw in text for kw in _WEATHER_KEYWORDS),
        "has_ac_attr": any(kw in text for kw in _AC_ATTRIBUTE_KEYWORDS),
        # 增量/相对控制 信号
        "has_delta": has_delta,
        "has_delta_attr": has_delta_attr,
        "has_attr_binding": has_attr_binding,
        "bound_device_types": list(set(bound_type_list)),
        # 【★ 新增】纯状态查询信号
        "has_device_query": has_device_query,
    }

    intent = route_intent(text)

    # 置信度估算
    if intent == "device_query":
        # device_query 是特征词强匹配 → 置信度拉满
        confidence = 0.95 if has_device_query else 0.5
    elif intent == "device_control":
        score = 0.0
        if signals["has_device"]:
            score += 0.3
        if signals["has_action"]:
            score += 0.3
        if signals["has_scene"]:
            score += 0.4
        if signals["has_value"]:
            score += 0.2
        if signals["has_ac_attr"]:
            score += 0.35  # 空调属性是强设备控制信号
        if signals["has_env"] and signals["has_room"]:
            score += 0.8  # 环境词+房间名 → 高置信度
        elif signals["has_env"]:
            score += 0.3
        confidence = min(score, 1.0)
    else:
        score = 0.0
        if signals["has_question"]:
            score += 0.4
        if signals["has_knowledge_domain"]:
            score += 0.4
        if signals["has_weather"]:
            score += 0.3
        if not signals["has_device"] and not signals["has_action"]:
            score += 0.2
        confidence = min(score, 1.0)

    return {
        "intent": intent,
        "confidence": round(confidence, 2),
        "signals": signals,
    }
