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

IntentType = Literal["device_control", "knowledge_qa"]

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
    "扫风", "摆风",
]

# 数值关键词（温度、档位等，通常伴随设备控制）
_VALUE_KEYWORDS = [
    r'\d+\s*度', r'\d+\s*档', r'\d+\s*%',
    "最亮", "最暗", "全开", "全关", "静音",
    "制冷", "制热", "除湿", "送风", "自动模式",
    "上下扫风", "左右扫风", "一档", "二档", "三档", "四档", "五档",
]

# 环境感知词（"很热"、"太暗"等隐式触发设备控制）
# 注意：此列表与 nlu_parser 的语义触发器保持一致
_ENV_KEYWORDS = [
    r'(?:很|太|特别|有点|好|超|巨)(?:热|闷|冷|冻|凉|黑|暗|亮|晃眼|刺眼)',
    r'(?:没|未)(?:开|亮|启动)',
    r'(?:需要|想要|想|要)(?:开|启动|打开)',
    r'(?:空调|灯|电视|窗帘).*(?:没|未|不)(?:工作|开|启动|反应|亮)',
]

# 房间关键词（与环境词配合提高置信度）
_ROOM_KEYWORDS = ["卧室", "主卧", "房间", "屋内", "里面", "客厅", "大厅", "起居室"]

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


def route_intent(user_text: str) -> IntentType:
    """
    意图路由：判断用户输入是设备控制还是知识问答。

    策略：
    1. 优先检测知识问答信号（疑问句式 + 无动作词 → 知识问答）
    2. 检测设备控制信号（设备词/动作词/场景词/数值/环境感知 → 设备控制）
    3. 默认安全降级为知识问答

    Returns:
        "device_control" 或 "knowledge_qa"
    """
    if not user_text or not user_text.strip():
        return "knowledge_qa"

    text = user_text.strip()

    # ---- 检测设备控制信号 ----
    has_device = any(kw in text for kw in _DEVICE_KEYWORDS)
    has_action = any(kw in text for kw in _ACTION_KEYWORDS)
    has_scene = any(kw in text for kw in _SCENE_KEYWORDS)
    has_value = any(re.search(p, text) for p in _VALUE_KEYWORDS)
    has_env = any(re.search(p, text) for p in _ENV_KEYWORDS)

    # ---- 检测知识问答信号 ----
    has_question = any(re.search(p, text) for p in _QUESTION_PATTERNS)
    has_knowledge_domain = any(kw in text for kw in _KNOWLEDGE_DOMAIN_KEYWORDS)

    # ---- 天气查询：有天气词即可走知识问答轨 ----
    has_weather = any(kw in text for kw in _WEATHER_KEYWORDS)
    if has_weather and not has_device:
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
    # 注意：has_env 和 has_room 在函数开头统一计算
    has_env = any(re.search(p, text) for p in _ENV_KEYWORDS)
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

    # ---- 默认降级：无法明确判断时走知识问答 ----
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

    signals = {
        "has_device": any(kw in text for kw in _DEVICE_KEYWORDS),
        "has_action": any(kw in text for kw in _ACTION_KEYWORDS),
        "has_scene": any(kw in text for kw in _SCENE_KEYWORDS),
        "has_value": any(re.search(p, text) for p in _VALUE_KEYWORDS),
        "has_env": any(re.search(p, text) for p in _ENV_KEYWORDS),
        "has_room": any(kw in text for kw in _ROOM_KEYWORDS),
        "has_question": any(re.search(p, text) for p in _QUESTION_PATTERNS),
        "has_knowledge_domain": any(kw in text for kw in _KNOWLEDGE_DOMAIN_KEYWORDS),
        "has_weather": any(kw in text for kw in _WEATHER_KEYWORDS),
    }

    intent = route_intent(text)

    # 置信度估算
    if intent == "device_control":
        score = 0.0
        if signals["has_device"]:
            score += 0.3
        if signals["has_action"]:
            score += 0.3
        if signals["has_scene"]:
            score += 0.4
        if signals["has_value"]:
            score += 0.2
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
