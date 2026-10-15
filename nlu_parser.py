import re
from typing import Dict, Any, Optional, List, Tuple

from device_controller import DEVICE_STATES


# ================================================================
# 【统一语音/文本预处理管道】preprocess_voice_text
#   所有入口（rule parse / llm parse / voice_control / asr）都应先过本函数，
#   确保 90+ 条热词/同音词典 + 填充词/标点清理一致生效。
#
# 处理顺序严格按 spec：
#   1) 清洗与标点清理（首尾空格/零宽字符/标点）
#   2) 剥离唤醒词前缀（按需求若后续接入流式端上唤醒可忽略，此层只处理文本残留的唤醒词）
#   3) 剥离口语填充词/礼貌词前缀
#   4) 热词纠错 (asr_service.post_process_text)
#   5) 高频同音字纠错 (asr_phonetic_correct)
#   6) 收尾：再清理一遍可能因替换引入的首尾标点/空格，归一为空串则返回 ""
# ================================================================

# 唤醒词（按长度降序，"嘿Siri" 这种含空格的也要能匹配）
_WAKE_WORDS: Tuple[str, ...] = (
    "嘿 siri", "嘿siri",
    "你好助手", "小爱同学", "天猫精灵",
    "小管家", "小度",
)
_WAKE_WORDS = tuple(sorted(set(s.lower() for s in _WAKE_WORDS), key=lambda x: -len(x)))

# 口语填充词 / 礼貌词前缀（按长度降序，正则自动加 (您|你)? 等可选分组，避免误拆）
_FILLER_PREFIX_PATTERNS: Tuple[str, ...] = (
    r"麻烦您(一下)?",
    r"麻烦你(一下)?",
    r"麻烦(一下)?",
    r"请问一下", r"请问",
    r"帮个忙", r"帮忙",
    r"帮我(一下)?",
    r"帮你(一下)?",  # 极少，兼容 ASR 识别错"帮我"
    r"能不能(请)?", r"可不可以(请)?",
    r"你能不能(请)?", r"你可不可以(请)?",
    r"那个", r"嗯那个", r"呃那个",
    r"嗯", r"呃", r"额", r"哎", r"唉", r"哦",
)

# 中文/英文标点全集（用于"有效字符数 < 2"早退判定）
_PUNCT_CHARS = set("。，！？、；：\"'（）《》【」「『』—…·,.!?;:()[]{}/\\<>@#$%^&*~`-+_=|")


def _strip_whitespace_and_punct(text: str) -> str:
    if not text:
        return ""
    s = str(text)
    # step A: 零宽字符/不可见清理（逐个 Unicode 码点）
    s = "".join(
        ch for ch in s
        if ord(ch) not in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x00AD, 0x034F)
    )
    # step B: 首尾空白
    s = s.strip()
    # step C: 首尾标点（中英文都剥），最多剥 3 轮，避免 "，。，帮我开灯，。，" 这种两边套标点
    for _ in range(3):
        if not s:
            break
        if s[0] in _PUNCT_CHARS or s[0].isspace():
            s = s[1:]
            continue
        if s[-1] in _PUNCT_CHARS or s[-1].isspace():
            s = s[:-1]
            continue
        break
    return s.strip()


def _strip_prefix_repeatedly(text: str, pattern: str) -> str:
    """循环剥前缀（pattern 尾部自带 [，,\s。.!?]* 会拼接上），直到剥不动为止。"""
    if not text:
        return ""
    full = r"^\s*(" + pattern + r")\s*[，,\s。.!?;:]*"
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        cur = re.sub(full, "", cur, count=1, flags=re.IGNORECASE)
    return cur


def _strip_wake_word(text: str) -> str:
    if not text:
        return ""
    s = text
    # 逐个唤醒词试（已按长度降序），循环剥直到没变化
    for w in _WAKE_WORDS:
        w_pat = re.escape(w).replace(r"\ ", r"\s*")  # "嘿 siri" 中间的空格允许任意空白
        s = _strip_prefix_repeatedly(s, w_pat)
        if not s:
            return ""
    return _strip_whitespace_and_punct(s)


def _strip_filler_prefix(text: str) -> str:
    if not text:
        return ""
    s = text
    # 每条填充模式循环剥（直到没变化），整体再循环 3 轮，确保"嗯那个麻烦你帮我…"这种嵌套顺序都能剥掉
    for _ in range(3):
        before = s
        for pat in _FILLER_PREFIX_PATTERNS:
            s = _strip_prefix_repeatedly(s, pat)
            if not s:
                return ""
        s = _strip_whitespace_and_punct(s)
        if s == before:
            break
    return s


def _effective_char_count(text: str) -> int:
    """去掉所有空白 + 标点后的有效字符数。"""
    if not text:
        return 0
    cnt = 0
    for ch in text:
        if ch.isspace() or ch in _PUNCT_CHARS:
            continue
        cnt += 1
    return cnt


def is_effective_empty(text: Optional[str]) -> bool:
    """公共工具：判断有效字符数 < 2。供 voice_control / llm_nlu 早退使用。"""
    return _effective_char_count("" if text is None else str(text)) < 2


def preprocess_voice_text(text: Optional[str]) -> str:
    """
    统一的语音/文本预处理管道。所有 NLU / rule / 语音接口消费文本前都应调用。

    返回：处理后的字符串（如果处理后是空/纯标点，返回 ""）。
    """
    if not isinstance(text, str):
        return ""
    s = text

    # 1) 清洗标点 + 零宽字符
    s = _strip_whitespace_and_punct(s)
    if not s:
        return ""

    # 2) 剥唤醒词前缀（spec 允许忽略端上唤醒词，这里只剥文本残留）
    s = _strip_wake_word(s)
    if not s:
        return ""

    # 3) 剥口语填充词/礼貌词前缀
    s = _strip_filler_prefix(s)
    if not s:
        return ""

    # 4) asr_service 热词纠错（含标点清理 + HOTWORD_CORRECTIONS + 空格归一）
    try:
        from asr_service import post_process_text  # 延迟导入避免循环依赖
        s = post_process_text(s)
    except Exception:
        # 如果 asr_service 没加载成功（极端场景），至少保持剥标点/空格
        s = _strip_whitespace_and_punct(s)
    if not s:
        return ""

    # 5) nlu_parser 高频同音纠错（通用 50+ 条词典）
    s = asr_phonetic_correct(s)

    # 5.1) 用户显式指定的 11 条智能家居高频同音替换（独立字典）
    #      —— 在 parse_intent / SmartBrain 接收文本的第一时间执行
    s = correct_asr_homophones(s)

    # 5.2) ASR 纠错后常见"数字+属性词连体"归一：
    #        "26温度" → "26度" / "70亮度" → "70亮度"（保留，值提取正则通常接受）
    #        但 rule _match_value 只认 r'(\d+)\s*度' / r'(\d+)\s*%'，所以把 "XX温度" 强掰成 "XX度" 即可命中
    if s:
        s = re.sub(r'(\d+)\s*温度', r'\1度', s)
        s = re.sub(r'(\d+)\s*摄氏度', r'\1度', s)
        s = re.sub(r'(\d+)\s*华氏度', r'\1度', s)

    # 6) 收尾再清一遍，防止替换引入首尾标点/空格
    s = _strip_whitespace_and_punct(s)
    return s


# ================================================================
# 【ASR 同音错别字自动纠错】Phonetic Correction Pipeline
#   · 简化版：只保留【第 1 层 高频谐音精确替换】，
#     彻底移除 difflib 滑窗 + pypinyin 模糊匹配，避免破坏长句语义。
#   · 更"模糊"的拼音纠错能力下发给 LLM NLU System Prompt 处理（见 llm_nlu.py）。
#   · 在 parse_intent / parse_multi_intent / SmartBrain / llm_nlu
#     等所有消费用户文本的入口前调用，保证所有链路先经过高频精确替换。
# ================================================================

# —— 第 1 层：高频同音替换表（按【长→短】排序，保证"大开"先于"大"这种更准确的匹配）——
# 关键原则：只做"错词→正确词"的精确替换，绝不做单字或可能破坏长句结构的替换。
_PHONETIC_REPLACE_RULES: List[Tuple[str, str]] = [
    # 3~4 字组合（最长优先，避免短词先行把 "客厅的等" 拆成 "客厅的灯" 前被 "等→灯" 吞掉）
    ("我是空调", "卧室空调"), ("客听空调", "客厅空调"),
    ("客厅的等", "客厅的灯"), ("卧室的等", "卧室的灯"), ("所有的等", "所有的灯"),
    ("把等", "把灯"), ("等打开", "灯打开"), ("等关掉", "灯关掉"),
    # 2 字动作（同音错词→标准动作）
    ("开等", "开灯"), ("关等", "关灯"),
    ("关毕", "关闭"), ("官闭", "关闭"), ("观毕", "关闭"), ("管必", "关闭"),
    ("观闭", "关闭"), ("官比", "关闭"),
    ("大开", "打开"), ("达开", "打开"), ("搭开", "打开"), ("开打", "打开"),
    ("开起", "开启"),
    # 2 字设备（同音错词→标准设备名）
    ("关调", "空调"), ("空条", "空调"), ("空跳", "空调"), ("空凋", "空调"),
    ("窗连", "窗帘"), ("创联", "窗帘"), ("传帘", "窗帘"), ("床帘", "窗帘"),
    ("电世", "电视"), ("电试", "电视"), ("电是", "电视"),
    # 2 字房间（同音错词→标准房间名）—— 必须覆盖 parse_intent 中房间提取逻辑
    ("卧试", "卧室"), ("卧世", "卧室"), ("卧是", "卧室"), ("我室", "卧室"),
    ("卧事", "卧室"), ("握室", "卧室"),
    ("克厅", "客厅"), ("客听", "客厅"), ("克听", "客厅"), ("课厅", "客厅"),
    ("课听", "客厅"), ("刻厅", "客厅"), ("可听", "客厅"),
    ("书放", "书房"), ("书坊", "书房"),
    ("餐厨", "餐厅"), ("参厅", "餐厅"), ("餐听", "餐厅"),
    ("养台", "阳台"), ("扬台", "阳台"),
    ("出防", "厨房"), ("橱房", "厨房"),
    # 2 字属性（ASR 也会识别错属性名）
    ("分速", "风速"), ("风束", "风速"), ("风诉", "风速"),
    ("问度", "温度"), ("温渡", "温度"), ("问渡", "温度"),
    ("粮度", "亮度"), ("明渡", "亮度"),
    ("因量", "音量"), ("声量", "音量"),
    ("平道", "频道"), ("贫道", "频道"),
]
# 去重 + 去"正确词->正确词"的恒等占位，同时按长度从长到短排序保证长词优先替换
_unique_seen: set = set()
_PHONETIC_REPLACE_RULES_UNIQUE: List[Tuple[str, str]] = []
for k, v in _PHONETIC_REPLACE_RULES:
    if k == v:
        continue
    if k in _unique_seen:
        continue
    _unique_seen.add(k)
    _PHONETIC_REPLACE_RULES_UNIQUE.append((k, v))
_PHONETIC_REPLACE_RULES_UNIQUE.sort(key=lambda kv: -len(kv[0]))
del _unique_seen, _PHONETIC_REPLACE_RULES


# ================================================================
# 【ASR 同音字修正（独立字典，与 _PHONETIC_REPLACE_RULES_UNIQUE 并行）】
#   用法：correct_asr_homophones(text) —— 文本进入 NLU 解析前的第一道替换，
#   相比 asr_phonetic_correct 的 50+ 条通用谐音表，这里收窄为用户显式指定的
#   智能家居高频错字（如"个厅→客厅 / 红立→风力 / 双帘→窗帘 ……），
#   并在命中时打印一行 [NLU-Homophone] 日志，便于事后对照 ASR 原文。
# ================================================================
HOMOPHONE_MAP: Dict[str, str] = {
    "个厅": "客厅",
    "哥厅": "客厅",
    "各厅": "客厅",
    "卧试": "卧室",
    "卧是": "卧室",
    "卧式": "卧室",
    "红立": "风力",
    "红力": "风力",
    "风立": "风力",
    "穿帘": "窗帘",
    "双帘": "窗帘",
}


def correct_asr_homophones(text: str) -> str:
    """独立的 ASR 同音词修正函数（按 spec 显式实现）。

    遍历 HOMOPHONE_MAP，将 text 中匹配到的错字替换为正确的智能家居关键词；
    若发生任何替换，打印一条 [NLU-Homophone] 修正记录到 stdout。
    """
    if not text:
        return text
    raw_text = text
    for wrong, right in HOMOPHONE_MAP.items():
        if wrong in text:
            text = text.replace(wrong, right)
    if text != raw_text:
        print(f"[NLU-Homophone] ASR 同音修正: '{raw_text}' → '{text}'")
    return text


# —— 房间 slug <-> 中文名 双向映射（供 detect_explicit_room / resolve_device_id 使用）——
ROOM_CN_TO_SLUG: Dict[str, str] = {
    "卧室": "bedroom", "客厅": "living_room", "厨房": "kitchen",
    "书房": "study", "餐厅": "dining_room", "卫生间": "bathroom",
    "浴室": "bathroom", "阳台": "balcony",
}
# "主卧/次卧" 都视为卧室，"起居室"=客厅
_ROOM_EXTRA_CN: Dict[str, str] = {
    "主卧": "卧室", "次卧": "卧室", "起居室": "客厅",
}


def detect_explicit_room(text: str, asr_corrected: bool = False) -> str:
    """
    【需求 4】从用户原始文本中检测【显式房间】。

    返回房间 slug（"bedroom"/"living_room"），没检测到返回 ""。

    关键语义（修正 Bug：asr_corrected=True 时反而检测不到"克厅/卧试"）：
    - 无论调用者是否声称文本已被 asr_phonetic_correct 处理过，我们都 **先强制跑一次校正**，
      保证"克厅/卧试/客听/卧是…"这类同音错词能被当成"明确房间"。
    - 然后同时在【校正后文本 + 原始文本】里扫描：
        ① _ROOM_EXTRA_CN 的别名（克厅/卧试/客听…）
        ② ROOM_CN_TO_SLUG 的标准房间名（卧室/客厅）
    - 只要命中任何一个，就【强制锁定】该房间，禁止后续触发消歧反问。
    """
    if not isinstance(text, str) or not text:
        return ""

    corrected = asr_phonetic_correct(text)
    for maybe_s in (corrected, text):
        # ① 先扫别名（覆盖"克厅/卧试/客听/卧是/起居室"等）
        for alias, room_cn in _ROOM_EXTRA_CN.items():
            if alias in maybe_s:
                slug = ROOM_CN_TO_SLUG.get(room_cn)
                if slug:
                    return slug
        # ② 再扫标准房间名（卧室/客厅/书房…）
        for room_cn, slug in ROOM_CN_TO_SLUG.items():
            if room_cn in maybe_s:
                return slug
    return ""


def asr_phonetic_correct(text: str) -> str:
    """
    ASR 语音识别文本 → 纠错后的文本（供后续 parse_multi_intent / route_intent / SmartBrain 使用）。

    【简化版】：只做高频同音精确替换（不再做 difflib 滑窗 / pypinyin 模糊匹配）。
    更"模糊"的同音纠错交给 LLM NLU System Prompt（见 llm_nlu.py）处理。

    日志：只有当文本实际改动时才打印（减少 stdout 噪声）。
    """
    if not text:
        return text
    original = str(text)
    s = original

    # ——— 仅第 1 层：高频谐音硬替换（subn 方便统计命中次数）———
    n_total = 0
    for bad, good in _PHONETIC_REPLACE_RULES_UNIQUE:
        new_s, n = re.subn(re.escape(bad), good, s)
        if n > 0:
            s = new_s
            n_total += n

    if s != original:
        print(f"[ASR-Correct] 原文本='{original}' → 校正后='{s}' (高频谐音替换 {n_total} 次)")
    return s


# 会话上下文存储 - 追踪每个会话最后操作的设备
_last_device_context: Dict[str, str] = {}  # session_id -> device_id

# 取消指令关键词
CANCEL_KEYWORDS = ["算了", "不要了", "先不开", "不用了", "算了吧", "算了", "不要", "不用", "取消"]


# ==================== 属性→设备类型 强绑定（避免"风速"→灯 这种误识别）====================
# 只要文本中出现这些属性词，并且没有显式设备名，就推断对应设备类型
ATTR_DEVICE_TYPE_BINDING: Dict[str, str] = {
    # --- 空调专属：出现任何一个 → 设备类型 ac ---
    "风速": "ac", "风力": "ac", "风量": "ac", "风档": "ac", "档位": "ac",
    "扫风": "ac", "摆风": "ac", "风向": "ac",
    "温度": "ac", "制冷": "ac", "制热": "ac", "除湿": "ac", "送风": "ac", "模式": "ac",
    # --- 灯专属 ---
    "亮度": "light", "明度": "light",
    # --- TV专属 ---
    "音量": "tv", "声音": "tv", "声量": "tv", "频道": "tv",
}

# 属性 → intent 中对应的状态字段名
ATTR_TO_STATE_FIELD: Dict[str, str] = {
    "温度": "temperature",
    "风速": "fan_speed", "风力": "fan_speed", "风量": "fan_speed", "风档": "fan_speed", "档位": "fan_speed",
    "亮度": "brightness", "明度": "brightness",
    "音量": "volume", "声音": "volume", "声量": "volume",
    "频道": "channel",
}

# 每个状态字段的【默认边界】（可被 handle_delta_control 里的 DEVICE_BOUNDS 覆盖）
DEFAULT_FIELD_BOUNDS: Dict[str, Tuple[int, int]] = {
    "temperature": (16, 30),
    "fan_speed": (1, 5),
    "brightness": (0, 100),
    "volume": (0, 100),
    "channel": (1, 999),
}

# ==================== 增量/相对控制词典 ====================
# 正向增量（调高/加大/亮一点 等）→ delta 为正
_DELTA_POSITIVE_KEYWORDS: List[str] = [
    "调高", "升高", "加大", "调大", "增强", "调亮", "开大",
    "大一点", "大一些", "高一点", "高一些", "亮一点", "亮一些",
    "大一档", "升一档", "往上调", "向上调", "提高",
]
# 负向增量（调低/减小/暗一点 等）→ delta 为负
_DELTA_NEGATIVE_KEYWORDS: List[str] = [
    "调低", "降低", "减小", "调小", "减弱", "调暗", "关小",
    "小一点", "小一些", "低一点", "低一些", "暗一点", "暗一些",
    "小一档", "降一档", "往下调", "向下调",
]
# 增量调节的默认步长（单位：1 档 / 1℃ / 1%）
DELTA_DEFAULT_STEP = 1


def _detect_delta_polarity(text: str) -> int:
    """检测文本中增量的正负方向。返回 +1（增）、-1（减）、0（没检测到）。"""
    if any(kw in text for kw in _DELTA_POSITIVE_KEYWORDS):
        return +1
    if any(kw in text for kw in _DELTA_NEGATIVE_KEYWORDS):
        return -1
    return 0


def _extract_number(text: str) -> Optional[int]:
    """从文本中提取第一个阿拉伯数字/中文数字（支持一~五/两），找不到返回 None。"""
    # 阿拉伯数字
    m = re.search(r'(\d+)', text)
    if m:
        return int(m.group(1))
    # 中文数字（只取1~5 范围内常用的）
    cn_num = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    for ch, val in cn_num.items():
        if ch in text:
            return val
    return None


def _detect_delta_attr_and_field(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    根据文本中的属性词，推断要做增量调节的属性（中文）和 状态字段名（英文）。
    返回 (attr_cn, state_field)，例如 ("温度", "temperature")。
    """
    # 优先按出现顺序匹配（匹配多个时取第一个出现的）
    best_pos: Optional[int] = None
    best_attr: Optional[str] = None
    for kw in ATTR_DEVICE_TYPE_BINDING:
        pos = text.find(kw)
        if pos >= 0 and (best_pos is None or pos < best_pos):
            best_pos = pos
            best_attr = kw
    if best_attr is None:
        return None, None
    return best_attr, ATTR_TO_STATE_FIELD.get(best_attr)


def set_last_device(session_id: str, device_id: str):
    """记录会话最后操作的设备"""
    _last_device_context[session_id] = device_id


def get_last_device(session_id: str) -> Optional[str]:
    """获取会话最后操作的设备"""
    return _last_device_context.get(session_id)


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
# 【★ Verb-Action Binding 强绑定】：把 关/开 动作词精确匹配，并把 off 优先级放在最前面避免混淆
ACTION_PATTERNS: List[Tuple[str, str]] = [
    # --- 强绑定关（最高优先级，避免与"开"字在子串里混淆）---
    (r"(关闭|关掉|关上|关一下|关|灭|熄灭|灭掉|停止|停)", "off"),
    # --- 强绑定开 ---
    (r"(打开|开启|启动|开一下|开|亮|点亮|通电)", "on"),
    # --- 增减 / 设置 ---
    (r"(调高|升高|加大|调大|增强|上调|往上调)", "increase"),
    (r"(调低|降低|减小|调小|减弱|下调|往下调)", "decrease"),
    (r"(调到|调至|设为|设置成|变成|设置为|调成)", "set"),
    # --- 窗帘 ---
    (r"(拉上|收上|合上|关窗帘|拉窗帘)", "close_curtain"),  # close_curtain 最终会走 curtain state=closed
    (r"(拉开|打开窗帘|掀窗帘|开窗帘)", "open_curtain"),
    # 语义触发：根据环境/状态推断动作
    (r"(?:很|太|特别|有点|好|超|巨)(?:黑|暗|黑了|暗了)", "on"),  # 很黑/太暗 → 开灯
    (r"(?:很|太|特别|有点|好|超|巨)(?:亮|晃眼|刺眼)", "off"),  # 太亮 → 关灯/调暗
    (r"(?:很|太|特别|有点|好|超|巨)(?:热|闷|潮热)", "cool"),   # 很热 → 开制冷
    (r"(?:很|太|特别|有点|好|超|巨)(?:冷|冻|凉)", "heat"),     # 很冷 → 开制热
    (r"(?:没|未)(?:开|启动)", "on"),                             # 没开 → 打开
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


def _infer_device_by_attrs_room(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    当文本里没有显式设备名（"卧室空调/客厅灯"）时，
    通过属性强绑定 + 房间名，推断目标设备。

    Returns:
        (device_id, device_display_name, bound_device_type)
        例如 "卧室风速调大一档" → ("bedroom_ac", "卧室空调", "ac")
             "客厅亮度调低"  → ("living_room_light", "客厅灯", "light")
             推断不出 → (None, None, None)
    """
    # 1) 先看属性是否能绑定设备类型
    bound_type_list: List[str] = []
    for kw, t in ATTR_DEVICE_TYPE_BINDING.items():
        if kw in text:
            bound_type_list.append(t)
    if not bound_type_list:
        return None, None, None
    bound_type = bound_type_list[0]  # 取第一个命中的属性绑定类型

    # 2) 再看有没有房间名
    mentioned_bedroom = bool(re.search(r'卧室|主卧|房间|屋里|里面', text))
    mentioned_livingroom = bool(re.search(r'客厅|大厅|起居室', text))

    if mentioned_bedroom and mentioned_livingroom:
        return None, None, bound_type  # 歧义：用户同时提了两个房间，交给上下文处理

    prefer_room = "卧室" if mentioned_bedroom else ("客厅" if mentioned_livingroom else None)

    # 3) 在 DEVICE_STATES 里找：type == bound_type 且 room == prefer_room（或 room 任意）
    candidates: List[Tuple[str, Dict[str, Any]]] = [
        (k, v) for k, v in DEVICE_STATES.items() if v.get("type") == bound_type
    ]
    if not candidates:
        return None, None, bound_type

    # 优先选 prefer_room 匹配的
    if prefer_room:
        for k, v in candidates:
            if v.get("room") == prefer_room:
                room = v.get("room", "")
                cn_type = {"ac": "空调", "light": "灯", "tv": "电视", "curtain": "窗帘"}.get(bound_type, "设备")
                return k, f"{room}{cn_type}", bound_type

    # 没指定房间，返回第一个（卧室优先，通常在 DEVICE_STATES 排前面）
    k, v = candidates[0]
    room = v.get("room", "")
    cn_type = {"ac": "空调", "light": "灯", "tv": "电视", "curtain": "窗帘"}.get(bound_type, "设备")
    return k, f"{room}{cn_type}", bound_type


def parse_delta(text: str) -> Optional[Dict[str, Any]]:
    """
    增量/相对控制解析器。
    从"卧室温度调高2度"、"风速调大一些"这类文本中抽取增量控制元信息。

    Returns:
        None → 不是增量控制
        {
            "control_mode": "delta",       # 固定
            "delta_value": int,            # ±N（例如 +2、-1）
            "attr_cn": str,                # 中文属性："温度"/"风速"/"亮度"...
            "attr_field": str,             # 状态字段："temperature"/"fan_speed"/"brightness"...
            "bound_type": Optional[str],   # 推断的设备类型："ac"/"light"/"tv"
            "is_qualitative": bool,        # True=模糊（大一点/小一点），False=显式数字（2度）
        }
    """
    polarity = _detect_delta_polarity(text)
    if polarity == 0:
        return None
    attr_cn, attr_field = _detect_delta_attr_and_field(text)
    if attr_field is None:
        # 没属性词，但如果是"大一档"/"小一档"这种通用相对档位，尝试从已有属性绑定（或 AC_FAN_MAP 类似）推
        fan_like = re.search(r'[大小高低升]{1,2}(一|二|两|三|四|五|1|2|3|4|5)?档', text)
        if fan_like:
            attr_cn = "风力"
            attr_field = "fan_speed"
        else:
            return None
    # 绑定类型
    bound_type = ATTR_DEVICE_TYPE_BINDING.get(attr_cn) if attr_cn else None

    # 数字提取：如"调高2度" → 2；没数字 → 默认步长 DELTA_DEFAULT_STEP
    num = _extract_number(text)
    if num is not None:
        delta = polarity * num
        is_qualitative = False
    else:
        delta = polarity * DELTA_DEFAULT_STEP
        is_qualitative = True

    # 安全：delta 不能太离谱（单次调整最多 ±10 档/度）
    delta = max(-10, min(10, delta))

    return {
        "control_mode": "delta",
        "delta_value": delta,
        "attr_cn": attr_cn,
        "attr_field": attr_field,
        "bound_type": bound_type,
        "is_qualitative": is_qualitative,
    }


def _match_action(text: str) -> str:
    """
    【★ Verb-First 动作绑定】从文本中匹配动作，返回动作字符串。

    关键策略：当多个 pattern 同时匹配时，取【在文本中位置最靠前】的匹配结果，
             而非按 pattern 顺序选择。彻底避免"关闭X打开Y"中，
             如果拆分不彻底时，后面的"打开"反而覆盖前面的"关闭"。

    例：_match_action("客厅灯打开卧室") → 虽匹配"打开"(on)，但在多设备切分时
         prefix_action="关闭"(off) 已生效，这里兜底时仍取最靠前的动作词。
    """
    best_pos: Optional[int] = None
    best_act: Optional[str] = None
    for pattern, act in ACTION_PATTERNS:
        for m in re.finditer(pattern, text):
            pos = m.start()
            # 同位置时，保持先出现的 pattern（长词优先：关闭 > 关）
            if best_pos is None or pos < best_pos:
                best_pos = pos
                best_act = act
    return best_act if best_act is not None else "unknown"


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
      → [("curtain.all", "tv.all"), ("tv.all", "on")]
    """
    # ★ 所有 rule 解析入口先统一走预处理管道，保证 90+ 条词典命中
    text = preprocess_voice_text(text)

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

    # ========== 【状态感知】语义触发：优先级最高，在上下文兜底之前 ==========
    # --------------------------------------------------------------------------
    # 触发类型 → 动态决策表（关键：不再一律 action=on，而是读 DEVICE_STATES 后分支）：
    #  too_hot   : AC off → 开机+制冷+默认26°；AC on  → 增量 delta 降温 -2℃（边界16~30）
    #  too_cold  : AC off → 开机+制热+默认26°；AC on  → 增量 delta 升温 +2℃
    #  too_noisy : 遍历目标房间 type=tv/speaker 且 power=on → action=off；无开启设备→专用提示
    #  too_dark  : light off→开灯；light on→保持（避免重复打开）
    #  too_bright: light on→关灯；light off→保持
    # --------------------------------------------------------------------------
    PERCEPTION_PATTERNS: List[Tuple[str, str]] = [
        # --- 温度感知 ---
        (r"(?:很|太|特别|有点|好|超|巨)(?:热|闷|潮热)", "too_hot"),
        (r"(?:很|太|特别|有点|好|超|巨)(?:冷|冻|凉)", "too_cold"),
        (r"(?:没|不)(?:制冷|制热)", "too_hot"),
        (r"(?:空调|冷气).*(?:没|不|未).*(?:工作|启动|反应|亮)", "too_hot"),
        (r"(?:空调|冷气).*(?:需要|想要|想|要)(?:开|启动|打开)", "too_hot"),
        # --- 噪音/媒体感知 ---
        (r"(?:很|太|特别|有点|好|超|巨)(?:吵|响|闹|嘈杂)", "too_noisy"),
        (r"(?:声音|音量|声响).*(?:太|很|过|特别|有点)?(?:大|高|吵|响)", "too_noisy"),
        # --- 光照感知 ---
        (r"(?:很|太|特别|有点|好|超|巨)(?:黑|暗|黑了|暗了)", "too_dark"),
        (r"(?:没|未)(?:开|亮)", "too_dark"),
        (r"(?:很|太|特别|有点|好|超|巨)(?:亮|晃眼|刺眼)", "too_bright"),
    ]

    mentioned_bedroom = bool(re.search(r'卧室|主卧|房间|屋里|里面', text))
    mentioned_livingroom = bool(re.search(r'客厅|大厅|外面|起居室', text))

    def _filter_by_room(candidates_list: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
        """根据文本中提到的房间名，缩小设备候选范围"""
        if mentioned_bedroom:
            room_devs = [(k, v) for k, v in candidates_list if v.get("room") == "卧室"]
            return room_devs or candidates_list
        if mentioned_livingroom:
            room_devs = [(k, v) for k, v in candidates_list if v.get("room") == "客厅"]
            return room_devs or candidates_list
        # 未提房间：默认优先客厅（公共区域感知更强），没有则返回全部
        living_room = [(k, v) for k, v in candidates_list if v.get("room") == "客厅"]
        return living_room or candidates_list

    type_names_cn = {"ac": "空调", "light": "灯", "tv": "电视", "curtain": "窗帘", "speaker": "音响"}

    for pattern, perception_type in PERCEPTION_PATTERNS:
        if not re.search(pattern, text):
            continue

        # ============================================================
        #  【Case A】温度感知：太热/太冷 → 目标类型 AC
        # ============================================================
        if perception_type in ("too_hot", "too_cold"):
            ac_candidates = [(k, v) for k, v in DEVICE_STATES.items() if v.get("type") == "ac"]
            if not ac_candidates:
                break
            filtered = _filter_by_room(ac_candidates)
            dev_id, dev_state = filtered[0]
            room = dev_state.get("room", "")
            display = room + "空调"
            power_on = dev_state.get("power") == "on"
            cur_temp = int(dev_state.get("temperature", 26))
            cur_mode = str(dev_state.get("mode", "cooling"))

            intent: Dict[str, Any] = {
                "device_id": dev_id,
                "device_name": display,
                "action": "on",
                "value": None,
                "value_type": None,
                "attrs": {},
                "control_mode": "absolute",
                "delta": None,
                "implicit_intent": True,
            }

            target_mode = "cooling" if perception_type == "too_hot" else "heating"
            polarity = -2 if perception_type == "too_hot" else +2  # 增量默认 2℃

            # ---- 状态分支 ----
            if not power_on:
                # 空调是关的：开机 + 切到目标模式 + 设一个舒适默认温度
                default_temp = 26
                intent["action"] = "on"
                intent["attrs"]["mode"] = target_mode
                intent["attrs"]["temperature"] = default_temp
                intent["value"] = default_temp
                intent["value_type"] = "temperature"
                print(f"[NLU] 语义触发(温度感知): '{text}' → {display} 关闭 → 开机+{('制冷' if target_mode=='cooling' else '制热')} {default_temp}°")
            else:
                # 空调已开：走 delta 增量模式（降温-2 / 升温+2），边界由 _execute_ac clamp
                intent["control_mode"] = "delta"
                intent["action"] = "set"  # delta 模式下 action 不关键，但设为 set 更语义化
                intent["attrs"]["mode"] = target_mode if cur_mode not in ("cooling", "heating", "dehumidify", "fan", "auto") else cur_mode
                intent["delta"] = {
                    "attr_field": "temperature",
                    "attr_cn": "温度",
                    "delta_value": polarity,
                    "bound_type": "ac",
                    "is_qualitative": True,
                }
                print(f"[NLU] 语义触发(温度感知): '{text}' → {display} 已开({cur_mode} {cur_temp}°) → 增量 {'降温' if polarity<0 else '升温'} {abs(polarity)}°")

            if session_id:
                set_last_device(session_id, dev_id)
            return [intent]

        # ============================================================
        #  【Case B】噪音感知：太吵 → 找目标房间内"正在开"的 TV/Speaker → 关
        # ============================================================
        if perception_type == "too_noisy":
            # 候选：所有媒体类设备（当前系统里只有 tv）
            media_candidates = [
                (k, v) for k, v in DEVICE_STATES.items()
                if v.get("type") in ("tv", "speaker")
            ]
            if not media_candidates:
                # 系统里根本没有电视/音响 → 提示无设备
                return [{
                    "device_id": "virtual_no_media",
                    "device_name": "播放设备",
                    "action": "noop",
                    "value": None,
                    "value_type": None,
                    "attrs": {},
                    "implicit_intent": True,
                    "no_running_media": True,
                    "no_running_room": (
                        "卧室" if mentioned_bedroom else
                        "客厅" if mentioned_livingroom else ""
                    ),
                }]
            filtered_media = _filter_by_room(media_candidates)
            # 关键：只挑"正在播放(power=on)"的设备
            running = [(k, v) for k, v in filtered_media if v.get("power") == "on"]
            if not running:
                # 目标房间的媒体设备都没开 → 核查后告知用户
                target_room = (
                    "卧室" if mentioned_bedroom else
                    "客厅" if mentioned_livingroom else
                    (filtered_media[0][1].get("room", "") if filtered_media else "")
                )
                return [{
                    "device_id": "virtual_no_media",
                    "device_name": f"{target_room}播放设备" if target_room else "播放设备",
                    "action": "noop",
                    "value": None,
                    "value_type": None,
                    "attrs": {},
                    "implicit_intent": True,
                    "no_running_media": True,
                    "no_running_room": target_room,
                }]
            # 有关闭的目标：优先关 TV（如果 TV 和音响同时开，先关 TV 最常见）
            running_tvs = [(k, v) for k, v in running if v.get("type") == "tv"]
            chosen = running_tvs[0] if running_tvs else running[0]
            dev_id, dev_state = chosen
            room = dev_state.get("room", "")
            dtype = dev_state.get("type", "tv")
            display = room + type_names_cn.get(dtype, "播放设备")
            intent = {
                "device_id": dev_id,
                "device_name": display,
                "action": "off",
                "value": None,
                "value_type": None,
                "attrs": {},
                "control_mode": "absolute",
                "delta": None,
                "implicit_intent": True,
            }
            print(f"[NLU] 语义触发(噪音感知): '{text}' → 发现{display}正在播放 → 关闭")
            if session_id:
                set_last_device(session_id, dev_id)
            return [intent]

        # ============================================================
        #  【Case C】光照感知：太暗/太亮 → 灯（做状态感知，避免重复开关）
        # ============================================================
        if perception_type in ("too_dark", "too_bright"):
            light_candidates = [(k, v) for k, v in DEVICE_STATES.items() if v.get("type") == "light"]
            if not light_candidates:
                break
            filtered = _filter_by_room(light_candidates)
            dev_id, dev_state = filtered[0]
            room = dev_state.get("room", "")
            display = room + "灯"
            power_on = dev_state.get("power") == "on"
            want_on = perception_type == "too_dark"

            # 状态一致时，仍返回 intent 但在 message 上做去重（由 _execute_light 已处理 "已经开/关着了"）
            intent = {
                "device_id": dev_id,
                "device_name": display,
                "action": "on" if want_on else "off",
                "value": None,
                "value_type": None,
                "attrs": {},
                "implicit_intent": True,
            }
            print(f"[NLU] 语义触发(光照感知): '{text}' → {display} 当前{'开' if power_on else '关'} → {'开' if want_on else '关'}")
            if session_id:
                set_last_device(session_id, dev_id)
            return [intent]

    # ========== 语义触发 END ==========

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

        # ========== 【★ Bug修复：sub_text 动作词后截断】==========
        # 关键：对于 sub_text = "客厅灯打开卧室" 这种切分，
        #      "打开"是下一个设备的动作词，不属于当前设备，必须在 raw_name 之后的动作词处截断，
        #      否则 parse_intent("客厅灯打开") 会匹配到"打开"→action=on，覆盖 prefix_action=off
        raw_start_in_sub = sub_text.find(raw_name)
        if raw_start_in_sub >= 0:
            after_raw = raw_start_in_sub + len(raw_name)
            tail = sub_text[after_raw:]
            # 在 tail 中找第一个动作词的起始位置（相对于 sub_text 的偏移）
            earliest_verb: Optional[int] = None
            for pat, _ in ACTION_PATTERNS:
                for m in re.finditer(pat, tail):
                    abs_pos = after_raw + m.start()
                    if earliest_verb is None or abs_pos < earliest_verb:
                        earliest_verb = abs_pos
            if earliest_verb is not None:
                # ★ 截断：只保留设备名 + 设备名与动作词之间的内容（通常是空），
                #   后续动作词留给下一个设备用 prev_action 传播
                sub_text = sub_text[:earliest_verb].strip()

        if not sub_text.strip():
            sub_text = raw_name

        result = parse_intent(sub_text)
        if result["success"] and result["intent"]:
            # 如果是泛化引用，强制替换 device_id
            if dev_id.endswith(".all"):
                result["intent"]["device_id"] = dev_id
                result["intent"]["device_name"] = raw_name
            intents.append(result["intent"])
            if result["intent"].get("action") and result["intent"]["action"] != "unknown":
                prev_action = result["intent"]["action"]
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


# ======================================================================
# 【★ 设备状态查询解析器 parse_device_query】
#   - 单设备开关状态  (single_power)
#   - 列表/聚合查询    (list_on)
#   - 属性数值查询    (attribute)
#   - 带条件的查询控制 (conditional: "看看X没开就打开")
# ======================================================================

# 房间词 → 简写 key（与 DEVICE_MAP/DEVICE_STATES 保持一致）
_ROOM_KEYWORDS: List[Tuple[str, str]] = [
    ("主卧", "卧室"), ("次卧", "卧室"), ("客厅", "客厅"), ("起居室", "客厅"),
    ("餐厅", "餐厅"), ("厨房", "厨房"), ("书房", "书房"), ("卫生间", "卫生间"),
    ("浴室", "卫生间"), ("阳台", "阳台"), ("家里", ""), ("全屋", ""), ("整个屋子", ""),
]

# 条件控制短语：(正则, expected_power, then_action)
_CONDITIONAL_PATTERNS: List[Tuple[str, str, str]] = [
    # "没开就打开/开开/开启" → expected_power=off, then_action=on
    (r'(?:没开|是关着|关着|关闭|关掉|是关|没开着).*(?:就(?:|帮我|给我|马上|立刻|赶紧)?.?(?:打开|开开|开启|开了|开起来|点亮|开))', "off", "on"),
    # "关了就关/关闭" → expected_power=on, then_action=off
    (r'(?:开着|在开|开了|亮着|没关|开起来).*(?:就(?:|帮我|给我|马上|立刻|赶紧)?.?(?:关闭|关掉|关了|关上|灭掉|熄灭|关))', "on", "off"),
    # "没关就关"
    (r'(?:没关|是开|在运行).*(?:就(?:|帮我|给我)?.?(?:关闭|关掉|关了|关上))', "on", "off"),
    # 反向条件："开着就/如果开着就关上"
    (r'(?:如果|要是|假若).{0,4}(?:开着|开了|亮着).{0,4}(?:就|那就).{0,4}(?:关|关闭|灭掉)', "on", "off"),
    # 反向条件："关着就/如果关着就打开"
    (r'(?:如果|要是|假若).{0,4}(?:关着|关了|灭了|没开).{0,4}(?:就|那就).{0,4}(?:开|打开|点亮)', "off", "on"),
]


def _extract_room(text: str) -> str:
    """返回最靠前命中的房间中文名（"卧室"/"客厅"/""等）"""
    for raw, room_cn in _ROOM_KEYWORDS:
        if raw in text:
            return room_cn
    return ""


def _extract_device_type_for_list(text: str) -> str:
    """聚合查询时判断用户要查哪类设备：light / ac / tv / curtain / all"""
    if "灯" in text or "亮" in text:
        return "light"
    if "空调" in text:
        return "ac"
    if "电视" in text or "电视" in text:
        return "tv"
    if "窗帘" in text:
        return "curtain"
    return "all"


def _find_device_by_room_and_type(text: str, room_cn: str, dev_type: str) -> Optional[Tuple[str, str]]:
    """
    依据【房间 + 设备类型】从 DEVICE_MAP 中寻找最合适的设备。
    返回 (device_id, device_name_cn)，找不到返回 None。
    """
    candidates: List[Tuple[int, str, str]] = []
    for raw_name, dev_id in DEVICE_MAP:
        # dev_id 形如 bedroom_ac / living_room_ac / bedroom_light.all 等
        parts = dev_id.replace(".all", "").split("_")
        dev_room = parts[0]  # bedroom / living_room / ...
        # 将 dev_room 转为中文名与 _extract_room 返回对齐
        dev_room_cn = {"bedroom": "卧室", "living_room": "客厅", "kitchen": "厨房",
                       "bathroom": "卫生间", "study": "书房", "dining": "餐厅",
                       "balcony": "阳台"}.get(dev_room, "")
        # 类型判断
        if dev_type == "ac" and "ac" not in dev_id:
            continue
        if dev_type == "light" and "light" not in dev_id:
            continue
        if dev_type == "tv" and "tv" not in dev_id:
            continue
        if dev_type == "curtain" and "curtain" not in dev_id:
            continue
        # 原始名称匹配：若文本里有 raw_name（如 "客厅空调"），直接优先
        if raw_name in text:
            candidates.append((0, dev_id, raw_name))
            continue
        # 房间匹配：用户说的 room_cn 为空或与 dev_room_cn 相等
        if not room_cn or dev_room_cn == room_cn:
            candidates.append((1, dev_id, raw_name))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, dev_id, raw_name = candidates[0]
    return dev_id.replace(".all", ""), raw_name


def _detect_query_kind(text: str) -> str:
    """仅判断查询类型：single_power / list_on / attribute"""
    # 聚合查询（有哪些灯亮着 / 哪些设备开着）
    for p in [
        r'有哪些(?:灯|设备|家电|电器|空调)',
        r'(?:哪些|什么|啥)(?:灯|设备|家电|空调)(?:.*(?:亮|开|关)|$)',
        r'(?:灯|设备|家电).*有哪些',
        r'家里.*(?:灯|设备).*(?:亮着|开着)',
    ]:
        if re.search(p, text):
            return "list_on"
    # 属性数值查询（多少度/几度/几档/风速多少/什么模式）
    for p in [
        r'多少(?:度|档|级)',
        r'(?:几|几多)(?:度|档|级)',
        r'什么模式', r'什么风向', r'什么风速',
        r'风速多少', r'温度多少', r'亮度多少', r'音量多少',
        r'(?:现在|目前|这会儿).*(?:温度|风速|模式|档位|亮度|音量).*(?:是|多少|几)',
    ]:
        if re.search(p, text):
            return "attribute"
    # 默认 single_power（单设备开关状态问句）
    return "single_power"


def _infer_attr_cn_and_field(text: str) -> Tuple[Optional[str], Optional[str]]:
    """从属性查询句中推断 (属性中文名, 状态字段名)"""
    if ("多少度" in text) or ("几度" in text) or ("温度" in text):
        return "温度", "temperature"
    if ("几档" in text) or ("风速" in text) or ("风力" in text) or ("风量" in text):
        return "风速", "fan_speed"
    if "什么模式" in text or "模式" in text:
        return "模式", "mode"
    if "风向" in text or "扫风" in text or "摆风" in text:
        return "扫风", "swing_mode"
    if "亮度" in text or "明度" in text:
        return "亮度", "brightness"
    if "音量" in text or "声音" in text or "声量" in text:
        return "音量", "volume"
    if "频道" in text:
        return "频道", "channel"
    return None, None


def _infer_dev_type_from_attr(field: Optional[str], text: str) -> Optional[str]:
    """依据属性名推断设备类型（供属性查询找设备时使用）"""
    if field == "temperature" or field == "fan_speed" or field == "mode" or field == "swing_mode":
        return "ac"
    if field == "brightness":
        return "light"
    if field == "volume" or field == "channel":
        return "tv"
    # 兜底：文本中出现的设备名
    if "空调" in text:
        return "ac"
    if "灯" in text:
        return "light"
    if "电视" in text:
        return "tv"
    return None


def _extract_conditional(text: str, device_id: str) -> Optional[Dict[str, Any]]:
    """尝试从句中提取条件控制结构"""
    for pattern, expected, then_action in _CONDITIONAL_PATTERNS:
        if re.search(pattern, text):
            return {
                "expected_power": expected,
                "then_action": then_action,
                "then_device_id": device_id,
            }
    return None


def parse_device_query(text: str, session_id: str = "") -> Dict[str, Any]:
    """
    纯设备状态查询意图解析（仅在 router 判定 device_query 时调用）。

    Returns 结构：
        query_kind: "single_power" | "list_on" | "attribute"
        target_device_id / target_device_name: 针对 single_power / attribute
        target_device_type: 针对 list_on（"light"/"ac"/"tv"/"curtain"/"all"）
        target_room: 房间中文名（"" 表示全屋）
        target_attribute / target_attribute_cn: 针对 attribute
        conditional: 条件控制（可选）
    """
    base: Dict[str, Any] = {
        "query_kind": "single_power",
        "target_device_id": None,
        "target_device_name": None,
        "target_device_type": None,
        "target_room": _extract_room(text),
        "target_attribute": None,
        "target_attribute_cn": None,
        "conditional": None,
    }

    # 1) 判定 query_kind
    base["query_kind"] = _detect_query_kind(text)

    # 2) 列表/聚合查询
    if base["query_kind"] == "list_on":
        base["target_device_type"] = _extract_device_type_for_list(text)
        return base

    # 3) 属性数值查询
    if base["query_kind"] == "attribute":
        attr_cn, attr_field = _infer_attr_cn_and_field(text)
        base["target_attribute_cn"] = attr_cn
        base["target_attribute"] = attr_field
        dev_type = _infer_dev_type_from_attr(attr_field, text)
        if dev_type:
            found = _find_device_by_room_and_type(text, base["target_room"], dev_type)
            if found:
                base["target_device_id"], base["target_device_name"] = found
        # 兜底：无设备名但有会话上下文
        if not base["target_device_id"] and session_id:
            ctx = _last_device_context.get(session_id, "")
            if ctx:
                base["target_device_id"] = ctx
                for raw_name, dev_id in DEVICE_MAP:
                    if dev_id.replace(".all", "") == ctx:
                        base["target_device_name"] = raw_name
                        break
        return base

    # 4) single_power（单设备开关状态）
    #    先尝试 DEVICE_MAP 文本里的完整设备名
    found = None
    for raw_name, dev_id in DEVICE_MAP:
        if raw_name in text:
            found = (dev_id.replace(".all", ""), raw_name)
            break
    if not found:
        # 若文本里只有"空调"这种通用设备名 → 用 房间+类型 推断
        if any(kw in text for kw in ["空调"]):
            found = _find_device_by_room_and_type(text, base["target_room"], "ac")
        elif any(kw in text for kw in ["灯", "灯光"]):
            found = _find_device_by_room_and_type(text, base["target_room"], "light")
        elif any(kw in text for kw in ["电视"]):
            found = _find_device_by_room_and_type(text, base["target_room"], "tv")
        elif any(kw in text for kw in ["窗帘"]):
            found = _find_device_by_room_and_type(text, base["target_room"], "curtain")
    if not found and session_id:
        ctx = _last_device_context.get(session_id, "")
        if ctx:
            for raw_name, dev_id in DEVICE_MAP:
                if dev_id.replace(".all", "") == ctx:
                    found = (ctx, raw_name)
                    break
    if found:
        base["target_device_id"], base["target_device_name"] = found

    # 5) 条件控制提取（只针对 single_power，因为聚合查询不会接"没开就打开"）
    if base["target_device_id"]:
        base["conditional"] = _extract_conditional(text, base["target_device_id"])

    return base


# ======================================================================
# 【★ 复合指令预拆分 split_compound_command】
#   解决："关闭客厅灯打开卧室灯" 这种把动宾结构直接串起来的句子
#   策略：Verb-First，先找动作词，按动作词的位置切分；再配合显式连接词
# ======================================================================

# 动作词正则（长词优先，避免单字"开/关"误匹配"空调/关心"等）；每条 pattern 作为子指令的起点
_COMPOUND_ACTION_VERBS: List[str] = [
    r"(?:关闭|关掉|关上|关一下|灭掉|熄灭)",
    r"(?:打开|开启|启动|开一下|点亮|开开)",
    r"(?:调高|升高|加大|调大|增强|上调|往上调)",
    r"(?:调低|降低|减小|调小|减弱|下调|往下调)",
    r"(?:调到|调至|设为|设置成|设置为|调成|变成)",
    r"(?:拉上|收上|合上窗帘|关窗帘|拉窗帘)",
    r"(?:拉开|打开窗帘|开窗帘|掀窗帘)",
    r"(?:切换到|换到|切到|切换为)",
    # 语义触发（隐式意图）：虽然是单意图，但为了不被其他动作词拆错，放在这里
    r"(?:太热了|温度太高|好热|有点热|闷热)",
    r"(?:太冷了|冻死了|有点冷|好冷)",
    r"(?:太吵了|太响了|声音太大)",
    r"(?:太暗了|太黑了|看不清|光线不足)",
    # 单字兜底：开/关/亮/灭（只在 字符级动作词边界匹配）
]

# 语义连接词（分隔复合指令）：清洗用
_COMPOUND_CHUNK_CLEAN = re.compile(
    r"^[，,。；;、\s]*(?:然后|接着|顺便|同时|还有|另外|一并|再|加上|完了再|完再|去|把|接着再|然后再|之后|随后|马上|立刻)?[，,。；;、\s]*"
    r"|[，,。；;、\s]*(?:然后|接着|顺便|同时|还有|另外|一并|再|加上|完了再|完再|去|把|接着再|然后再|之后|随后|马上|立刻)?[，,。；;、\s]*$"
)

# 显式连接词切分：把文本按"然后/同时/并且/并且/再/逗号/句号"切粗片
_COMPOUND_EXPLICIT_SPLIT_RE = re.compile(
    r"(?:然后|接着|顺便|同时|还有|另外|一并|再|加上|完了再|完再|接着再|然后再|之后|随后|并且|且|外加|再加上)"
    r"|[，,。；;、]+"
)


def _find_action_boundaries(text: str) -> List[int]:
    """返回文本中所有动作词的起始字符位置（从小到大）"""
    positions: List[int] = []
    for pat in _COMPOUND_ACTION_VERBS:
        for m in re.finditer(pat, text):
            positions.append(m.start())
    # 去重 + 排序
    positions = sorted(set(positions))
    return positions


def _extract_leading_action_phrase(text: str) -> Optional[str]:
    """如果文本 0 位置就是动作词，返回动作词的完整匹配字符串（"打开"/"关闭客厅灯"中的"关闭"）；否则返回 None"""
    if not text:
        return None
    best_end = 0
    for pat in _COMPOUND_ACTION_VERBS:
        m = re.match(pat, text)
        if m and m.end() > best_end:
            best_end = m.end()
    if best_end <= 0:
        return None
    return text[:best_end]


def split_compound_command(text: str) -> List[str]:
    """
    把复合指令拆成独立子指令列表。两种切分策略叠加：
      1) 显式切分：按 "然后 / 同时 / 并且 / 逗号 / 句号" 等连接词/标点切粗片。
      2) 隐式切分（Verb-boundary）：对每个粗片再按动作词的位置切细片。
      +) Propagation：顿号列举纯设备名时（"打开A、B、C"），把前一个 chunk 的动作前缀传播到后续纯设备名 chunk。
    例：
      "关闭客厅灯打开卧室灯"                  → ["关闭客厅灯", "打开卧室灯"]
      "关闭客厅灯，然后打开卧室灯，再拉上窗帘" → ["关闭客厅灯", "打开卧室灯", "拉上窗帘"]
      "把空调调到26度并打开客厅灯"            → ["把空调调到26度", "打开客厅灯"]
      "打开卧室灯、客厅灯、厨房灯"            → ["打开卧室灯", "打开客厅灯", "打开厨房灯"]（★ Verb传播）
    """
    if not text or not text.strip():
        return []

    # Step 1: 先做显式粗切（按连接词/标点，这些通常是用户手写的分断）
    raw_chunks: List[str] = []
    parts = _COMPOUND_EXPLICIT_SPLIT_RE.split(text)
    for p in parts:
        clean = p.strip()
        if clean:
            raw_chunks.append(clean)
    if not raw_chunks:
        raw_chunks = [text.strip()]

    # Step 1b: ★ Verb 传播 —— 把前面的动作前缀"传染"给后面的纯设备名 stub chunks
    #   典型："打开卧室灯、客厅灯、厨房灯" → raw_chunks = ["打开卧室灯", "客厅灯", "厨房灯"]
    #         后两个不含动作词 → prepend "打开"
    propagated: List[str] = []
    inherit_action_phrase: Optional[str] = None
    for raw in raw_chunks:
        has_action = len(_find_action_boundaries(raw)) > 0
        if has_action:
            propagated.append(raw)
            inherit_action_phrase = _extract_leading_action_phrase(raw)
        else:
            # 纯设备名：如果前面有可继承的动作词，就 prepend
            if inherit_action_phrase:
                propagated.append(f"{inherit_action_phrase}{raw}")
            else:
                propagated.append(raw)
    raw_chunks = propagated

    # Step 2: 对每个粗片按动作词位置再切（Verb-boundary 细切）
    final_chunks: List[str] = []
    for raw in raw_chunks:
        boundaries = _find_action_boundaries(raw)
        if len(boundaries) <= 1:
            # 0 个或 1 个动作词：保留整段
            final_chunks.append(raw)
            continue
        # 多个动作词：按边界切成多段
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(raw)
            chunk = raw[start:end]
            chunk = _COMPOUND_CHUNK_CLEAN.sub("", chunk).strip()
            if chunk:
                final_chunks.append(chunk)

    # Step 3: 再清洗一次残余连接词、空白、前后标点 + 【过滤纯连接词 chunks】
    # （兜底：当显式切分因为正则问题把连接词本身当片段时，这里二次排除）
    _STOP_PURE_CONNECTIVES = {
        "然后", "接着", "顺便", "同时", "还有", "另外", "一并", "再", "加上",
        "去", "把", "之后", "随后", "马上", "立刻", "且", "外加", "再加上",
        "完了再", "完再", "接着再", "然后再",
    }
    out: List[str] = []
    for c in final_chunks:
        c = _COMPOUND_CHUNK_CLEAN.sub("", c).strip()
        # 去掉 "把XX" 前缀的 "把" 通常是辅助词，不影响解析，但保持完整片段
        if not c:
            continue
        if c in _STOP_PURE_CONNECTIVES:
            continue
        out.append(c)
    if not out:
        # 兜底：没有任何动作词时（比如纯场景名"回家模式"），返回整句
        t = text.strip()
        return [t] if t else []
    return out


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

    【新增增量控制】如果检测到相对词（调高/调小/大一点 等）：
        intent["control_mode"] = "delta"
        intent["delta"] = {"delta_value": ±N, "attr_field": "temperature"/"fan_speed"...}
    否则默认 intent["control_mode"] = "absolute"

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
                "value_type": str,
                "attrs": dict,
                "control_mode": "absolute" | "delta",
                "delta": dict | None,
            } | None
        }
    """
    # ★ 所有 rule 解析入口先统一走预处理管道，保证 90+ 条词典命中
    text = preprocess_voice_text(text)
    if is_effective_empty(text):
        # 纯规则模式下的"空指令早退"
        return {
            "success": False,
            "msg": "未检测到有效语音指令，请重新表达",
            "method": "rule",
            "confidence": 0.0,
            "need_clarification": True,
            "intent": None,
        }

    # 1) 尝试【显式设备名】匹配：卧室空调/客厅灯 等
    target_device_id, target_device_raw = _match_device(text)
    inferred_by_attr = False

    # 2) 如果显式设备没命中，尝试【属性强绑定 + 房间名】推断（比如"卧室风速调大一档"里没有"空调"二字）
    if target_device_id is None:
        inferred_id, inferred_name, bound_type = _infer_device_by_attrs_room(text)
        if inferred_id is not None:
            target_device_id = inferred_id
            target_device_raw = inferred_name
            inferred_by_attr = True

    action = _match_action(text)
    value, value_type = _match_value(text)

    # 3) 增量控制解析（必须在动作匹配之后，因为 parse_delta 能告诉我们更细的控制意图）
    delta_info = parse_delta(text)

    # —— ★ 防兜底：只有房间没有设备时，直接澄清，禁止后续 DEVICE_RESOLUTION 泛 ID 兜底到空调 ——
    #    触发条件：
    #      (1) 还没识别到任何设备 (target_device_id 为空)
    #      (2) 文本里明确提到了某个房间 (explicit_room_slug 非空)
    #      (3) 动作词也被识别了 (action != unknown) —— 即用户明确说了"打开/关闭"，只是没说控啥
    #    → 返回 need_clarification=True，候选设备列当前房间全部（灯/空调/窗帘/电视）
    explicit_room_slug_for_check = detect_explicit_room(text, asr_corrected=True)
    if not target_device_id and explicit_room_slug_for_check and action != "unknown":
        room_cn_for_msg = {v: k for k, v in ROOM_CN_TO_SLUG.items()}.get(explicit_room_slug_for_check, "")
        if not room_cn_for_msg:
            room_cn_for_msg = "该房间"
        # 该房间下的候选设备（从 DEVICE_STATES 取，避免硬编码）
        candidate_names: List[str] = []
        _cn = {"ac": "空调", "light": "灯", "curtain": "窗帘", "tv": "电视"}
        for sid, s in DEVICE_STATES.items():
            if sid.startswith(explicit_room_slug_for_check + "_"):
                t = s.get("type", "")
                if t in _cn:
                    candidate_names.append(_cn[t])
        if not candidate_names:
            # 兜底：最常见的 2 类（至少会问出用户期待）
            candidate_names = ["灯", "空调"]
        if len(candidate_names) == 1:
            cand_txt = candidate_names[0]
        else:
            cand_txt = "、".join(candidate_names[:-1]) + f"还是{candidate_names[-1]}"
        return {
            "success": False,
            "msg": f"未识别到具体设备，请问您是要控制{room_cn_for_msg}的{cand_txt}？",
            "method": "rule",
            "confidence": 0.1,
            "need_clarification": True,
            "intent": {
                "raw_text": text,
                "device_id": None,
                "device_name": None,
                "action": action,
                "value": value,
                "value_type": value_type,
                "attrs": {},
                "control_mode": "absolute",
                "delta": None,
                "_inferred_by_attr": False,
                "_explicit_room_slug": explicit_room_slug_for_check,
                "_room_only_missing_device": True,
                "_clarify_candidates": candidate_names,
            },
        }

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

    # —— ★ 关键：检测用户是否明确提到某个房间（含同音错词，如"克厅"→客厅，"卧试"→卧室）——
    explicit_room_slug = detect_explicit_room(text, asr_corrected=True)
    # 【需求 4】：文本含明确房间名时，强制锁定该房间，禁止消歧反问
    if explicit_room_slug and target_device_id in DEVICE_RESOLUTION:
        fixed_device_id = resolve_device_id(target_device_id, explicit_room=explicit_room_slug)
        # 把 device_name 也带上房间前缀（保持与"卧室空调/客厅灯"一致）
        room_cn = {v: k for k, v in ROOM_CN_TO_SLUG.items()}.get(explicit_room_slug, "")
        type_cn = {"air_conditioner": "空调", "light": "灯", "curtain": "窗帘", "tv": "电视"}.get(target_device_id, target_device_raw or "")
        if room_cn and type_cn:
            target_device_raw = room_cn + type_cn
        target_device_id = fixed_device_id

    # 通用设备（无房间限定）需要消歧；若用户已明确房间则跳过
    need_clarification = target_device_id in DEVICE_RESOLUTION and not explicit_room_slug

    # 提取空调专属属性
    ac_attrs: Dict[str, Any] = {}
    if target_device_id.endswith("_ac") or target_device_id == "air_conditioner":
        ac_attrs = _extract_ac_attrs(text)

    # 构建 intent 基础结构（control_mode 字段）
    control_mode = delta_info["control_mode"] if delta_info else "absolute"
    base_intent = {
        "raw_text": text,
        "device_id": target_device_id,
        "device_name": target_device_raw,
        "action": action,
        "value": value,
        "value_type": value_type,
        "attrs": ac_attrs,
        "control_mode": control_mode,
        "delta": delta_info,  # None 或 {"control_mode":"delta","delta_value":±N,...}
        "_inferred_by_attr": inferred_by_attr,
        "_explicit_room_slug": explicit_room_slug,  # 供上层 device_controller 拼接回复文案用
    }

    # 如果是 delta 控制，且 action 还没定 → 用 increase/decrease
    if control_mode == "delta" and action == "unknown":
        base_intent["action"] = "increase" if (delta_info and delta_info["delta_value"] > 0) else "decrease"

    # 未识别到动作，但有额外属性 或 是 delta 控制
    if action == "unknown":
        if ac_attrs or delta_info is not None:
            # 有空调属性 或 增量控制 → 也算有效意图
            if delta_info is not None and control_mode == "delta":
                base_intent["action"] = "increase" if delta_info["delta_value"] > 0 else "decrease"
            else:
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

    confidence = 0.6 if need_clarification else (0.8 if inferred_by_attr else 0.9)

    return {
        "success": True,
        "msg": "意图解析成功",
        "method": "rule",
        "confidence": confidence,
        "need_clarification": need_clarification,
        "intent": base_intent,
    }


def resolve_device_id(device_id: str, prefer_room: str = "bedroom",
                      explicit_room: Optional[str] = None) -> str:
    """
    将通用设备ID解析为具体设备ID。

    Args:
        device_id:    通用（如 air_conditioner / curtain）或具体设备 ID
        prefer_room:  未指定 explicit_room 时的默认房间 slug（默认卧室）
        explicit_room:【强约束】用户文本中明确出现的房间 slug（由 detect_explicit_room 给出）。
                      当传入 explicit_room 时，优先在候选列表中匹配该房间。
    """
    if device_id in DEVICE_RESOLUTION:
        candidates = DEVICE_RESOLUTION[device_id]
        # 1) 优先强制锁定显式房间
        if explicit_room:
            for c in candidates:
                if explicit_room in c:
                    return c
        # 2) 再按 prefer_room 兜底
        for c in candidates:
            if prefer_room in c:
                return c
        return candidates[0]
    return device_id
