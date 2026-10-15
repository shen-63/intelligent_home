from __future__ import annotations
import time
from typing import Dict, Any, Optional, List, Tuple

# ============== 会话超时时间（秒）= 用户要求 30s ==============
SESSION_TTL = 30

# ============== 房间关键词映射（兼容中→英 slug）==============
# 注意：需覆盖所有显式空间词（书房/厨房/儿童房…），即便未注册设备也需识别
ROOM_KEYWORDS: Dict[str, str] = {
    # 卧室
    "卧室": "bedroom",
    "主卧": "bedroom",
    "次卧": "bedroom",
    "房间": "bedroom",
    "屋里": "bedroom",
    "里面": "bedroom",
    # 客厅
    "客厅": "living_room",
    "大厅": "living_room",
    "起居室": "living_room",
    "厅里": "living_room",
    # 其他常用空间
    "书房": "study",
    "书斋": "study",
    "厨房": "kitchen",
    "后厨": "kitchen",
    "餐厅": "dining",
    "饭厅": "dining",
    "卫生间": "bathroom",
    "厕所": "bathroom",
    "洗手间": "bathroom",
    "浴室": "bathroom",
    "卫浴": "bathroom",
    "阳台": "balcony",
    "露台": "balcony",
    "儿童房": "kids",
    "小孩房": "kids",
    "婴儿房": "kids",
    "阁楼": "attic",
    "地下室": "basement",
    "玄关": "entrance",
    "走廊": "hallway",
}

# 房间 slug -> 中文显示名（反向）
ROOM_CN_MAP: Dict[str, str] = {
    "bedroom": "卧室",
    "living_room": "客厅",
    "study": "书房",
    "kitchen": "厨房",
    "dining": "餐厅",
    "bathroom": "卫生间",
    "balcony": "阳台",
    "kids": "儿童房",
    "attic": "阁楼",
    "basement": "地下室",
    "entrance": "玄关",
    "hallway": "走廊",
}

# 设备类型关键字（用于判断短文本"空调/客厅空调"是否含设备词）
TYPE_KEYWORDS_CN = ("空调", "灯", "灯光", "窗帘", "窗纱", "帘子", "电视", "电视机", "彩电")

# 通用设备ID -> 房间前缀拼接规则（与 DEVICE_STATES 对齐）
DEVICE_PREFIX_MAP = {
    "air_conditioner": "{room}_ac",
    "ac": "{room}_ac",
    "light": "{room}_light",
    "curtain": "{room}_curtain",
    "tv": "{room}_tv",
}

# 泛化设备 id → type（与 DEVICE_STATES 的 type 字段一致）
GENERIC_DEV_TO_TYPE = {
    "air_conditioner": "ac",
    "ac": "ac",
    "light": "light",
    "curtain": "curtain",
    "tv": "tv",
}


class PendingIntent:
    """
    待确认的悬挂意图（Slot Filling 上下文）。
    结构同时兼容：
      - 内部完整 intent（`self.intent`，直接用于 execute_command）
      - 对外诊断 JSON（`to_slot_dict()`，按需求 1 的 schema 输出）
    """

    def __init__(
        self,
        intent: Dict[str, Any],
        original_text: str = "",
        target_device_type: Optional[str] = None,
        target_val: Any = None,
        action_label: str = "",
        candidate_rooms_cn: Optional[List[str]] = None,
    ):
        self.intent = dict(intent)               # 完整原始 intent（合并槽位后直接执行）
        self.original_text = original_text        # 用户第一次输入的原句（Debug）
        self.timestamp = time.time()              # 创建时间戳，用于 30s 超时

        # ---- 需求 1 的结构化字段 ----
        self.action_label = action_label or intent.get("action", "") or intent.get("intent_action", "")
        self.target_device_type = target_device_type or GENERIC_DEV_TO_TYPE.get(intent.get("device_id", ""), "")
        self.target_val = target_val              # 已做过边界处理后的最终目标值（如 50→30）
        self.status = "waiting_for_room"          # 固定状态：等待补充房间槽位

        # ---- 槽位补全辅助信息 ----
        # 候选房间中文名（用于动态反问文案+验证用户回复的有效性）
        self.candidate_rooms_cn: List[str] = candidate_rooms_cn or []

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > SESSION_TTL

    def to_slot_dict(self) -> Dict[str, Any]:
        """按需求 1 定义的 schema 输出诊断 JSON（供日志/前端/调试查看）"""
        return {
            "action": self.action_label,
            "target_device_type": self.target_device_type,
            "target_val": self.target_val,
            "status": self.status,
        }

    # ------------------------------------------------------------------
    # 槽位补全：房间 + 泛化 device_id → 完整具体设备 id
    # ------------------------------------------------------------------
    def resolve_with_room_slug(self, room_slug: str) -> Optional[Dict[str, Any]]:
        """用英文房间 slug（bedroom/living_room）补全 device_id，返回可执行的完整 intent"""
        generic_dev_id = self.intent.get("device_id", "")
        template = DEVICE_PREFIX_MAP.get(generic_dev_id)

        if template is None:
            # 已经是具体设备（如 bedroom_curtain），直接返回副本
            if "_" in generic_dev_id and generic_dev_id not in GENERIC_DEV_TO_TYPE:
                resolved = dict(self.intent)
                resolved["from_context"] = True
                return resolved
            return None

        resolved_id = template.format(room=room_slug)
        resolved = dict(self.intent)
        resolved["device_id"] = resolved_id
        room_cn = ROOM_CN_MAP.get(room_slug, room_slug)
        dev_name_raw = self.intent.get("device_name", "") or self._default_type_cn()
        # 拼出中文显示名：卧室 + 空调 = 卧室空调
        if not dev_name_raw.startswith(room_cn):
            resolved["device_name"] = f"{room_cn}{dev_name_raw}"
        resolved["from_context"] = True
        return resolved

    def resolve_with_room(self, room_anything: str) -> Optional[Dict[str, Any]]:
        """
        兼容入口：接受 中文房间名 或 英文 slug，自动转换后补全槽位。
        voice_control / handle_context_clarification 中使用此方法。
        """
        if not room_anything:
            return None
        # 如果传的是中文（如 "卧室"），先转成 slug
        if room_anything in ROOM_CN_MAP.values():
            # 中文 → 反查 slug
            for slug, cn in ROOM_CN_MAP.items():
                if cn == room_anything:
                    return self.resolve_with_room_slug(slug)
            # ROOM_KEYWORDS 中可能还有别名（如 "主卧" → bedroom）
            slug = ROOM_KEYWORDS.get(room_anything)
            if slug:
                return self.resolve_with_room_slug(slug)
            return None
        # 如果传的是 slug 或已在 ROOM_CN_MAP 中
        if room_anything in ROOM_CN_MAP:
            return self.resolve_with_room_slug(room_anything)
        # 再尝试 ROOM_KEYWORDS 完整匹配
        slug = ROOM_KEYWORDS.get(room_anything)
        if slug:
            return self.resolve_with_room_slug(slug)
        # 最后原样尝试（防止奇怪的输入）
        return self.resolve_with_room_slug(room_anything)

    def _default_type_cn(self) -> str:
        t = self.target_device_type or ""
        return {"ac": "空调", "light": "灯", "curtain": "窗帘", "tv": "电视"}.get(t, "设备")


class DialogManager:
    """多轮对话上下文管理器（session_id → PendingIntent，带 30s 自动清理）"""

    def __init__(self):
        self._sessions: Dict[str, PendingIntent] = {}

    # ------------------------------------------------------------------
    # 基础 CRUD
    # ------------------------------------------------------------------
    def save_pending(
        self,
        session_id: str,
        intent: Dict[str, Any],
        original_text: str = "",
        target_device_type: Optional[str] = None,
        target_val: Any = None,
        action_label: str = "",
        candidate_rooms_cn: Optional[List[str]] = None,
    ) -> PendingIntent:
        """保存待确认的悬挂意图（覆盖同一 session 旧的）"""
        self._cleanup()
        pending = PendingIntent(
            intent=intent,
            original_text=original_text,
            target_device_type=target_device_type,
            target_val=target_val,
            action_label=action_label,
            candidate_rooms_cn=candidate_rooms_cn,
        )
        self._sessions[session_id] = pending
        slot_json = pending.to_slot_dict()
        print(f"[Dialog] 保存悬挂意图 session={session_id}: {slot_json} | 候选房间={candidate_rooms_cn}")
        return pending

    def get_pending(self, session_id: str) -> Optional[PendingIntent]:
        """获取未过期的悬挂意图；过期/不存在 → None"""
        self._cleanup()
        pending = self._sessions.get(session_id)
        if pending is None:
            return None
        if pending.is_expired():
            del self._sessions[session_id]
            print(f"[Dialog] 悬挂意图已超时({SESSION_TTL}s), 已清除 session={session_id}")
            return None
        return pending

    def clear(self, session_id: str, reason: str = ""):
        """主动清除会话上下文"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            print(f"[Dialog] 清除悬挂意图 session={session_id} 原因={reason or 'user_cancel'}")

    def _cleanup(self):
        """惰性清理：每次访问时扫描所有 session，剔除已过期的"""
        now = time.time()
        expired = [sid for sid, p in self._sessions.items() if now - p.timestamp > SESSION_TTL]
        for sid in expired:
            del self._sessions[sid]
            print(f"[Dialog][cleanup] 悬挂意图超时清理 session={sid}")


# ======================================================================
# 辅助函数：从文本中提取房间信息 / 判断消歧回复有效性
# ======================================================================

def extract_room_slug(text: str) -> Optional[str]:
    """从文本中提取房间关键词 → 英文 slug（如 "卧室空调" → "bedroom"）"""
    text = text.strip()
    # 优先匹配长词（如"次卧"优先于"卧室"）
    for kw in sorted(ROOM_KEYWORDS.keys(), key=len, reverse=True):
        if kw in text:
            return ROOM_KEYWORDS[kw]
    return None


# ---- 兼容别名：main.py / voice_control 中调用的是 extract_room() ----
def extract_room(text: str) -> Optional[str]:
    """extract_room_slug 的别名，兼容 voice_control / handle_context_clarification 中的调用"""
    return extract_room_slug(text)


def extract_room_cn(text: str) -> Optional[str]:
    """从文本中提取房间关键词 → 中文（如 "卧室空调" → "卧室"）"""
    text = text.strip()
    for kw in sorted(ROOM_KEYWORDS.keys(), key=len, reverse=True):
        if kw in text:
            return kw
    return None


def contains_any_device_word(text: str) -> bool:
    return any(w in text for w in TYPE_KEYWORDS_CN)


def is_clarification_response(
    text: str,
    pending: Optional[PendingIntent] = None,
) -> bool:
    """
    判断文本是否是对"消歧反问"的有效房间回复。
    触发条件（同时满足）：
      1) 文本 ≤ 16 个字符（短回复特征，如"卧室的"、"客厅空调"）
      2) 包含候选房间词（若有 pending 则用其候选集，否则遍历 ROOM_KEYWORDS 全量）
      3) 不含"全新动作动词 + 显式宾语"的完整新指令（如"打开卧室灯"是完整新指令，不视为消歧）
    """
    text = text.strip()
    if not text:
        return False
    if len(text) > 16:
        return False

    # ① 必须有房间词（优先匹配 pending 的候选房间）
    if pending and pending.candidate_rooms_cn:
        has_room = any(r in text for r in pending.candidate_rooms_cn)
    else:
        has_room = any(kw in text for kw in ROOM_KEYWORDS)
    if not has_room:
        return False

    # ② 排除明显的完整新指令：包含动作动词 + 设备词同时出现，且文本长
    #    如 "打开客厅空调" → action word + 设备词 = 新指令，不应当作消歧回复。
    action_words = ("打开", "关闭", "开启", "关掉", "调到", "调高", "调低",
                    "拉开", "关上", "调大", "调小", "切换", "设为", "设定",
                    "播放", "暂停", "唱", "放首", "大声点", "小声点", "停止")
    action_hit = sum(1 for w in action_words if w in text)
    dev_hit = 1 if contains_any_device_word(text) else 0
    if action_hit > 0 and dev_hit > 0 and len(text) >= 6:
        return False
    # 只有动作词但无房间设备名（如用户改口"算了大声点"、"放首歌"）→ 也不是消歧回复
    if action_hit > 0 and dev_hit == 0:
        return False
    return True


def build_clarification_question(
    pending: PendingIntent,
) -> str:
    """
    根据悬挂意图 + 候选房间动态生成反问文案。
    例：候选=[卧室,客厅], type=ac → "请问您是要调节卧室空调，还是客厅空调？"
    """
    rooms = pending.candidate_rooms_cn or ["卧室", "客厅"]
    type_cn = {"ac": "空调", "light": "灯", "curtain": "窗帘", "tv": "电视"}.get(
        pending.target_device_type, pending.intent.get("device_name") or "设备"
    )
    # 根据动作选择一个友好动词
    action_verb = "调节"
    act = pending.action_label or ""
    if act in ("on", "turn_on", "open", "open_curtain"):
        action_verb = "打开"
    elif act in ("off", "turn_off", "close", "close_curtain"):
        action_verb = "关闭"
    elif act == "set_temperature" or "temperature" in (pending.intent.get("value_type") or ""):
        action_verb = "调节"
    if pending.target_device_type == "curtain":
        action_verb = "操作"
    # 动态拼接备选项：只有 2 个用 "是X还是Y"，>2 个用顿号
    if len(rooms) == 2:
        return f"请问您是要{action_verb}{rooms[0]}{type_cn}，还是{rooms[1]}{type_cn}？"
    options = "、".join(f"{r}{type_cn}" for r in rooms)
    return f"请问您要{action_verb}哪个房间的{type_cn}？可选：{options}"


# ======================================================================
# 动态候选房间查询：基于 DEVICE_STATES 实际注册设备，不依赖硬编码
# ======================================================================

def get_candidate_rooms_for_type(device_type: str) -> List[str]:
    """
    根据设备类型（ac/light/curtain/tv）从 DEVICE_STATES 中找出所有已注册的房间中文名。
    用于 build_clarification_question 和歧义判断。
    返回示例：device_type="ac" → ["卧室", "客厅"]
    """
    try:
        from device_controller import DEVICE_STATES
    except Exception:
        return ["卧室", "客厅"]

    rooms: List[str] = []
    seen = set()
    for dev_id, state in DEVICE_STATES.items():
        if state.get("type") == device_type:
            room_cn = state.get("room", "")
            if room_cn and room_cn not in seen:
                seen.add(room_cn)
                rooms.append(room_cn)
    return rooms or ["卧室", "客厅"]


def count_devices_of_type(device_type: str) -> int:
    """统计 DEVICE_STATES 中某类型设备的数量（用于判断是否需要歧义反问）"""
    try:
        from device_controller import DEVICE_STATES
    except Exception:
        return 1
    return sum(1 for s in DEVICE_STATES.values() if s.get("type") == device_type)


def is_generic_device_id(device_id: str) -> bool:
    """
    判断 device_id 是否是【泛化类型ID】（未指定房间的设备名）。
    如："ac"、"air_conditioner"、"light"、"curtain"、"tv" → True
    如："bedroom_ac"、"客厅灯" → False（已经具体到房间）
    """
    GENERIC_IDS = {"ac", "air_conditioner", "light", "curtain", "tv"}
    # 精确匹配泛化集合
    if device_id in GENERIC_IDS:
        return True
    # 还有 .all 结尾的是批量操作，不算歧义（它们会自己展开）
    if device_id.endswith(".all"):
        return False
    # 如果是 GENERIC_DEV_TO_TYPE 中的 key 且不含下划线/中文（排除具体设备）
    if device_id in GENERIC_DEV_TO_TYPE and "_" not in device_id:
        # 防呆：中文房间名（如"卧室空调"）是具体设备，不应被视作泛化
        has_cn_room = any(kw in device_id for kw in ROOM_KEYWORDS)
        return not has_cn_room
    return False


# ======================================================================
# 全局单例：DialogManager（main.py / smart_brain.py 共享，确保 session 互通）
# ======================================================================
dialog_manager = DialogManager()
