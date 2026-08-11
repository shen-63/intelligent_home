import time
from typing import Dict, Any, Optional

# 会话超时时间（秒）
SESSION_TTL = 120

# 房间关键词映射
ROOM_KEYWORDS: Dict[str, str] = {
    "卧室": "bedroom",
    "主卧": "bedroom",
    "次卧": "bedroom",
    "客厅": "living_room",
    "起居室": "living_room",
}

# 通用设备ID -> 房间前缀拼接规则
DEVICE_PREFIX_MAP = {
    "air_conditioner": "{room}_ac",
    "light": "{room}_light",
    "curtain": "{room}_curtain",
    "tv": "{room}_tv",
}


class PendingIntent:
    """待确认的意图上下文"""

    def __init__(self, intent: Dict[str, Any], text: str):
        self.intent = intent
        self.text = text
        self.timestamp = time.time()
        self.device_id = intent.get("device_id", "")
        self.device_name = intent.get("device_name", "")
        self.action = intent.get("action", "")
        self.value = intent.get("value")
        self.value_type = intent.get("value_type")

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > SESSION_TTL

    def resolve_with_room(self, room: str) -> Optional[Dict[str, Any]]:
        """用房间信息解析出具体设备 ID，返回完整 intent"""
        template = DEVICE_PREFIX_MAP.get(self.device_id)
        if template is None:
            # 已经是具体设备（如 bedroom_curtain），直接返回
            if "_" in self.device_id and self.device_id not in DEVICE_PREFIX_MAP:
                return dict(self.intent)
            return None

        resolved_id = template.format(room=room)
        resolved = dict(self.intent)
        resolved["device_id"] = resolved_id
        resolved["device_name"] = f"{'卧室' if room == 'bedroom' else '客厅'}{self.device_name}"
        resolved["from_context"] = True
        return resolved


class DialogManager:
    """多轮对话上下文管理器"""

    def __init__(self):
        self._sessions: Dict[str, PendingIntent] = {}

    def save_pending(self, session_id: str, intent: Dict[str, Any], text: str = ""):
        """保存待确认的意图"""
        self._cleanup()
        self._sessions[session_id] = PendingIntent(intent, text)
        print(f"[Dialog] 保存上下文: session={session_id}, device={intent.get('device_id')}, action={intent.get('action')}")

    def get_pending(self, session_id: str) -> Optional[PendingIntent]:
        """获取待确认的意图，过期返回 None"""
        self._cleanup()
        pending = self._sessions.get(session_id)
        if pending is None:
            return None
        if pending.is_expired():
            del self._sessions[session_id]
            print(f"[Dialog] 上下文已过期: session={session_id}")
            return None
        return pending

    def clear(self, session_id: str):
        """清除会话上下文"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            print(f"[Dialog] 清除上下文: session={session_id}")

    def _cleanup(self):
        """清理过期会话"""
        now = time.time()
        expired = [sid for sid, p in self._sessions.items() if now - p.timestamp > SESSION_TTL]
        for sid in expired:
            del self._sessions[sid]


def extract_room(text: str) -> Optional[str]:
    """从文本中提取房间关键词，返回英文房间名"""
    text = text.strip()
    for keyword, room in ROOM_KEYWORDS.items():
        if keyword in text:
            return room
    return None


def is_clarification_response(text: str) -> bool:
    """
    判断文本是否是对消歧提问的回复。
    判断依据：短文本 + 包含房间关键词 + 无明显动作动词
    """
    text = text.strip()
    if len(text) > 12:
        return False
    # 必须包含房间关键词
    has_room = any(kw in text for kw in ROOM_KEYWORDS)
    if not has_room:
        return False
    # 排除包含明显动作的指令（如"打开卧室空调"是完整新指令，不是消歧回复）
    action_words = ["打开", "关闭", "开启", "关掉", "调到", "调高", "调低", "拉开", "关上", "打开"]
    action_count = sum(1 for w in action_words if w in text)
    # 如果有动作词且文本较长，可能是新指令而非消歧回复
    if action_count > 0 and len(text) > 6:
        return False
    return True
