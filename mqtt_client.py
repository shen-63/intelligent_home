"""
MQTT 客户端封装：将 Python 后端的内存设备控制与真实 MQTT Broker 打通。

设计目标：
1) Python 后端执行设备控制后，通过 publish_command() 将 JSON 指令发布到真实设备的 set Topic
2) 订阅所有设备的 state Topic，接收真实设备状态反馈，回调同步回 DEVICE_STATES
3) Broker 不可用时自动降级为内存 Mock，不影响本地 Mock 场景的开发调试

Topic 约定（与 Dify 知识库 JSON 格式保持一致）：
    发布（控制）： home/{room}/{type}/set
    订阅（反馈）： home/{room}/{type}/state

    {room}:   bedroom | living_room
    {type}:   light | ac | curtain | tv

Payload 格式（JSON）：
    {
        "entity_id": "climate.bedroom_ac",
        "action": "turn_on" | "turn_off" | "set",
        "temperature": 26,
        "mode": "cooling",
        "fan_speed": 1,
        "swing": "horizontal"
    }
"""
from __future__ import annotations

import json
import threading
from typing import Any, Callable, Dict, Optional

try:
    import paho.mqtt.client as mqtt
    _PAHO_AVAILABLE = True
except Exception:
    _PAHO_AVAILABLE = False

from config import (
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    MQTT_ENABLED,
    MQTT_PASSWORD,
    MQTT_TOPIC_PREFIX,
    MQTT_USERNAME,
)


# ---------- 类型简写 ----------
StateUpdateCallback = Callable[[str, Dict[str, Any]], None]


class MQTTClient:
    """单例 MQTT 客户端，内部封装 paho.mqtt + 自动降级"""

    _instance: Optional["MQTTClient"] = None
    _instance_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "MQTTClient":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = MQTTClient()
            return cls._instance

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._enabled: bool = MQTT_ENABLED and _PAHO_AVAILABLE
        self._connected: bool = False
        self._state_cb: Optional[StateUpdateCallback] = None
        self._client: Optional[mqtt.Client] = None
        self._prefix: str = MQTT_TOPIC_PREFIX.rstrip("/")
        self._lock: threading.Lock = threading.Lock()

    # ----------------------------- 对外 API -----------------------------

    def init(self, state_update_cb: Optional[StateUpdateCallback] = None) -> bool:
        """
        连接 Broker，并订阅所有设备的 state Topic。

        Args:
            state_update_cb: 收到真实设备状态反馈时触发的回调，签名为
                             callback(device_id: str, new_state_dict: dict)

        Returns:
            True 表示已成功连接到真实 Broker，False 表示降级为 Mock 模式
        """
        self._state_cb = state_update_cb

        if not self._enabled:
            print(f"[MQTT] 已禁用或 paho-mqtt 未安装，使用内存 Mock 模式 (enabled={MQTT_ENABLED}, paho={_PAHO_AVAILABLE})")
            return False

        try:
            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
            if MQTT_USERNAME:
                self._client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD or None)
            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message
            self._client.on_disconnect = self._on_disconnect
            self._client.connect_async(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=30)
            self._client.loop_start()
            print(f"[MQTT] 正在连接 {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT} ...")
            # 返回 True 说明已经启动了连接流程；是否真正连接上会在 _on_connect 里打印
            return True
        except Exception as e:
            print(f"[MQTT] 连接失败，降级为内存 Mock 模式: {e}")
            self._client = None
            return False

    def shutdown(self) -> None:
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
            self._connected = False

    def publish_command(
        self,
        device_id: str,
        room: str,
        dev_type: str,
        action: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        将控制指令发布到 home/{room}/{type}/set Topic。

        Args:
            device_id: 设备 ID，如 "climate.bedroom_ac"
            room: "bedroom" | "living_room"
            dev_type: "ac" | "light" | "curtain" | "tv"
            action: "turn_on" | "turn_off" | "set"
            extra:  {"temperature": 26, "fan_speed": 1, "mode": "cooling", ...}

        Returns:
            真实 Broker 发布成功返回 True；Mock 模式永远返回 True
        """
        topic = f"{self._prefix}/{room}/{dev_type}/set"
        payload: Dict[str, Any] = {
            "entity_id": device_id,
            "action": action,
        }
        if extra:
            payload.update(extra)

        # 无 Broker 时只打印日志，不做真实发布（本地 Mock 开发阶段）
        if not self._enabled or self._client is None:
            print(f"[MQTT Mock] → {topic}: {json.dumps(payload, ensure_ascii=False)}")
            return True

        if not self._connected:
            # 尚未完成连接，先打印日志，不抛异常（不阻塞主流程）
            print(f"[MQTT] Broker 未连接，跳过发布: {topic} {json.dumps(payload, ensure_ascii=False)}")
            return False

        try:
            msg = json.dumps(payload, ensure_ascii=False)
            with self._lock:
                self._client.publish(topic, payload=msg, qos=1)
            print(f"[MQTT] → {topic}: {msg}")
            return True
        except Exception as e:
            print(f"[MQTT] publish_command 失败: {e}")
            return False

    # ----------------------------- 内部回调 -----------------------------

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            print(f"[MQTT] 已连接到 {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
            # 订阅所有设备的 state Topic: home/{room}/{type}/state
            room_types = ("bedroom", "living_room")
            dev_types = ("light", "ac", "curtain", "tv")
            for room in room_types:
                for dtype in dev_types:
                    t = f"{self._prefix}/{room}/{dtype}/state"
                    client.subscribe(t, qos=1)
                    print(f"[MQTT] 订阅状态主题: {t}")
        else:
            print(f"[MQTT] 连接失败，错误码 rc={rc}")
            self._connected = False

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            print(f"[MQTT] 异常断开（rc={rc}），paho 将自动尝试重连")

    def _on_message(self, client, userdata, msg):
        """
        接收 home/{room}/{type}/state 的真实状态反馈，
        解析后通过回调同步到 DEVICE_STATES。

        预期 Payload 示例：
            {"power": "on"}
            {"power": "on", "temperature": 26, "mode": "cooling", "fan_speed": 1, "swing": "horizontal"}
            {"state": "open"}     ← 窗帘
            {"position": 50}       ← 窗帘半开
        """
        try:
            topic = msg.topic
            raw = msg.payload.decode("utf-8").strip()
            state = json.loads(raw) if raw else {}
            if not isinstance(state, dict):
                print(f"[MQTT] state payload 非 JSON 对象，忽略: {topic} -> {raw!r}")
                return

            # 从 topic home/{room}/{type}/state 中提取房间与设备类型
            parts = topic.strip("/").split("/")
            if len(parts) != 4 or parts[-1] != "state":
                print(f"[MQTT] 未知 Topic 格式，忽略: {topic}")
                return
            _, room, dtype, _ = parts

            # 用 room + type 反推 device_id（与 DEVICE_STATES 的 key 对齐）
            dtype_entity_map = {
                "ac": "climate",
                "light": "light",
                "curtain": "cover",
                "tv": "media_player",
            }
            entity_prefix = dtype_entity_map.get(dtype, dtype)
            device_id = f"{entity_prefix}.{room}_{dtype}"

            print(f"[MQTT] ← {topic}: {json.dumps(state, ensure_ascii=False)}")

            if callable(self._state_cb):
                try:
                    self._state_cb(device_id, state)
                except Exception as e:
                    print(f"[MQTT] state_update_cb 异常: {e}")
        except json.JSONDecodeError as e:
            print(f"[MQTT] state payload 非 JSON: {msg.topic} -> {e}")
        except Exception as e:
            print(f"[MQTT] _on_message 异常: {e}")


# ----------------------------- 便捷函数 -----------------------------

def get_mqtt_client() -> MQTTClient:
    return MQTTClient.get_instance()
