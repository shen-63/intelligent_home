import os
import re
import json
import copy
import urllib.parse
import uuid
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import imageio_ffmpeg
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from funasr import AutoModel

from nlu_parser import (parse_intent, parse_multi_intent, resolve_device_id, get_device_risk,
                        set_last_device, asr_phonetic_correct, detect_explicit_room,
                        ROOM_CN_TO_SLUG, preprocess_voice_text, is_effective_empty)
from dialog_context import is_generic_device_id
from device_controller import (
    execute_command,
    DEVICE_STATES,
    state_lock,
    get_mqtt_meta,
    apply_mqtt_state_update,
)
from llm_nlu import parse_intent_llm
from dialog_context import (
    dialog_manager,          # 共享单例（与 smart_brain 共用 sessions 字典）
    is_clarification_response,
    extract_room,
    extract_room_cn,
    build_clarification_question,
)
from tts_service import synthesize, synthesize_to_bytes, synthesize_to_response, list_voices, CHINESE_VOICES, clean_tts_text
from config import SILICONFLOW_API_KEY, RULE_CONFIDENCE_THRESHOLD
from mqtt_client import get_mqtt_client

# 设备名映射：支持中文/英文/HA格式的 entity_id 解析
DEVICE_NAME_MAP: Dict[str, str] = {
    # 卧室设备
    "卧室空调": "bedroom_ac",
    "卧室灯": "bedroom_light",
    "卧室窗帘": "bedroom_curtain",
    "卧室电视": "bedroom_tv",
    # 客厅设备
    "客厅空调": "living_room_ac",
    "客厅灯": "living_room_light",
    "客厅窗帘": "living_room_curtain",
    "客厅电视": "living_room_tv",
    # 常见别名
    "空调": "bedroom_ac",
    "灯": "bedroom_light",
    "窗帘": "bedroom_curtain",
    "电视": "bedroom_tv",
    # HA 标准格式 (domain.room_type)
    "climate.bedroom_ac": "bedroom_ac",
    "climate.living_room_ac": "living_room_ac",
    "light.bedroom_light": "bedroom_light",
    "light.living_room_light": "living_room_light",
    "cover.bedroom_curtain": "bedroom_curtain",
    "cover.living_room_curtain": "living_room_curtain",
    "media_player.bedroom_tv": "bedroom_tv",
    "media_player.living_room_tv": "living_room_tv",
    # HA 缩写格式 (domain.room) - Dify LLM 容易生成这种
    "light.bedroom": "bedroom_light",
    "light.living_room": "living_room_light",
    "climate.bedroom": "bedroom_ac",
    "climate.living_room": "living_room_ac",
    "cover.bedroom": "bedroom_curtain",
    "cover.living_room": "living_room_curtain",
    "media_player.bedroom": "bedroom_tv",
    "media_player.living_room": "living_room_tv",
    # 无域缩写格式
    "bedroom": "bedroom_ac",
    "living_room": "living_room_ac",
}

# HA 域到设备类型的映射
HA_DOMAIN_TYPE_MAP: Dict[str, str] = {
    "climate": "ac",
    "light": "light",
    "cover": "curtain",
    "media_player": "tv",
    "sensor": "sensor",
    "binary_sensor": "sensor",
    "switch": "switch",
    "fan": "fan",
}

# 房间名到英文 key 的映射
ROOM_NAME_MAP: Dict[str, str] = {
    "卧室": "bedroom",
    "客厅": "living_room",
    "书房": "study_room",
    "厨房": "kitchen",
    "阳台": "balcony",
}


def resolve_entity_id(entity_id: str) -> Optional[str]:
    """将任意格式的 entity_id 解析为 DEVICE_STATES 的 key"""
    if not entity_id:
        return None

    # 1. 直接匹配 DEVICE_STATES key
    if entity_id in DEVICE_STATES:
        return entity_id

    # 2. 映射表查找
    if entity_id in DEVICE_NAME_MAP:
        return DEVICE_NAME_MAP[entity_id]

    # 3. HA 域前缀剥离后匹配 (climate.living_room_ac -> living_room_ac)
    if "." in entity_id:
        parts = entity_id.split(".", 1)
        domain = parts[0]
        remainder = parts[1]

        # 3a. 完整格式剥离后匹配
        if remainder in DEVICE_STATES:
            return remainder
        if remainder in DEVICE_NAME_MAP:
            return DEVICE_NAME_MAP[remainder]

        # 3b. 缩写格式推断: 如 light.bedroom -> bedroom_light
        # 根据 domain 推断设备类型
        dev_type = HA_DOMAIN_TYPE_MAP.get(domain, "")
        if dev_type:
            # 尝试在 DEVICE_STATES 中查找匹配
            for key in DEVICE_STATES:
                if key.startswith(remainder + "_") and DEVICE_STATES[key]["type"] == dev_type:
                    return key

            # 如果只有一个该类型设备，直接匹配
            candidates = [k for k in DEVICE_STATES
                         if k.startswith(remainder + "_") and DEVICE_STATES[k]["type"] == dev_type]
            if len(candidates) == 1:
                return candidates[0]

        # 3c. 尝试所有类型匹配 (适用于非标准 domain)
        for key in DEVICE_STATES:
            if key.startswith(remainder + "_"):
                return key

    # 4. 尝试中文房间名推断: 如 "卧室" + "灯" -> bedroom_light
    for cn_room, en_room in ROOM_NAME_MAP.items():
        if cn_room in entity_id:
            # 尝试匹配所有以该房间开头的设备
            for dev_key in DEVICE_STATES:
                if dev_key.startswith(en_room + "_") and dev_key == entity_id:
                    return dev_key
            break

    return None

app = FastAPI(title="Smart Home Voice Assistant API")

# 添加 CORS 中间件，允许 Dify 跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Voice-Meta", "X-Device-States"],
)


@app.on_event("startup")
def _startup_mqtt_init():
    """服务启动时初始化 MQTT 客户端：连接 Broker + 订阅所有设备的状态反馈 Topic。"""
    def _mqtt_state_cb(device_id: str, new_state: Dict[str, Any]) -> None:
        """MQTT 状态反馈回调：paho-mqtt 在后台线程中调用，需要安全地把协程调度到事件循环"""
        try:
            import asyncio as _ai
            try:
                loop = _ai.get_running_loop()
                loop.create_task(apply_mqtt_state_update(device_id, new_state))
            except RuntimeError:
                # 事件循环尚未就绪（理论上 startup 之后才会收到消息），降级为同步模式
                loop = _ai.new_event_loop()
                loop.run_until_complete(apply_mqtt_state_update(device_id, new_state))
                loop.close()
        except Exception as e:
            print(f"[MQTT] state_cb 调度失败: {e}")

    mqtt = get_mqtt_client()
    ok = mqtt.init(state_update_cb=_mqtt_state_cb)
    if ok:
        print("[MQTT] 已启动 MQTT 客户端，等待 Broker 连接完成...")
    else:
        print("[MQTT] 使用内存 Mock 模式（未连接真实 Broker），publish 仅打印日志不发送。")


@app.on_event("shutdown")
def _shutdown_mqtt_cleanup():
    """服务停止时优雅关闭 MQTT 连接"""
    try:
        mqtt = get_mqtt_client()
        mqtt.shutdown()
        print("[MQTT] 已关闭 MQTT 客户端")
    except Exception as e:
        print(f"[MQTT] shutdown 异常: {e}")

# 多轮对话上下文管理器 → 现在使用 dialog_context.dialog_manager 共享单例（避免 main/smart_brain 各自一份 sessions）
# 见 dialog_context.py 末尾的：dialog_manager = DialogManager()

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".webm", ".pcm", ".amr", ".opus"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# 音频有效性校验阈值（与 asr_service.py 保持一致）
MIN_AUDIO_DURATION_SEC = 0.2   # 最短有效时长（秒）
MIN_FILE_SIZE = 44             # 最小有效文件大小（字节）

CJK_PUNCTUATION = "。，！？、；：""''（）《》"


def _probe_audio_duration(file_path: str) -> float:
    """
    快速探测音频文件时长（秒）。
    优先用 ffprobe，降级为 ffmpeg stderr 解析。
    返回 0.0 表示无法获取。
    """
    import re as _re

    # 优先尝试 ffprobe
    ffprobe = os.path.join(os.path.dirname(FFMPEG_EXE), "ffprobe.exe")
    if not os.path.exists(ffprobe):
        ffprobe = os.path.join(os.path.dirname(FFMPEG_EXE), "ffprobe")
    if not os.path.exists(ffprobe):
        ffprobe = "ffprobe"

    try:
        res = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True, text=True, timeout=15,
        )
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip())
    except Exception:
        pass

    # 降级：从 ffmpeg -i 的 stderr 输出解析
    try:
        res = subprocess.run(
            [FFMPEG_EXE, "-i", file_path],
            capture_output=True, text=True, timeout=15,
        )
        m = _re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", res.stderr)
        if m:
            return (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                    + int(m.group(3)) + int(m.group(4)) / 100.0)
    except Exception:
        pass

    return 0.0


print("正在初始化 ASR 模型...")
asr_model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    disable_update=True,
)
print("模型加载完成！")

llm_available = bool(SILICONFLOW_API_KEY)
print(f"LLM 状态: {'已就绪' if llm_available else '未配置 (规则模式)'}")

# Dify 已移除，项目全面使用 SmartBrain 本地大脑


def convert_to_16k_wav(input_path: str, output_path: str):
    cmd = [
        FFMPEG_EXE, "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg 转码失败: {res.stderr.strip()}")
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("转码后文件为空")


def clean_text(text: str) -> str:
    return text.strip().strip(CJK_PUNCTUATION + ",.!? ")


# extract_device_intent_from_reply removed — Dify dead code (2026-08 cleanup)

async def handle_context_clarification(text: str, session_id: str) -> Optional[str]:
    """
    检查对话上下文，如果是消歧回复则解析并执行（需求 2 槽位补全 / 需求 3 改口重置）。

    返回:
      - 执行结果文本（str）：当是有效消歧回复时
      - "__NEW_CMD__" 哨兵：当有 pending 但用户改口（非澄清回复），调用方应按新指令重新处理
      - None：无 pending，完全走正常流程
    """
    pending = dialog_manager.get_pending(session_id)
    if not pending:
        return None

    # 情况A：是消歧回复（房间名/设备名）→ 合并槽位 + 立即执行
    if is_clarification_response(text, pending):
        room_slug = extract_room(text)
        if not room_slug:
            # 有时用户只说"卧室" → extract_room_cn 可识别
            room_cn = extract_room_cn(text)
            if room_cn:
                from dialog_context import ROOM_KEYWORDS
                room_slug = ROOM_KEYWORDS.get(room_cn)
        if room_slug:
            resolved_intent = pending.resolve_with_room(room_slug)
            if resolved_intent:
                print(f"[Dialog] 消歧回复 '{text}' → room={room_slug}, device_id={resolved_intent['device_id']}")
                dialog_manager.clear(session_id, reason="clarification_resolved")
                # 记录最后操作设备（供后续上下文续控）
                if resolved_intent.get("device_id"):
                    set_last_device(session_id, resolved_intent["device_id"])
                execution = await execute_command(resolved_intent)
                msg = execution.get("message", "指令已执行")
                # —— ★ handle_context_clarification 返回消息强制非空 ——
                if not isinstance(msg, str) or not msg.strip():
                    msg = "操作已完成。"
                    print(f"[Reply-Safe] handle_context_clarification: message 为空，兜底 → {msg!r}")
                return msg

    # 情况B：有 pending 但用户【改口】或 【提取不到房间】 → 清 pending，告诉上层重走新指令
    dialog_manager.clear(session_id, reason="user_changed_mind")
    print(f"[Dialog] 用户改口（非消歧回复: '{text}'），已清空 pending_intent，按新指令重处理")
    return "__NEW_CMD__"


def hybrid_parse_intent(text: str) -> dict:
    """
    混合意图解析：规则优先，LLM 兜底消歧。
    - 规则解析 success=False / confidence < threshold / need_clarification=True 时触发 LLM
    - LLM 可用时，即使规则成功也让 LLM 处理消歧（LLM 能自动选择默认房间）
    - LLM 失败时回退到规则结果
    """
    rule_result = parse_intent(text)

    need_llm = False
    if not rule_result["success"]:
        need_llm = True
    elif rule_result.get("confidence", 0) < RULE_CONFIDENCE_THRESHOLD:
        need_llm = True
    elif rule_result.get("need_clarification", False):
        need_llm = True

    if need_llm and llm_available:
        print(f"[NLU] 规则解析: success={rule_result['success']}, conf={rule_result.get('confidence', 0)}, need_clar={rule_result.get('need_clarification', False)}, 触发 LLM")
        llm_call_ok = False
        try:
            llm_result = parse_intent_llm(text)
            if llm_result["success"]:
                if rule_result.get("need_clarification") and not llm_result.get("need_clarification"):
                    print(f"[NLU] LLM 成功消歧: {rule_result['intent']['device_id']} → {llm_result['intent']['device_id']}")
                return llm_result
            print(f"[NLU] LLM 也解析失败: {llm_result['msg']}, 降级到规则结果")
        except Exception as e:
            print(f"[NLU] LLM 调用异常: {e}, 降级到规则结果")

    # 规则需要消歧（LLM 不可用或 LLM 失败）：按设备风险分级处理
    if rule_result.get("need_clarification"):
        intent = rule_result["intent"] or {}
        device_id = intent.get("device_id") if isinstance(intent, dict) else None

        # ★ 防兜底：只有房间没有设备（_room_only_missing_device）时，保持澄清，不做卧室→卧室_ac 这种兜底
        if intent and intent.get("_room_only_missing_device"):
            print(f"[NLU] 只识别到房间无设备，保留澄清提示: {rule_result.get('msg')}")
            return rule_result

        if not device_id:
            # 没有 device_id 的消歧（如空设备、属性不足）也保持原澄清，拒绝任何默认值兜底
            return rule_result

        risk = get_device_risk(device_id)

        if risk == "high_risk":
            print(f"[NLU] 高风险设备 {device_id}，需要用户确认房间")
            rule_result["need_clarification"] = True
        else:
            intent["device_id"] = resolve_device_id(device_id)
            rule_result["intent"] = intent
            rule_result["need_clarification"] = False
            print(f"[NLU] 低风险设备 {device_id}，默认卧室 → {intent['device_id']}")

    return rule_result


async def batch_execute_intents(intents: list, session_id: str = "", text: str = "") -> str:
    """
    批量执行多个意图，合并回复文本。

    Args:
        intents: parse_multi_intent 返回的意图列表
        session_id: 会话ID，用于保存消歧上下文
        text: 原始用户文本（用于【需求 4】房间锁定，含"客厅/克厅/卧室/卧试"时禁止反问）

    Returns:
        合并后的回复文本
    """
    if not intents:
        return "抱歉，无法识别您的指令。"

    messages: list = []
    has_clarification = False
    clarification_msg = ""
    clarification_intent = None

    # 【需求 4】最高优先级：文本明确提到客厅/卧室（含同音错词）时，通用设备ID直接 resolve 到指定房间
    explicit_room = detect_explicit_room(text, asr_corrected=True) if isinstance(text, str) else ""
    if explicit_room:
        for intent in intents:
            did = intent.get("device_id", "") or ""
            if is_generic_device_id(did):
                fixed = resolve_device_id(did, explicit_room=explicit_room)
                if fixed != did:
                    intent["device_id"] = fixed
                    intent["need_clarification"] = False
                    room_cn = {v: k for k, v in ROOM_CN_TO_SLUG.items()}.get(explicit_room, "")
                    type_cn = {"air_conditioner": "空调", "ac": "空调", "light": "灯",
                               "curtain": "窗帘", "tv": "电视"}.get(did, "")
                    if room_cn and type_cn:
                        intent["device_name"] = room_cn + type_cn
        print(f"[batch_execute_intents] ★ 房间锁定(text={text!r}): room={explicit_room!r}，跳过 clarify")

    for intent in intents:
        device_id = intent.get("device_id", "")
        risk = get_device_risk(device_id)

        # 高风险通用设备需要消歧（但用户已给明确房间时跳过）
        if device_id in ("air_conditioner", "light", "curtain", "tv"):
            if risk == "high_risk" and not explicit_room and not intent.get("_explicit_room_slug"):
                has_clarification = True
                device_name = intent.get("device_name", "该设备")
                candidates = {
                    "curtain": "卧室窗帘、客厅窗帘",
                }
                options = candidates.get(device_id, "不同房间的同类设备")
                clarification_msg = f"您想操作的是哪个房间的{device_name}？可选：{options}"
                clarification_intent = intent
                continue
            else:
                intent["device_id"] = resolve_device_id(device_id)

        execution = await execute_command(intent)
        if execution["success"]:
            messages.append(execution["message"])
            # 记录最后操作的设备，用于上下文追踪
            if session_id:
                dev_id = execution.get("resolved_device_id", device_id)
                if dev_id:
                    set_last_device(session_id, dev_id)
        else:
            messages.append(execution["message"])

    # 保存消歧上下文
    if has_clarification and clarification_intent and session_id:
        dialog_manager.save_pending(session_id, clarification_intent)

    if has_clarification and clarification_msg:
        reply = ("，".join(m.strip() for m in messages if isinstance(m, str) and m.strip())
                 + "。另外，" + clarification_msg) if messages else clarification_msg
        if not reply.strip():
            reply = "操作已完成。"
        return reply

    if not messages:
        return "抱歉，无法识别您的指令。"

    # 清除旧上下文
    if session_id:
        dialog_manager.clear(session_id)

    # 合并：如果有多个成功，用"和"连接
    if len(messages) == 1:
        s = messages[0]
        if isinstance(s, str) and s.strip():
            return s
        return "操作已完成。"

    # 提取共性前缀，生成自然的合并句
    merged = _merge_messages(messages)
    # —— ★ batch_execute_intents 最终出口：非空兜底 ——
    if not isinstance(merged, str) or not merged.strip():
        merged = "操作已完成。"
        print(f"[Reply-Safe] batch_execute_intents: 合并后为空，兜底 → {merged!r}")
    return merged


def _merge_messages(messages: list) -> str:
    """
    合并多条回复为一句自然中文。

    例如：["已为您开启卧室空调", "已为您关闭客厅电视"]
    → "已为您开启卧室空调，关闭客厅电视"
    """
    if not messages:
        return ""
    if len(messages) == 1:
        return messages[0]

    # 特殊优化：多条"已为您打开所有X" → "已为您打开所有X和Y"
    all_pattern = re.compile(r"^已为您(打开|关闭|开启|设置)所有(.+)$")
    all_matches = [all_pattern.match(m) for m in messages]
    if all(all_matches):
        verbs = set(m.group(1) for m in all_matches)
        if len(verbs) == 1:
            verb = all_matches[0].group(1)
            objects = [m.group(2) for m in all_matches]
            return f"已为您{verb}了所有{'和'.join(objects)}"

    # 尝试提取公共前缀 "已为您"
    prefixes = []
    for msg in messages:
        if msg.startswith("已为您"):
            prefixes.append("已为您")
        else:
            prefixes.append("")

    # 如果所有消息的公共前缀相同
    if prefixes and all(p == prefixes[0] for p in prefixes if p):
        common = prefixes[0]
        rest_parts = []
        for msg in messages:
            if msg.startswith(common):
                rest_parts.append(msg[len(common):])
            else:
                rest_parts.append(msg)
        return common + "，".join(rest_parts)

    # 否则用逗号连接
    return "，".join(messages)


# ============ Dify 对接接口已移除 ============
# 项目全面使用 SmartBrain 本地大脑，不再依赖 Dify 服务。
# 相关接口 (/api/v1/parse_intent, /api/v1/execute_device, /api/v1/dify_control)
# 已于 2026-08 清理。


@app.get("/api/v1/health")
async def health_check():
    snapshot = _normalize_device_states_snapshot()
    return {
        "status": "ok",
        "model_loaded": asr_model is not None,
        "devices": list(DEVICE_STATES.keys()),
        "device_states": snapshot,
        "llm_available": llm_available,
        "llm_mode": "siliconflow",
        "rule_confidence_threshold": RULE_CONFIDENCE_THRESHOLD,
    }


# ============ Mock Home Assistant 接口（兼容 HA 原生路径 + 后端自定义路径） ============
# HA 原生路径: /api/states, /api/states/{entity_id}, /api/services/{domain}/{service}
# 后端自定义路径: /api/v1/ha/* （供前端使用）

@app.get("/api/states")
@app.get("/api/v1/ha/states")
async def ha_get_states(room: Optional[str] = Query(default=None, description="按房间过滤")):
    """获取所有设备状态"""
    states = []
    for dev_id, state in DEVICE_STATES.items():
        if room and state.get("room") != room:
            continue
        snap = copy.deepcopy(state)
        _normalize_ac_state(snap)
        states.append({
            "entity_id": dev_id,
            "state": snap.get("power", snap.get("state", "off")),
            "attributes": snap,
        })
    return states


@app.get("/api/states/{entity_id}")
@app.get("/api/v1/ha/states/{entity_id}")
async def ha_get_entity_state(entity_id: str):
    """获取单个设备状态（支持中文/英文/HA格式的 entity_id）"""
    raw_id = resolve_entity_id(entity_id)
    if not raw_id:
        raise HTTPException(status_code=404, detail=f"设备 {entity_id} 不存在")

    state = copy.deepcopy(DEVICE_STATES[raw_id])
    _normalize_ac_state(state)
    return {
        "entity_id": entity_id,
        "state": state.get("power", state.get("state", "off")),
        "attributes": state,
    }

@app.post("/api/services/{domain}/{service}")
@app.post("/api/v1/ha/services/{domain}/{service}")
async def ha_call_service(
    domain: str,
    service: str,
    request: Request,
    entity_id: Optional[str] = Query(default=None, description="设备ID，如 bedroom_ac"),
    temperature: Optional[int] = Query(default=None),
    mode: Optional[str] = Query(default=None),
    fan_speed: Optional[str] = Query(default=None),
    brightness: Optional[int] = Query(default=None),
):
    """调用设备服务（模拟 Home Assistant /api/services/{domain}/{service}）
    支持 query 参数和 JSON body 两种传参方式
    """
    # 支持 JSON body 传参（HA 原生方式）
    if entity_id is None:
        try:
            body = await request.json()
            entity_id = body.get("entity_id")
            if temperature is None:
                temperature = body.get("temperature")
            if mode is None:
                mode = body.get("mode")
            if fan_speed is None:
                fan_speed = body.get("fan_speed") or body.get("fan_mode")
            if brightness is None:
                brightness = body.get("brightness")
        except Exception:
            pass

    if not entity_id:
        raise HTTPException(status_code=400, detail="缺少 entity_id 参数")

    # 批量操作支持
    ALL_TARGETS_MAP = {
        "all": None, "light.all": "light", "all_lights": "light",
        "所有灯": "light", "所有的灯": "light", "全部灯": "light",
        "cover.all": "curtain", "all_curtains": "curtain",
        "所有窗帘": "curtain", "所有的窗帘": "curtain",
        "climate.all": "ac", "all_acs": "ac", "所有空调": "ac",
        "media_player.all": "tv", "all_tvs": "tv", "所有电视": "tv",
    }
    if entity_id in ALL_TARGETS_MAP:
        target_type = ALL_TARGETS_MAP[entity_id]
        if target_type:
            target_devices = [(k, v) for k, v in DEVICE_STATES.items() if v.get("type") == target_type]
            type_cn = {"ac": "空调", "light": "灯", "tv": "电视", "curtain": "窗帘"}.get(target_type, "设备")
        else:
            target_devices = list(DEVICE_STATES.items())
            type_cn = "设备"

        changed = 0
        for dev_key, dev_state in target_devices:
            dev_type = dev_state.get("type", "")
            if dev_type == "curtain":
                if service in ("turn_on", "on"):
                    if dev_state.get("state") != "open":
                        dev_state["state"] = "open"
                        changed += 1
                elif service in ("turn_off", "off"):
                    if dev_state.get("state") != "closed":
                        dev_state["state"] = "closed"
                        changed += 1
            else:
                if service in ("turn_on", "on"):
                    if dev_state.get("power") != "on":
                        dev_state["power"] = "on"
                        changed += 1
                elif service in ("turn_off", "off"):
                    if dev_state.get("power") != "off":
                        dev_state["power"] = "off"
                        changed += 1

        act_cn = "开启" if service in ("turn_on", "on") else "关闭"
        return {
            "success": True,
            "already_in_target_state": changed == 0,
            "message": f"已为您{act_cn}所有{type_cn}" if changed > 0 else f"所有{type_cn}已经是目标状态了",
            "entity_id": entity_id,
            "changed_count": changed,
        }

    # 使用统一的实体 ID 解析（支持中文/英文/HA 格式）
    raw_id = resolve_entity_id(entity_id)
    if not raw_id:
        raise HTTPException(status_code=404, detail=f"设备 {entity_id} 不存在")

    state = DEVICE_STATES[raw_id]

    attrs = {}
    if temperature is not None:
        attrs["temperature"] = temperature
    if mode is not None:
        attrs["mode"] = AC_MODE_MAP.get(mode, mode)
    if fan_speed is not None:
        # 支持相对调整：如 "+1"、"调大一档"
        current_fan = state.get("fan_speed") if state.get("type") == "ac" else None
        resolved = _resolve_fan_speed(fan_speed, current_fan)
        if isinstance(resolved, str) and resolved in AC_FAN_MAP:
            attrs["fan_speed"] = AC_FAN_MAP[resolved]
        else:
            attrs["fan_speed"] = resolved if not isinstance(resolved, str) else AC_FAN_MAP.get(resolved, resolved)
    if brightness is not None:
        attrs["brightness"] = brightness

    state_type = state.get("type", "")
    room = state.get("room", "")

    # 统一获取当前状态
    def get_current_state():
        if state_type == "curtain":
            return state.get("state", "closed")
        else:
            return state.get("power", "off")

    # 统一设置状态
    def set_current_state(new_val):
        if state_type == "curtain":
            state["state"] = new_val
        else:
            state["power"] = new_val

    # 判断是否为"开"的状态
    def is_on_state(val):
        if state_type == "curtain":
            return val == "open"
        else:
            return val == "on"

    prev_state = get_current_state()
    display = room + {"ac": "空调", "light": "灯", "tv": "电视", "curtain": "窗帘"}.get(state_type, "设备")

    # 判断是否为纯开关操作（无其他属性变更）
    has_extra_attrs = bool(attrs)

    # 检查是否已经是目标状态
    if not has_extra_attrs:
        if service in ("turn_on", "on") and is_on_state(prev_state):
            return {
                "success": True,
                "already_in_target_state": True,
                "message": f"{display}已经开着了",
                "entity_id": entity_id,
                "new_state": prev_state,
            }
        elif service in ("turn_off", "off") and not is_on_state(prev_state):
            return {
                "success": True,
                "already_in_target_state": True,
                "message": f"{display}已经关了",
                "entity_id": entity_id,
                "new_state": prev_state,
            }

    # 执行状态变更
    if service in ("turn_on", "on"):
        if state_type == "curtain":
            set_current_state("open")
        else:
            set_current_state("on")
    elif service in ("turn_off", "off"):
        if state_type == "curtain":
            set_current_state("closed")
        else:
            set_current_state("off")
    elif service == "toggle":
        if state_type == "curtain":
            set_current_state("closed" if prev_state == "open" else "open")
        else:
            set_current_state("off" if prev_state == "on" else "on")

    state.update(attrs)

    new_state_val = get_current_state()
    return {
        "success": True,
        "already_in_target_state": False,
        "entity_id": entity_id,
        "new_state": new_state_val,
    }


# 批量操作映射表
ALL_TARGETS_MAP = {
    "all": None, "light.all": "light", "all_lights": "light",
    "所有灯": "light", "所有的灯": "light", "全部灯": "light",
    "cover.all": "curtain", "curtain.all": "curtain", "all_curtains": "curtain",
    "所有窗帘": "curtain", "所有的窗帘": "curtain",
    "climate.all": "ac", "ac.all": "ac", "all_acs": "ac", "all_ac": "ac",
    "所有空调": "ac",
    "media_player.all": "tv", "tv.all": "tv", "all_tvs": "tv", "all_tv": "tv",
    "所有电视": "tv",
}

# 场景模式映射表：用户关键词 → 需要执行的设备操作列表
SCENE_PRESETS: Dict[str, Dict[str, Any]] = {
    "sleep": {
        "keywords": ["睡觉", "晚安", "睡眠", "睡了", "休息", "要睡", "入眠"],
        "reply": "晚安，已为您切换到睡眠模式",
        "actions": [
            {"entity_id": "light.all", "action": "turn_off"},
            {"entity_id": "tv.all", "action": "turn_off"},
            {"entity_id": "curtain.all", "action": "turn_off"},
            {"entity_id": "living_room_ac", "action": "turn_off"},
            # 卧室空调条件控制：只有开着时才调到26度1档，关着则保持原状
            {
                "entity_id": "bedroom_ac",
                "action": "turn_on",
                "condition": {"power": "on"},
                "value": 26,
                "fan_mode": "1档",
            },
        ],
        "description": "睡眠模式",
    },
    "away": {
        "keywords": ["出门", "离家", "外出", "上班", "离开", "全关", "全部关闭"],
        "reply": "注意安全，已为您切换到离家模式",
        "actions": [
            {"entity_id": "light.all", "action": "turn_off"},
            {"entity_id": "tv.all", "action": "turn_off"},
            {"entity_id": "curtain.all", "action": "turn_off"},
            {"entity_id": "ac.all", "action": "turn_off"},
        ],
        "description": "离家模式",
    },
    "movie": {
        "keywords": ["看电影", "电影模式", "观影"],
        "reply": "已为您切换到观影模式，祝您观影愉快",
        "actions": [
            {"entity_id": "light.all", "action": "turn_off"},
            {"entity_id": "curtain.all", "action": "turn_off"},
        ],
        "description": "观影模式",
    },
}


def detect_scene_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    从用户输入文本中检测场景模式。

    Returns:
        匹配的场景配置，或 None
    """
    if not text:
        return None
    for scene_key, scene_cfg in SCENE_PRESETS.items():
        keywords = scene_cfg.get("keywords", [])
        for kw in keywords:
            if kw in text:
                print(f"[SceneDetect] 检测到场景 '{scene_cfg['description']}'，触发关键词: '{kw}'")
                return {
                    "scene_key": scene_key,
                    "description": scene_cfg.get("description", scene_key),
                    "reply": scene_cfg.get("reply", ""),
                    "actions": scene_cfg.get("actions", []),
                }
    return None


async def execute_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行场景模式中的所有设备操作。

    Args:
        scene: detect_scene_from_text 返回的场景配置

    Returns:
        执行结果汇总

    支持条件动作：action_cfg 中的 "condition" 字段用于判断是否执行该动作。
    例如 {"condition": {"power": "on"}} 表示只有当设备 power 为 "on" 时才执行。
    """
    actions = scene.get("actions", [])
    scene_desc = scene.get("description", "场景")
    results = []
    executed_messages = []
    skipped_count = 0
    already_count = 0

    for action_cfg in actions:
        eid = action_cfg.get("entity_id", "")
        act = action_cfg.get("action", "")
        if not eid or not act:
            continue

        # 条件检查：判断是否应该执行该动作
        condition = action_cfg.get("condition")
        if condition:
            # 解析实际设备ID
            raw_id = resolve_entity_id(eid)
            if raw_id:
                async with state_lock:
                    dev_state = DEVICE_STATES.get(raw_id, {})
                # 检查所有条件字段
                should_execute = True
                for cond_key, cond_val in condition.items():
                    actual_val = dev_state.get(cond_key, "")
                    if actual_val != cond_val:
                        should_execute = False
                        break
                if not should_execute:
                    skipped_count += 1
                    print(f"[SceneDetect] 跳过 {eid}：条件不满足 {condition}（当前状态: {dev_state.get('power', 'unknown')}）")
                    continue

        # 提取可选参数
        value = action_cfg.get("value")
        fan_mode = action_cfg.get("fan_mode", "")
        hvac_mode = action_cfg.get("hvac_mode", "")

        result = await _execute_single_command(
            eid, act, hvac_mode=hvac_mode, value=value, fan_mode=fan_mode
        )
        results.append(result)
        msg = result.get("message", "")
        if msg:
            # 过滤"已是目标状态"等冗余消息
            already_patterns = ["已经都是目标状态", "已是目标状态", "已经开着了", "已经关了"]
            is_redundant = any(p in msg for p in already_patterns)
            if is_redundant:
                already_count += 1
            else:
                executed_messages.append(msg)

    all_ok = all(r.get("success", False) for r in results) if results else True

    # 优先采用场景预设的简洁回复，避免把子操作逐条列出导致啰嗦
    combined_msg = scene.get("reply", "").strip()
    has_new_changes = len(executed_messages) > 0
    if not combined_msg:
        # 兜底：如果没有预设 reply，才拼接动态消息
        if executed_messages:
            combined_msg = f"已为您执行{scene_desc}"
            combined_msg += "：" + "，".join(executed_messages)
        elif skipped_count > 0:
            combined_msg = f"已为您切换到{scene_desc}"

    return {
        "success": all_ok,
        "message": combined_msg,
        "scene_description": scene_desc,
        "total_actions": len(actions),
        "executed_count": len(results),
        "skipped_count": skipped_count,
        "already_count": already_count,
        "has_new_changes": has_new_changes or bool(scene.get("reply")),
        "results": results,
    }

# 设备类型中文名
TYPE_NAMES = {"ac": "空调", "light": "灯", "tv": "电视", "curtain": "窗帘"}

# 空调属性映射（规范化：将 Dify/HA 英文值统一为前端可识别的标准值）
AC_MODE_MAP = {
    "cool": "cooling", "heat": "heating", "dry": "dehumidify",
    "cooling": "cooling", "heating": "heating", "dehumidify": "dehumidify",
    "fan": "fan", "auto": "auto",
}
AC_MODE_CN_MAP = {
    "cooling": "制冷", "heating": "制热", "dehumidify": "除湿",
    "fan": "送风", "auto": "自动",
}
AC_SWING_MAP = {
    "on": "horizontal", "left_right": "horizontal", "horizontal": "horizontal",
    "up_down": "vertical", "vertical": "vertical", "off": "off",
    "stop": "off",
}
AC_SWING_CN_MAP = {
    "horizontal": "左右扫风", "vertical": "上下扫风", "off": "关扫风",
}
AC_FAN_MAP = {
    "quiet": "quiet", "auto": "auto",
    "1档": 1, "2档": 2, "3档": 3, "4档": 4, "5档": 5,
    "一挡": 1, "二挡": 2, "三挡": 3, "四挡": 4, "五挡": 5,
    "一档": 1, "二档": 2, "三档": 3, "四档": 4, "五档": 5,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
    "low": 1, "medium": 2, "mid": 2, "high": 3,
    "低档": 1, "低速": 1, "低风": 1, "小风": 1, "小档": 1,
    "中档": 2, "中速": 2, "中风": 2,
    "高档": 3, "高速": 3, "高风": 3, "大风": 3, "大档": 3,
    "静音档": "quiet", "静音风": "quiet", "静音": "quiet", "最低风": "quiet",
    "自动风": "auto", "自动档": "auto",
}
AC_FAN_CN_MAP = {
    1: "1档", 2: "2档", 3: "3档", 4: "4档", 5: "5档",
    "quiet": "静音", "auto": "自动",
}


def _resolve_fan_speed(fan_input: Any, current_fan: Any = None) -> Any:
    """将任意形式的风力输入解析为最终的标准化风速值。

    支持：
      - 绝对值：1、"2档"、"medium"、"低风" 等（走 AC_FAN_MAP）
      - 相对增减："+1"、"-1"、"1→+1档"、"调大一档"、"调小两档"、"调高"、"调低"、"大一档"、"小一档"
      - 若无法解析相对调整且 current_fan 可用，返回 current_fan（保持不变）
    """
    if fan_input is None or fan_input == "":
        return None

    # 先尝试直接匹配绝对值
    if fan_input in AC_FAN_MAP:
        return AC_FAN_MAP[fan_input]

    s = str(fan_input).strip()
    if not s:
        return None

    # 相对调整：显式的 +N / -N / +N档 / -N档
    m = re.match(r'^\s*([+-])\s*(\d+)\s*(档|挡)?\s*$', s)
    if m and current_fan is not None and isinstance(current_fan, int):
        delta = int(m.group(2)) * (1 if m.group(1) == '+' else -1)
        base = current_fan
        if not isinstance(base, int):
            base = 2  # 若当前不是数字档位，默认从 2 档开始调整
        new_val = max(1, min(5, base + delta))
        return new_val

    # 相对调整：中文关键词
    cn_keywords_incr = [r'调大', r'调高', r'加大', r'增强', r'大一档', r'大.*档', r'升.*档']
    cn_keywords_decr = [r'调小', r'调低', r'减小', r'减弱', r'小一档', r'小.*档', r'降.*档']

    def _get_cn_delta(text: str) -> int:
        # 匹配"调大两档"、"调小1档"这类数量词
        num_map = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}
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

    if any(re.search(p, s) for p in cn_keywords_incr) and isinstance(current_fan, int):
        d = _get_cn_delta(s)
        new_val = max(1, min(5, current_fan + d))
        return new_val
    if any(re.search(p, s) for p in cn_keywords_decr) and isinstance(current_fan, int):
        d = _get_cn_delta(s)
        new_val = max(1, min(5, current_fan - d))
        return new_val

    # 最后尝试：若纯数字字符串
    if re.match(r'^\d+$', s):
        n = int(s)
        if 1 <= n <= 5:
            return n
    if re.match(r'^(\d+)\s*(档|挡)$', s):
        m3 = re.match(r'^(\d+)\s*(档|挡)$', s)
        n = int(m3.group(1))
        if 1 <= n <= 5:
            return n

    # 无法识别：原样返回（后续代码会尝试 AC_FAN_MAP.get）
    return fan_input


def _normalize_ac_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """将设备状态字典中的 mode/swing/fan_speed 统一规范化为前端可识别的值。

    前端 index.html 仅识别 cooling/heating/dehumidify/fan/auto、horizontal/vertical/off、quiet/auto/1-5。
    若 Dify 或其它路径写入了 cool/heat/on/left_right 等原始英文值，会导致前端渲染成英文原文。
    此函数在发送状态快照前调用，保证界面始终显示中文 / 规范值。
    """
    if not state or not isinstance(state, dict):
        return state
    if state.get("type") != "ac":
        return state

    mode = state.get("mode")
    if isinstance(mode, str) and mode:
        state["mode"] = AC_MODE_MAP.get(mode, mode)

    swing = state.get("swing")
    if isinstance(swing, str) and swing:
        state["swing"] = AC_SWING_MAP.get(swing, swing)

    fan_speed = state.get("fan_speed")
    if fan_speed is not None and fan_speed != "":
        state["fan_speed"] = AC_FAN_MAP.get(fan_speed, fan_speed)

    return state


def _normalize_device_states_snapshot() -> Dict[str, Dict[str, Any]]:
    """生成已规范化的设备状态快照（深拷贝，避免污染运行时状态）。"""
    snapshot: Dict[str, Dict[str, Any]] = {}
    for dev_id, state in DEVICE_STATES.items():
        snap = copy.deepcopy(state)
        _normalize_ac_state(snap)
        snapshot[dev_id] = snap
    return snapshot


async def _execute_single_command(
    entity_id: str, action: str, hvac_mode: str = "", value: Any = None, fan_mode: str = "", swing_mode: str = "",
) -> Dict[str, Any]:
    """执行单条设备控制命令，返回结果字典

    直接复用 device_controller.execute_command，保证 MQTT 发布与设备状态更新一致。
    自动补全 device_name（基于 entity_id 推导中文显示名）。
    """
    from device_controller import execute_command, DEVICE_STATES, DEVICE_TYPE_NAMES

    def _resolve_display_name(eid: str) -> str:
        # .all 批量：curtain.all -> 窗帘
        if eid.endswith(".all"):
            short_type = eid[:-4]
            return DEVICE_TYPE_NAMES.get(short_type, short_type)
        # 精确设备：bedroom_ac -> 卧室空调
        if eid in DEVICE_STATES:
            info = DEVICE_STATES[eid]
            room = info.get("room", "")
            t = info.get("type", "")
            if room and t:
                return f"{room}{DEVICE_TYPE_NAMES.get(t, t)}"
            return info.get("name", eid)
        # 通用别名兜底：air_conditioner / tv / light / curtain
        alias_map = {"air_conditioner": "空调", "ac": "空调", "tv": "电视", "light": "灯", "curtain": "窗帘"}
        return alias_map.get(eid, eid)

    intent = {
        "device_id": entity_id,
        "device_name": _resolve_display_name(entity_id),
        "action": action,
        "value": value,
        "hvac_mode": hvac_mode,
        "fan_mode": fan_mode,
        "swing_mode": swing_mode,
        "attrs": {
            "hvac_mode": hvac_mode,
            "fan_mode": fan_mode,
            "swing_mode": swing_mode,
            "temperature": value,
        },
    }
    # 动作标准化：场景配置常用 turn_on/turn_off，统一为 on/off/set
    alias_map = {
        "turn_on": "on", "on": "on", "open": "on",
        "turn_off": "off", "off": "off", "close": "off",
        "toggle": "toggle",
        "set_fan_mode": "set", "set_hvac_mode": "set", "set_swing_mode": "set",
        "set_temperature": "set", "set_value": "set", "set_ac": "set", "set": "set",
        "adjust": "set", "adjust_ac": "set",
    }
    intent["action"] = alias_map.get(intent["action"], intent["action"])
    # 清洗 attrs：移除空字符串，避免空调逻辑把空值当成"有属性变更"
    clean_attrs = {k: v for k, v in (intent.get("attrs") or {}).items() if v not in (None, "", [])}
    intent["attrs"] = clean_attrs
    # 仅当存在温度时标记 value_type=temperature，让空调逻辑走温度设置分支
    if value is not None:
        intent["value_type"] = "temperature"
    return await execute_command(intent)
async def _execute_single_command_impl(entity_id: str, action: str, hvac_mode: str = "", value: Any = None, fan_mode: str = "", swing_mode: str = "") -> Dict[str, Any]:
    """[内部实现] 执行单条设备控制命令，返回结果字典

    支持参数：value(温度), fan_mode(风速), swing_mode(扫风)
    支持 entity_id 逗号分隔多设备批量控制，如 "light.all, climate.all"
    支持 Dify 风格的 action 别名：set_fan_mode/set_hvac_mode/set_swing_mode/set_temperature/set_value/set_ac
    """
# ============ Action 别名规范化（Dify 大脑返回 set_fan_mode 等自定义动作名） ============
    # 这些动作本质上都是"只改属性，不直接开关电源"
    ACTION_ALIAS_MAP = {
        "set_fan_mode": "set",
        "set_hvac_mode": "set",
        "set_swing_mode": "set",
        "set_temperature": "set",
        "set_value": "set",
        "set_ac": "set",
        "set_ac_mode": "set",
        "set_fan_speed": "set",
        "set_swing": "set",
        "adjust": "set",
        "adjust_ac": "set",
    }
    original_action = action
    action = ACTION_ALIAS_MAP.get(action, action)

    # 如果是 set_* 动作但没有带任何属性，且是空调，保持原样让后续逻辑处理
    is_set_attrs_action = original_action in ACTION_ALIAS_MAP
    # 逗号分隔多设备：拆分为多个 entity_id 分别执行
    if entity_id and "," in entity_id:
        eid_list = [e.strip() for e in entity_id.split(",") if e.strip()]
        if len(eid_list) > 1:
            results = []
            messages = []
            total_changed = 0
            total_skipped = 0
            for sub_eid in eid_list:
                sub_result = await _execute_single_command_impl(sub_eid, action, hvac_mode, value, fan_mode, swing_mode)
                results.append(sub_result)
                if sub_result.get("message"):
                    messages.append(sub_result["message"])
                total_changed += sub_result.get("changed_count", 0)
                total_skipped += sub_result.get("skipped_count", 0)
            combined_msg = "，".join(messages) if messages else "指令已处理"
            return {
                "success": all(r.get("success", False) for r in results),
                "message": combined_msg,
                "entity_id": entity_id,
                "multi_device": True,
                "sub_results": results,
                "changed_count": total_changed,
                "skipped_count": total_skipped,
            }

    # 批量操作
    if entity_id in ALL_TARGETS_MAP:
        async with state_lock:
            target_type = ALL_TARGETS_MAP[entity_id]
            if target_type:
                target_devices = [(k, v) for k, v in DEVICE_STATES.items() if v.get("type") == target_type]
                type_cn = TYPE_NAMES.get(target_type, "设备")
            else:
                target_devices = list(DEVICE_STATES.items())
                type_cn = "设备"

            changed = 0
            skipped = 0
            for dev_key, dev_state in target_devices:
                dev_type = dev_state.get("type", "")
                if dev_type == "curtain":
                    cur = dev_state.get("state", "closed")
                    is_on = cur == "open"
                    if action in ("turn_on", "on"):
                        if is_on:
                            skipped += 1
                            continue
                        dev_state["state"] = "open"
                    elif action in ("turn_off", "off"):
                        if not is_on:
                            skipped += 1
                            continue
                        dev_state["state"] = "closed"
                    elif action == "toggle":
                        dev_state["state"] = "closed" if is_on else "open"
                    changed += 1
                else:
                    cur = dev_state.get("power", "off")
                    is_on = cur == "on"
                    if action in ("turn_on", "on"):
                        if is_on:
                            skipped += 1
                            continue
                        dev_state["power"] = "on"
                    elif action in ("turn_off", "off"):
                        if not is_on:
                            skipped += 1
                            continue
                        dev_state["power"] = "off"
                    elif action == "toggle":
                        dev_state["power"] = "off" if is_on else "on"
                    changed += 1

            act_cn = "开启" if action in ("turn_on", "on") else ("关闭" if action in ("turn_off", "off") else "切换")
            msg = f"所有{type_cn}已经都是目标状态了" if changed == 0 else f"已为您{act_cn}所有{type_cn}"
            return {
                "success": True,
                "already_in_target_state": changed == 0,
                "message": msg,
                "entity_id": entity_id,
                "changed_count": changed,
                "skipped_count": skipped,
                "device_type": target_type or "all",
            }

    # 单设备操作
    raw_id = resolve_entity_id(entity_id)
    if not raw_id:
        available = list(DEVICE_STATES.keys()) + list(DEVICE_NAME_MAP.keys())
        return {
            "success": False,
            "message": f"设备 {entity_id} 不存在，可用设备: {', '.join(available[:8])}",
            "entity_id": entity_id,
        }

    async with state_lock:
        state = DEVICE_STATES[raw_id]
        state_type = state.get("type", "")
        room = state.get("room", "")
        display = room + TYPE_NAMES.get(state_type, "设备")

        # 统一获取/设置状态
        def get_current_state():
            if state_type == "curtain":
                return state.get("state", "closed")
            return state.get("power", "off")

        def set_current_state(new_val):
            if state_type == "curtain":
                state["state"] = new_val
            else:
                state["power"] = new_val

        def is_on_state(val):
            if state_type == "curtain":
                return val == "open"
            return val == "on"

        prev_state = get_current_state()

        # toggle
        if action == "toggle":
            if state_type == "curtain":
                new_val = "closed" if prev_state == "open" else "open"
            else:
                new_val = "off" if prev_state == "on" else "on"
            set_current_state(new_val)
            action_cn = "拉开" if (state_type == "curtain" and new_val == "open") or (state_type != "curtain" and new_val == "on") else "关闭"
            return {
                "success": True, "already_in_target_state": False,
                "message": f"已为您{action_cn}{display}",
                "entity_id": entity_id, "new_state": new_val, "device_type": state_type,
            }

        # 更新空调属性（温度、风速、扫风）—— 使用模块级规范化映射表
        def update_ac_extra():
            extra_parts = []
            if value is not None and value != "":
                state["temperature"] = int(value) if isinstance(value, (int, float)) else value
                extra_parts.append(f"温度调至{value}度")
            if fan_mode and fan_mode != "":
                # 支持相对调整：当前档位 +1 / -1，如 "调大一档"、"+1"
                current_fan = state.get("fan_speed")
                resolved_fan = _resolve_fan_speed(fan_mode, current_fan)
                # 若解析后还是非标准的字符串，再过一遍 AC_FAN_MAP
                if isinstance(resolved_fan, str) and resolved_fan in AC_FAN_MAP:
                    normalized_fan = AC_FAN_MAP[resolved_fan]
                elif isinstance(resolved_fan, str):
                    normalized_fan = AC_FAN_MAP.get(resolved_fan, resolved_fan)
                else:
                    normalized_fan = resolved_fan
                state["fan_speed"] = normalized_fan
                fan_cn = AC_FAN_CN_MAP.get(normalized_fan, (str(normalized_fan) + "档" if isinstance(normalized_fan, int) else str(normalized_fan)))
                extra_parts.append(f"风速设为{fan_cn}")
            if swing_mode and swing_mode != "":
                normalized_swing = AC_SWING_MAP.get(swing_mode, swing_mode)
                state["swing"] = normalized_swing
                swing_cn = AC_SWING_CN_MAP.get(normalized_swing, swing_mode)
                extra_parts.append(f"开启{swing_cn}" if normalized_swing != "off" else swing_cn)
            return extra_parts

        # turn_on
        if action == "turn_on":
            # 空调模式切换
            if state_type == "ac" and hvac_mode:
                # 规范化 hvac_mode：cool→cooling, heat→heating 等
                normalized_mode = AC_MODE_MAP.get(hvac_mode, hvac_mode)
                mode_cn = AC_MODE_CN_MAP.get(normalized_mode, hvac_mode)
                current_mode = state.get("mode", "")

                # 检查是否真的需要切换模式
                mode_changed = current_mode != normalized_mode
                if mode_changed:
                    state["mode"] = normalized_mode

                if is_on_state(prev_state):
                    # 已开启，只切换模式 + 更新属性
                    extra_parts = update_ac_extra()
                    action_cn = f"切换为{mode_cn}模式" + (f"，{'、'.join(extra_parts)}" if extra_parts else "")
                    if mode_changed:
                        msg = f"已为您将{display}{action_cn}"
                    else:
                        msg = f"{display}已经开着了" + (f"，{'、'.join(extra_parts)}" if extra_parts else "")
                    return {
                        "success": True, "already_in_target_state": False,
                        "message": msg,
                        "entity_id": entity_id, "new_state": prev_state, "device_type": state_type, "mode": normalized_mode,
                    }
                set_current_state("on")
                state["mode"] = normalized_mode
                extra_parts = update_ac_extra()
                extra_msg = f"，并切换为{mode_cn}模式" + (f"，{'、'.join(extra_parts)}" if extra_parts else "")
                return {
                    "success": True, "already_in_target_state": False,
                    "message": f"已为您开启{display}{extra_msg}",
                    "entity_id": entity_id, "new_state": "on", "device_type": state_type, "mode": normalized_mode,
                }

            # 空调（无模式指定）：开启 + 更新属性
            if state_type == "ac":
                extra_parts = update_ac_extra()
                if is_on_state(prev_state):
                    if extra_parts:
                        return {
                            "success": True, "already_in_target_state": False,
                            "message": f"已为您{display}{'、'.join(extra_parts)}",
                            "entity_id": entity_id, "new_state": prev_state, "device_type": state_type,
                        }
                    return {
                        "success": True, "already_in_target_state": True,
                        "message": f"{display}已经开着了",
                        "entity_id": entity_id, "new_state": prev_state, "device_type": state_type,
                    }
                set_current_state("on")
                extra_msg = f"，{'、'.join(extra_parts)}" if extra_parts else ""
                return {
                    "success": True, "already_in_target_state": False,
                    "message": f"已为您开启{display}{extra_msg}",
                    "entity_id": entity_id, "new_state": "on", "device_type": state_type,
                }

            # 普通 turn_on（灯、电视）
            if is_on_state(prev_state):
                return {
                    "success": True, "already_in_target_state": True,
                    "message": f"{display}已经开着了",
                    "entity_id": entity_id, "new_state": prev_state, "device_type": state_type,
                }
            if state_type == "curtain":
                set_current_state("open")
                new_state_val = "open"
            else:
                set_current_state("on")
                new_state_val = "on"
            return {
                "success": True, "already_in_target_state": False,
                "message": f"已为您开启{display}",
                "entity_id": entity_id, "new_state": new_state_val, "device_type": state_type,
            }

        # turn_off
        if action == "turn_off":
            if not is_on_state(prev_state):
                return {
                    "success": True, "already_in_target_state": True,
                    "message": f"{display}已经关了",
                    "entity_id": entity_id, "new_state": prev_state, "device_type": state_type,
                }
            if state_type == "curtain":
                set_current_state("closed")
                new_state_val = "closed"
            else:
                set_current_state("off")
                new_state_val = "off"
            return {
                "success": True, "already_in_target_state": False,
                "message": f"已为您关闭{display}",
                "entity_id": entity_id, "new_state": new_state_val, "device_type": state_type,
            }

        # set —— Dify 风格的属性调整动作（set_fan_mode/set_hvac_mode 等归一化后的结果）
        if action == "set":
            # AC 设备：若已关机则自动开启（用户要调属性，通常意味着要用空调）
            if state_type == "ac" and not is_on_state(prev_state):
                set_current_state("on")
                auto_on = True
            else:
                auto_on = False

            extra_parts = update_ac_extra()

            # hvac_mode 规范化
            if hvac_mode and state_type == "ac":
                normalized_mode = AC_MODE_MAP.get(hvac_mode, hvac_mode)
                mode_cn = AC_MODE_CN_MAP.get(normalized_mode, hvac_mode)
                current_mode = state.get("mode", "")
                if current_mode != normalized_mode:
                    state["mode"] = normalized_mode
                    extra_parts.insert(0, f"切换为{mode_cn}模式")

            if not extra_parts and not auto_on:
                return {
                    "success": True, "already_in_target_state": True,
                    "message": f"{display}已是目标状态",
                    "entity_id": entity_id, "new_state": get_current_state(), "device_type": state_type,
                }

            extra_msg = f"，并自动开启" if auto_on else ""
            details = f"，{'、'.join(extra_parts)}" if extra_parts else ""
            return {
                "success": True, "already_in_target_state": False,
                "message": f"已为您将{display}{details}{extra_msg}",
                "entity_id": entity_id, "new_state": get_current_state(), "device_type": state_type,
            }

        return {"success": False, "message": f"不支持的 action: {action}"}


@app.post("/api/v1/ha/smart_control")
async def ha_smart_control(request: Request):
    """智能控制：支持单条命令或 commands 数组批量执行"""
    try:
        body = await request.json()
        print(f"[SmartControl] 收到请求: {body}")
    except Exception as e:
        print(f"[SmartControl] JSON 解析失败: {e}")
        return {"success": False, "message": "请求体格式错误，需要 JSON", "error": str(e)}

    # 支持 commands 数组（批量）
    commands = body.get("commands")
    if commands and isinstance(commands, list) and len(commands) > 0:
        print(f"[SmartControl] 检测到 {len(commands)} 条命令")
        results = []
        messages = []
        for i, cmd in enumerate(commands):
            eid = cmd.get("entity_id", "")
            act = cmd.get("action", "")
            hm = cmd.get("hvac_mode", "") or cmd.get("mode", "")
            val = cmd.get("value") if cmd.get("value") is not None else cmd.get("temperature")
            # 兼容两种字段命名：fan_mode（新）/ fan_speed（Dify/其它路径常用）
            fm = cmd.get("fan_mode", "") or cmd.get("fan_speed", "")
            sm = cmd.get("swing_mode", "") or cmd.get("swing", "")
            if not eid or not act:
                results.append({"success": False, "message": f"第{i+1}条命令缺少参数"})
                continue
            result = await _execute_single_command(eid, act, hm, val, fm, sm)
            results.append(result)
            if result.get("message"):
                messages.append(result["message"])
            print(f"[SmartControl] 命令{i+1}: {eid} -> {result.get('message', 'OK')}")

        combined_msg = "，".join(messages) if messages else "指令已处理"
        all_success = all(r.get("success", False) for r in results)
        return {
            "success": all_success,
            "message": combined_msg,
            "total": len(commands),
            "results": results,
        }

    # 单条命令（兼容旧格式）
    entity_id = body.get("entity_id", "")
    action = body.get("action", "")
    hvac_mode = body.get("hvac_mode", "") or body.get("mode", "")
    value = body.get("value") if body.get("value") is not None else body.get("temperature")
    fan_mode = body.get("fan_mode", "") or body.get("fan_speed", "")
    swing_mode = body.get("swing_mode", "") or body.get("swing", "")

    print(f"[SmartControl] 单条命令: entity_id={entity_id}, action={action}, hvac_mode={hvac_mode}, value={value}, fan_mode={fan_mode}, swing_mode={swing_mode}")

    if not entity_id or not action:
        return {"success": False, "message": f"缺少参数或命令为空: entity_id={entity_id}, action={action}"}

    return await _execute_single_command(entity_id, action, hvac_mode, value, fan_mode, swing_mode)


@app.get("/api/v1/ha/debug")
async def ha_debug(entity_id: Optional[str] = None):
    """调试接口：测试 entity_id 解析"""
    if entity_id:
        raw_id = resolve_entity_id(entity_id)
        if raw_id:
            state = DEVICE_STATES[raw_id]
            return {
                "input": entity_id,
                "resolved": raw_id,
                "state": state,
                "status": "ok",
            }
        return {
            "input": entity_id,
            "resolved": None,
            "state": None,
            "status": "not_found",
            "available_keys": list(DEVICE_STATES.keys()),
            "available_names": list(DEVICE_NAME_MAP.keys()),
        }
    return {
        "device_states": list(DEVICE_STATES.keys()),
        "name_map": DEVICE_NAME_MAP,
    }


@app.post("/api/v1/voice_control")
async def voice_control(file: UploadFile = File(...), session_id: str = Query(default="default")):
    """端到端语音控制接口：上传语音 -> 识别 -> 意图提取 -> 设备控制（支持多轮对话上下文）"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件无效")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext and ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的音频格式: {ext}")

    task_id = str(uuid.uuid4())
    safe_ext = ext or ".wav"
    raw_path = f"temp_raw_{task_id}{safe_ext}"
    converted_path = f"temp_16k_{task_id}.wav"

    try:
        content = await file.read()

        # ---- 音频有效性预校验 ----
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="音频文件为空")
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"音频文件过大（{len(content) / 1024 / 1024:.1f}MB > 100MB）")
        if len(content) < MIN_FILE_SIZE:
            raise HTTPException(status_code=400, detail="音频文件过小，疑似无效录音")

        with open(raw_path, "wb") as f:
            f.write(content)

        # 音频时长校验（转码前快速检测）
        duration = _probe_audio_duration(raw_path)
        if 0 < duration < MIN_AUDIO_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=f"音频时长过短（{duration:.2f}秒 < {MIN_AUDIO_DURATION_SEC}秒），请录制更长的语音",
            )

        try:
            convert_to_16k_wav(raw_path, converted_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"音频转码失败: {str(e)}")

        # 转码后二次时长校验
        after_duration = _probe_audio_duration(converted_path)
        if 0 < after_duration < MIN_AUDIO_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=f"音频时长过短（{after_duration:.2f}秒），无法识别有效语音",
            )

        # B. ASR 语音识别
        asr_res = asr_model.generate(input=converted_path, cache={}, language="zh", use_itn=True)
        if not asr_res or len(asr_res) == 0:
            return {
                "code": 500,
                "msg": "音频未识别出有效文字",
                "text": "",
                "intent": None,
                "reply": None,
            }

        raw_text = asr_res[0]["text"]
        # ★ 统一预处理管道：洗标点/唤醒词残留/填充词 + 热词纠错 + 同音纠错
        recognized_text = preprocess_voice_text(raw_text)

        # ★ 空指令早退：避免无效文本触发 ASR 之后的 NLU / LLM / 上下文流程
        if is_effective_empty(recognized_text):
            return {
                "code": 200,
                "text": "",
                "intent": None,
                "method": "preprocess",
                "confidence": 0.0,
                "need_clarification": True,
                "reply": "未检测到有效语音指令，请重新表达",
                "device_state": None,
                "session_id": session_id,
            }

        # === 多轮对话上下文（需求 2 槽位补全 + 需求 3 改口重置） ===
        pending = dialog_manager.get_pending(session_id)
        if pending:
            # 情况A：是消歧回复（房间名/设备名）→ 合并槽位 + 立即执行
            if is_clarification_response(recognized_text, pending):
                room_slug = extract_room(recognized_text)
                if not room_slug:
                    room_cn = extract_room_cn(recognized_text)
                    if room_cn:
                        from dialog_context import ROOM_KEYWORDS
                        room_slug = ROOM_KEYWORDS.get(room_cn)
                if room_slug:
                    resolved_intent = pending.resolve_with_room(room_slug)
                    if resolved_intent:
                        print(f"[Dialog][Voice] 消歧回复 '{recognized_text}' → room={room_slug}, device_id={resolved_intent['device_id']}")
                        dialog_manager.clear(session_id, reason="clarification_resolved")
                        if resolved_intent.get("device_id"):
                            set_last_device(session_id, resolved_intent["device_id"])
                        execution = await execute_command(resolved_intent)
                        return {
                            "code": 200 if execution.get("success", False) else execution.get("code", 500),
                            "text": recognized_text,
                            "intent": resolved_intent,
                            "method": "context",
                            "confidence": 1.0,
                            "need_clarification": False,
                            "reply": execution.get("message", "指令已执行"),
                            "device_state": execution.get("state_snapshot"),
                            "session_id": session_id,
                        }
            # 情况B：有 pending 但用户改口 / 提取不到房间 → 清 pending，按新指令重处理
            dialog_manager.clear(session_id, reason="user_changed_mind")
            print(f"[Dialog][Voice] 用户改口或非澄清回复: '{recognized_text}'，清空 pending，重新解析")

        # C. 混合意图解析（规则优先 + LLM 兜底消歧）
        nlu_res = hybrid_parse_intent(recognized_text)

        # LLM 也认为需要消歧 / 高风险设备必须确认 -> 询问用户
        if nlu_res.get("need_clarification"):
            from dialog_context import (
                GENERIC_DEV_TO_TYPE, count_devices_of_type,
                get_candidate_rooms_for_type, is_generic_device_id,
            )
            intent_data = nlu_res.get("intent") or {}
            device_id = intent_data.get("device_id", "")
            device_name = intent_data.get("device_name", "该设备")
            dev_type = GENERIC_DEV_TO_TYPE.get(device_id, "")
            risk = get_device_risk(device_id) if device_id else "unknown"

            # 动态候选房间（基于 DEVICE_STATES 实际设备，不再硬编码）
            candidates_cn: list = []
            target_val = None
            action_label = intent_data.get("action") or ""
            if dev_type and count_devices_of_type(dev_type) > 1:
                candidates_cn = get_candidate_rooms_for_type(dev_type)
                # clamp target_val（空调温度 16-30 等）
                from smart_brain import _extract_target_val_with_clamp
                target_val = _extract_target_val_with_clamp(intent_data, dev_type)
                if not action_label and dev_type == "ac":
                    action_label = "set_temperature"

            # 保存完整的 pending_intent（结构化字段 + 候选房间）
            if intent_data:
                pending = dialog_manager.save_pending(
                    session_id=session_id,
                    intent=intent_data,
                    original_text=recognized_text,
                    target_device_type=dev_type,
                    target_val=target_val,
                    action_label=action_label,
                    candidate_rooms_cn=candidates_cn,
                )
                clarification_q = build_clarification_question(pending)
                pending_slot = pending.to_slot_dict()
            else:
                clarification_q = f"您想操作的是哪个房间的{device_name}？"
                pending_slot = {}

            print(f"[VoiceControl] 需要消歧: type={dev_type}, candidates={candidates_cn}, reply={clarification_q}")

            return {
                "code": 200,
                "text": recognized_text,
                "msg": "需要确认房间",
                "method": nlu_res.get("method", "rule"),
                "confidence": nlu_res.get("confidence", 0.0),
                "need_clarification": True,
                "risk_level": risk,
                "intent": nlu_res.get("intent"),
                "reply": clarification_q,
                "pending_intent": pending_slot,
                "session_id": session_id,
            }

        if not nlu_res["success"]:
            return {
                "code": 200,
                "text": recognized_text,
                "msg": nlu_res["msg"],
                "method": nlu_res.get("method", "rule"),
                "confidence": nlu_res.get("confidence", 0.0),
                "need_clarification": False,
                "intent": nlu_res.get("intent"),
                "reply": "抱歉，我听清了，但还不知道怎么控制该设备。",
            }

        # D. 设备指令执行
        execution = await execute_command(nlu_res["intent"])
        dialog_manager.clear(session_id)

        return {
            "code": 200 if execution["success"] else execution["code"],
            "text": recognized_text,
            "intent": nlu_res["intent"],
            "method": nlu_res.get("method", "rule"),
            "confidence": nlu_res.get("confidence", 0.0),
            "need_clarification": nlu_res.get("need_clarification", False),
            "reply": execution["message"],
            "device_state": execution.get("state_snapshot"),
        }

    except HTTPException:
        raise
    except Exception as e:
        return {
            "code": 500,
            "msg": f"服务内部错误: {str(e)}",
            "text": "",
            "intent": None,
            "reply": None,
        }
    finally:
        for p in [raw_path, converted_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# /api/v1/voice_dify — 已移除（Dify 依赖已清除，统一使用 SmartBrain）


# ============ TTS 语音合成接口 ============

class TTSRequest(BaseModel):
    text: str
    voice: str = "xiaoxiao"
    rate: str = "+0%"
    volume: str = "+0%"


@app.post("/api/v1/tts")
async def text_to_speech(req: TTSRequest):
    """文本转语音，返回音频文件。空/纯标点文本返回 200 + 零字节占位音频（不再抛 400/500）。"""
    voice_name = CHINESE_VOICES.get(req.voice, req.voice)
    task_id = str(uuid.uuid4())
    output_path = f"temp_tts_{task_id}.mp3"

    # ★【TTS 防空闸门 0 层：/api/v1/tts 入口】空/纯空白/纯标点 → 直接 200 + 占位 MP3，不调 edge-tts
    import re as _re_local
    _tts_in = (req.text or "").strip()
    _punct_stripped = _re_local.sub(
        r"[\s。，！？、；：,.!?;:'\"()（）《》【】\[\]<>/\\\-—…·]",
        "",
        _tts_in,
    )
    if not _punct_stripped:
        try:
            with open(output_path, "wb") as _f:
                pass
            print(f"[TTS-SAFEGUARD] /api/v1/tts 文本为空/纯标点，生成 0 字节占位 MP3: {output_path!r}")
            headers = {"X-TTS-Empty": "1"}
            return FileResponse(
                output_path,
                media_type="audio/mpeg",
                filename=f"tts_{task_id}.mp3",
                headers=headers,
            )
        except Exception as _e:
            raise HTTPException(status_code=500, detail=f"生成占位音频失败: {str(_e)}")

    try:
        await synthesize(req.text, output_path, voice=voice_name, rate=req.rate, volume=req.volume)
        return FileResponse(
            output_path,
            media_type="audio/mpeg",
            filename=f"tts_{task_id}.mp3",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 合成失败: {str(e)}")


@app.post("/api/v1/voice_tts")
async def voice_to_speech_reply(
    file: UploadFile = File(...),
    mode: str = Query(default="smart", description="smart / local"),
    voice: str = Query(default="xiaoxiao"),
    session_id: str = Query(default="default"),
):
    """
    语音闭环：上传语音 → ASR → (本地NLU/Dify/SmartBrain) → 文本回复 → TTS → 返回音频
    完整实现"语音进、语音出"
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件无效")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext and ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的音频格式: {ext}")

    task_id = str(uuid.uuid4())
    safe_ext = ext or ".wav"
    raw_path = f"temp_raw_{task_id}{safe_ext}"
    converted_path = f"temp_16k_{task_id}.wav"

    try:
        content = await file.read()

        # ---- 音频有效性预校验（与 asr_service.py 保持一致）----
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="音频文件为空")

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"音频文件过大（{len(content) / 1024 / 1024:.1f}MB > 100MB）")

        if len(content) < MIN_FILE_SIZE:
            raise HTTPException(status_code=400, detail="音频文件过小，疑似无效录音")

        with open(raw_path, "wb") as f:
            f.write(content)

        # 音频时长校验（在 ffmpeg 转码前快速检测）
        duration = _probe_audio_duration(raw_path)
        if 0 < duration < MIN_AUDIO_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=f"音频时长过短（{duration:.2f}秒 < {MIN_AUDIO_DURATION_SEC}秒），请录制更长的语音",
            )

        try:
            convert_to_16k_wav(raw_path, converted_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"音频转码失败: {str(e)}")

        # 转码后二次时长校验
        after_duration = _probe_audio_duration(converted_path)
        if 0 < after_duration < MIN_AUDIO_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=f"音频时长过短（{after_duration:.2f}秒），无法识别有效语音",
            )

        # A. ASR 识别
        asr_res = asr_model.generate(input=converted_path, cache={}, language="zh", use_itn=True)
        if not asr_res or len(asr_res) == 0:
            raise HTTPException(status_code=500, detail="音频未识别出有效文字")

        recognized_text = clean_text(asr_res[0]["text"])
        # ★ main.py 最外层：ASR 同音校正（语音接口 + 文本接口都要走一次，保证后续所有流程用校正后文本）
        corrected = asr_phonetic_correct(recognized_text)
        if corrected != recognized_text:
            print(f"[ASR] ★ 拼音校正: '{recognized_text}' → '{corrected}'")
        recognized_text = corrected
        print(f"[ASR] 识别结果: {recognized_text}")

        # 获取回复文本
        reply_text = ""
        if mode in ("smart", "dify"):
            # Smart 模式 / Dify 模式：统一走本地大脑（意图路由 + 设备控制 + RAG）
            from smart_brain import process_with_scene
            print(f"[SmartBrain] 调用本地大脑...")
            brain_result = await process_with_scene(recognized_text, session_id)
            reply_text = brain_result.get("reply", "指令处理失败")
            print(f"[SmartBrain] 回复成功: {reply_text[:50]}...")
        else:
            # 本地模式：直接用本地 NLU
            intents = parse_multi_intent(recognized_text, session_id)
            if intents:
                reply_text = await batch_execute_intents(intents, session_id, recognized_text)
            else:
                context_reply = await handle_context_clarification(recognized_text, session_id)
                if context_reply:
                    reply_text = context_reply
                else:
                    reply_text = "抱歉，无法识别您的指令。"

        # —— ★【TTS 防空闸门 第二层：voice_tts】调用 _tts_file_response 前再兜一次 ——
        if not isinstance(reply_text, str) or not reply_text.strip():
            reply_text = "操作已完成。"
            print(f"[Reply-Safe] voice_tts: reply_text 为空，强制兜底 → {reply_text!r}")

        print(f"[TTS] 合成文本: {reply_text}")

        # C. TTS + FileResponse via shared helper
        # (tts_path not needed anymore — _tts_file_response manages its own temp files)
        return await _tts_file_response(reply_text, voice=voice, asr_text=recognized_text)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务错误: {str(e)}")
    finally:
        for p in [raw_path, converted_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# ============ 共享 TTS 响应辅助函数 ============
# TTS 临时音频目录：不要放在项目根目录（否则批量清理 match "tts_*" 时容易误伤源码）
TTS_TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_audio")


async def _tts_file_response(
    reply_text: str,
    voice: str = "xiaoxiao",
    asr_text: str = "",
) -> FileResponse:
    """
    一站式 TTS 合成 + 文件返回 + 元数据封装。

    减少 text_chat / voice_tts 等端点的重复代码。
    自动处理：文本清洗 → 合成 → stale 文件清理 → 元数据编码 → FileResponse。

    ★【TTS 防空闸门 第一层】：进入合成引擎前强制兜底，彻底杜绝空文本触发 TTS 崩溃。
    """
    if not isinstance(reply_text, str) or not reply_text.strip():
        reply_text = "操作已完成。"
        print(f"[TTS-SAFEGUARD] _tts_file_response 收到空文本，强制兜底 → {reply_text!r}")

    # Clean + synthesize via the new helper
    resolved_voice = CHINESE_VOICES.get(voice) if voice in CHINESE_VOICES else voice
    result = await synthesize_to_response(
        reply_text,
        voice=resolved_voice,
        tts_dir=TTS_TEMP_DIR,
    )
    tts_path = result["file_path"]
    cleaned_text = result["cleaned_text"]
    is_empty_tts = bool(result.get("empty"))

    # URL-encode metadata for HTTP header compatibility
    meta = urllib.parse.urlencode(
        {"asr": asr_text, "reply": cleaned_text},
        quote_via=urllib.parse.quote,
    )
    snapshot = _normalize_device_states_snapshot()
    state_snapshot = urllib.parse.urlencode(
        {"states": json.dumps(snapshot, ensure_ascii=False)},
        quote_via=urllib.parse.quote,
    )

    # X-TTS-Empty: 前端识别到=1且size==0时，应静默跳过播放（不再把"音频文件为空"当异常弹窗）
    extra_headers = {
        "X-Voice-Meta": meta,
        "X-Device-States": state_snapshot,
    }
    if is_empty_tts:
        extra_headers["X-TTS-Empty"] = "1"

    return FileResponse(
        tts_path,
        media_type="audio/mpeg",
        filename=f"reply_{uuid.uuid4().hex[:12]}.mp3",
        headers=extra_headers,
    )


# ============ 文字对话接口 ============

@app.post("/api/v1/text_chat")
async def text_chat(
    text: str = Query(..., description="用户输入的文字指令"),
    mode: str = Query(default="smart", description="smart / local"),
    voice: str = Query(default="xiaoxiao"),
    session_id: str = Query(default="default"),
):
    """
    文字对话：直接传入文字 → (本地NLU/Dify) → 文本回复 → TTS → 返回音频

    多轮会话处理顺序（严格遵守需求 2 & 3）：
      1. 先查 pending_intent（handle_context_clarification）
         a. 澄清回复（卧室/客厅）→ 合并槽位 execute_command → 播报最终结果
         b. 用户改口（非房间/设备名）→ 清 pending，__NEW_CMD__ 哨兵 → 按新指令重新走 SmartBrain
         c. 无 pending → None，正常走流程
      2. 无 pending 或 已改口 → 调用 SmartBrain 处理新指令
         - SmartBrain 内部会在设备歧义时 save_pending 并反问（需求 1）
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="文字内容不能为空")

    # ★ main.py 最外层：ASR 同音校正（文本接口也走一次，便于直接用模拟错别字测试）
    corrected = asr_phonetic_correct(text)
    if corrected != text:
        print(f"[TextChat] ★ 拼音校正: '{text}' → '{corrected}'")
    text = corrected

    task_id = str(uuid.uuid4())
    tts_path = f"temp_tts_{task_id}.mp3"

    try:
        reply_text = ""
        if mode in ("smart", "dify"):
            # ---- Step 1: 优先处理多轮槽位补全（需求 2 & 3） ----
            ctx_result = await handle_context_clarification(text, session_id)
            if ctx_result is not None and ctx_result != "__NEW_CMD__":
                # 情况 1-a：已成功完成槽位补全 + 设备控制，直接播报
                reply_text = ctx_result
                print(f"[TextChat][槽位补全] 输入: {text} → 回复: {reply_text}")
                # —— ★【TTS 防空闸门 第三层：text_chat #1】 ——
                if not isinstance(reply_text, str) or not reply_text.strip():
                    reply_text = "操作已完成。"
                    print(f"[Reply-Safe] text_chat(槽位补全): reply_text 为空，强制兜底 → {reply_text!r}")
                return await _tts_file_response(reply_text, voice=voice, asr_text=text)
            # ctx_result == "__NEW_CMD__" → 情况 1-b：用户改口，清完 pending 后按新指令重处理

            # ---- Step 2: 无 pending / 已改口 → SmartBrain 处理新指令 ----
            from smart_brain import process_with_scene
            print(f"[SmartBrain] 调用本地大脑... (context={'NEW_CMD' if ctx_result == '__NEW_CMD__' else 'no_pending'})")
            brain_result = await process_with_scene(text, session_id)
            reply_text = brain_result.get("reply", "指令处理失败")

            # （可选）歧义时把 pending_intent slot 信息打到日志，前端/调试用
            if brain_result.get("need_clarification"):
                print(f"[TextBrain][歧义反问] pending_intent={brain_result.get('pending_intent')}")

            print(f"[SmartBrain] 回复成功: {reply_text[:60]}...")
            print(f"[TextChat] 输入: {text} → 回复: {reply_text}")

            # —— ★【TTS 防空闸门 第三层：text_chat #2】 ——
            if not isinstance(reply_text, str) or not reply_text.strip():
                reply_text = "操作已完成。"
                print(f"[Reply-Safe] text_chat(SmartBrain): reply_text 为空，强制兜底 → {reply_text!r}")
            # TTS + FileResponse via shared helper
            return await _tts_file_response(reply_text, voice=voice, asr_text=text)
        else:
            # 本地模式：直接用本地 NLU
            ctx_result = await handle_context_clarification(text, session_id)
            if ctx_result is not None and ctx_result != "__NEW_CMD__":
                reply_text = ctx_result
            else:
                intents = parse_multi_intent(text, session_id)
                if intents:
                    reply_text = await batch_execute_intents(intents, session_id, text)
                else:
                    reply_text = "抱歉，无法识别您的指令。"

        print(f"[TextChat] 输入: {text} → 回复: {reply_text}")

        # —— ★【TTS 防空闸门 第三层：text_chat #3】 ——
        if not isinstance(reply_text, str) or not reply_text.strip():
            reply_text = "操作已完成。"
            print(f"[Reply-Safe] text_chat(本地模式): reply_text 为空，强制兜底 → {reply_text!r}")

        # TTS + FileResponse via shared helper
        return await _tts_file_response(reply_text, voice=voice, asr_text=text)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务错误: {str(e)}")


@app.get("/api/v1/tts_voices")
async def get_tts_voices():
    """获取可用的中文语音列表"""
    return {"voices": list_voices()}


# ============ 前端静态文件 ============

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    """前端首页"""
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
