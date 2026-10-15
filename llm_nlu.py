# =================================================================
# 智能家居 NLU 解析 - 本地轻量版（Qwen2.5-1.5B-Instruct + LoRA）
#
# 目标：彻底替换原有的在线 SiliconFlow API 调用，解决 API 欠费 402 的问题。
# 加载方式：transformers.AutoTokenizer/AutoModelForCausalLM + peft.PeftModel
# 精度/设备：torch_dtype=torch.float32, device_map="cpu"（用户指定；有 GPU 可改 "auto"）
#
# 对外接口保持 100% 兼容：parse_intent_llm(text) -> dict，
# 字段契约与 nlu_parser.parse_intent() 完全对齐（main.py 的 hybrid_parse_intent 直接调用）。
# =================================================================

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, Optional

# ★ 用户 spec 要求：最上方设置 HF 镜像，确保国内能拉到 Qwen 权重
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows symlinks 告警静音

import torch  # noqa: E402  （os.environ 设完后再 import torch/HF，顺序很重要）

from nlu_parser import (  # noqa: E402
    DEVICE_MAP,
    ACTION_PATTERNS,
    VALUE_PATTERNS,
    get_device_risk,
    asr_phonetic_correct,
    detect_explicit_room,
    resolve_device_id,
    ROOM_CN_TO_SLUG,
    parse_intent as rule_parse_intent,
    preprocess_voice_text,
    is_effective_empty,
)

# ==============================================================
# 用户 spec 要求的"短 system prompt"（优先用）
# ==============================================================
SYSTEM_PROMPT_SHORT = (
    "你是一个智能家居 NLU 解析助手，请将用户的控制指令解析为对应的意图和槽位 JSON。"
)

# ==============================================================
# SFT 训练时用的"长 system prompt"（和 finetune dataset 对齐，
# 本地模型如果短 prompt 解析效果差，把 USE_SHORT_SYSTEM_PROMPT 改成 False 即可）
# ==============================================================
_DEVICE_TABLE = "\n".join(f'  - "{raw}" → "{dev_id}"' for raw, dev_id in DEVICE_MAP)
_ACTION_TABLE = "\n".join(f'  - 匹配 "{p[1:-1]}..." → "{act}"' for p, act in ACTION_PATTERNS)

SYSTEM_PROMPT_SFT = f"""你是一个智能家居语音助手的意图解析器。你的任务是将用户的自然语言指令解析为严格的 JSON 格式。

## ★ 重要：ASR 同音错别字容错（Phonetic Correction）
用户输入文本来自 ASR 语音识别，可能包含同音错别字（如：空条=空调、窗连=窗帘、克厅=客厅、卧试=卧室、关毕=关闭、大开=打开、客听=客厅、卧世=卧室、书放=书房、餐听=餐厅、问度=温度、分速=风速）。
你在提取 JSON 实体前，请先【自动将同音错别字映射为系统内合法的设备名、房间名与动作】：
   1) 把"客厅/克厅/客听/大厅/起居室"统一识别为 location=客厅，device 选择 living_room_*
   2) 把"卧室/卧试/卧世/卧是/主卧/次卧"统一识别为 location=卧室，device 选择 bedroom_*
   3) 把"空调/空条/关调"统一识别为设备类型 ac（bedroom_ac / living_room_ac）
   4) 把"窗帘/窗连/创联/传帘"统一识别为设备类型 curtain（bedroom_curtain / living_room_curtain）
   5) 把"电视/电世/电试/电是"统一识别为设备类型 tv
   6) 把"打开/大开/达开/搭开/开启/开起"识别为 action=on，把"关闭/关毕/官闭/关掉"识别为 action=off，把"调到/调到X度/设置为"识别为 action=set（X 度 → value_type=temperature）。
★ 核心规则：当用户文本中已经出现"客厅/卧室"或任何同音变体时，必须【直接输出具体 device_id】（如客厅窗帘 → living_room_curtain，卧室空调 → bedroom_ac），绝不能输出通用 device_id（curtain / air_conditioner / light / tv）。
★ 绝对禁止因为 1~2 字同音错别字就返回 success=false；用拼音相似度选择候选设备再输出。

## 已知设备映射表
{_DEVICE_TABLE}

## 动作映射规则
从用户文本中识别以下动作之一：
{_ACTION_TABLE}
  - 如果没有匹配，action 为 "unknown"

## 数值提取
- 温度："26度" → value=26, value_type="temperature"
- 亮度百分比："80%" / "最亮" / "最暗" → value=80/100/10, value_type="brightness"
- 窗帘状态："全开" / "全关" → value="全开"/"全关", value_type="curtain_state"

## 输出格式（严格 JSON，无其他文字）
{{
  "success": true/false,
  "msg": "解析成功" 或 错误描述,
  "device_id": "设备ID",
  "device_name": "原始设备名称",
  "action": "on/off/increase/decrease/set/set_temperature/set_brightness/open_curtain/close_curtain/unknown",
  "value": 数值或null,
  "value_type": "temperature/brightness/curtain_state/fan_mode/hvac_mode/channel/null",
  "attrs": {{"mode": "cool/heat/..."}} 或 {{}},
  "confidence": 0.0-1.0,
  "need_clarification": true/false
}}

## 规则
1. 只输出 JSON，不要任何解释或 markdown。
2. 如果无法识别设备或动作，success=false，其余字段尽可能填写。
3. device_id 必须严格使用映射表中的英文 ID，不要自己编造。
4. 对于环境感知指令（如"客厅有点热"），必须将模式信息放入 attrs.mode 字段。
"""

# ★ 开关：默认用用户 spec 要求的短 prompt；如果解析不准，把这里切到 SYSTEM_PROMPT_SFT
USE_SHORT_SYSTEM_PROMPT = True
SYSTEM_PROMPT = SYSTEM_PROMPT_SHORT if USE_SHORT_SYSTEM_PROMPT else SYSTEM_PROMPT_SFT


# ==============================================================
# 模型加载（懒加载 + 线程安全单例）—— 第一次调用 parse_intent_llm 才加载，
# 避免 import 模块就等 30s+；torch.load / peft 都不会重复。
# ==============================================================
BASE_MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_ADAPTER_PATH = r"G:\models\checkpoint-400"
TORCH_DTYPE = torch.float32
DEVICE_MAP_STR = "cpu"        # 用户 spec：device_map 参数（避免和 nlu_parser.DEVICE_MAP 同名冲突）；有 GPU 想提速改成 "auto"

_model_load_lock = threading.Lock()
_tokenizer: Any = None        # transformers.PreTrainedTokenizer
_model: Any = None            # PeftModel 包装后的 AutoModelForCausalLM
_load_error: Optional[str] = None


def _ensure_model_loaded() -> Optional[str]:
    """返回 None 表示 OK，否则返回错误原因字符串。"""
    global _tokenizer, _model, _load_error
    if _tokenizer is not None and _model is not None:
        return None
    with _model_load_lock:
        if _tokenizer is not None and _model is not None:
            return None
        if _load_error is not None:
            return _load_error
        try:
            print(f"[Local-NLU] 正在加载基座模型与 {LORA_ADAPTER_PATH} 权重...", flush=True)
            t0 = time.perf_counter()
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            _tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_NAME_OR_PATH,
                use_fast=False,
                trust_remote_code=True,
            )
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
                _tokenizer.padding_side = "left"  # batch-less generation 用 left pad 更稳

            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME_OR_PATH,
                torch_dtype=TORCH_DTYPE,
                device_map=DEVICE_MAP_STR,
                trust_remote_code=True,
            )
            _model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
            _model.eval()

            dt = time.perf_counter() - t0
            print(f"[Local-NLU] 加载完成，耗时 {dt:.1f}s", flush=True)
            return None
        except Exception as e:
            _load_error = f"模型加载失败: {type(e).__name__}: {e}"
            print(f"[Local-NLU] ❌ {_load_error}", flush=True)
            return _load_error


# ==============================================================
# 本地生成：apply_chat_template + max_new_tokens=128, temperature=0.1
# ==============================================================
def _call_local_llm(text: str) -> Optional[str]:
    err = _ensure_model_loaded()
    if err is not None:
        return None
    assert _tokenizer is not None and _model is not None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        prompt_ids = _tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # apply_chat_template 在旧版 transformers + 部分 tokenizer 下返回 BatchEncoding
        # (shape=() 即 0-dim)。这里统一拿 "input_ids" 字段并保证是 [1, seq_len] tensor。
        if isinstance(prompt_ids, dict) or hasattr(prompt_ids, "get"):
            prompt_ids = prompt_ids.get("input_ids")
        import torch as _T
        if not isinstance(prompt_ids, _T.Tensor):
            prompt_ids = _T.tensor(prompt_ids)
        if prompt_ids.dim() == 0:
            prompt_ids = prompt_ids.reshape(1, -1)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        # 按用户 spec: max_new_tokens=128, temperature=0.1
        # 注意：temperature!=0 时必须 do_sample=True
        with torch.no_grad():
            gen_ids = _model.generate(
                input_ids=prompt_ids.to(_model.device),
                max_new_tokens=128,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=_tokenizer.pad_token_id,
                eos_token_id=_tokenizer.eos_token_id,
            )
        # 只取新生成的部分（去掉 prompt）
        new_ids = gen_ids[:, prompt_ids.shape[-1]:]
        content = _tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
        return content or None
    except Exception as e:
        print(f"[Local-NLU] 生成失败: {type(e).__name__}: {e}", flush=True)
        return None


# ==============================================================
# 输出解析 + 旧容错格式 + 归一化为 main.py 所需的包装 schema
# ==============================================================
def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    t = m.group()
    try:
        return json.loads(t)
    except json.JSONDecodeError as _e0:
        # ================================================================
        # lenient 修复：基于 tokenizer 的状态机重建 JSON 文本
        # 能处理：未加引号的 key / key 后缺冒号 / value 后缺逗号 / 尾逗号 / 缺 }
        # ================================================================
        fixed = t
        open_b, close_b = fixed.count("{"), fixed.count("}")
        if open_b > close_b:
            fixed += "}" * (open_b - close_b)

        TOK_RE = re.compile(
            r'"(?:[^"\\]|\\.)*"'          # 1. 双引号字符串（含转义）
            r"|true|false|null"            # 2. 常量
            r"|-?\d+(?:\.\d+)?"            # 3. 数字
            r"|[{}\[\],:]"                 # 4. 结构符
            r"|[A-Za-z_\u4e00-\u9fff][\w\u4e00-\u9fff\-]*"  # 5. 未加引号 key / 中文设备名
        )
        tokens = [x.group() for x in TOK_RE.finditer(fixed)]
        if not tokens:
            return None

        out: list = []
        n, i = len(tokens), 0
        depth = 0
        IN_OBJ, IN_ARR, EXPECT_KEY, EXPECT_COLON, EXPECT_VALUE, EXPECT_SEP = range(6)
        state_stack = [EXPECT_KEY]  # 栈顶 = 当前对象/数组的"下一个期望"

        def cur_state():
            return state_stack[-1]

        while i < n:
            tok = tokens[i]
            st = cur_state()

            # --- 结构符 ---
            if tok == "{":
                if st in (EXPECT_VALUE, EXPECT_SEP) and out and out[-1] not in ("{", "[", ":", ","):
                    # 自动补逗号（上一个值直接跟新对象）
                    out.append(",")
                out.append("{")
                state_stack.append(EXPECT_KEY)
                depth += 1
                i += 1
                continue
            if tok == "}":
                if out and out[-1] == ",":
                    out.pop()
                out.append("}")
                # pop 对应的 EXPECT_KEY / EXPECT_SEP
                for _ in range(max(1, len(state_stack))):
                    if state_stack and state_stack[-1] in (EXPECT_KEY, EXPECT_SEP):
                        state_stack.pop()
                        break
                depth -= 1
                # 退出 obj 后，父级进入 EXPECT_SEP
                if state_stack and state_stack[-1] == EXPECT_VALUE:
                    state_stack[-1] = EXPECT_SEP
                i += 1
                continue
            if tok == "[":
                if st in (EXPECT_VALUE, EXPECT_SEP) and out and out[-1] not in ("{", "[", ":", ","):
                    out.append(",")
                out.append("[")
                state_stack.append(EXPECT_VALUE)
                i += 1
                continue
            if tok == "]":
                if out and out[-1] == ",":
                    out.pop()
                out.append("]")
                for _ in range(max(1, len(state_stack))):
                    if state_stack and state_stack[-1] in (EXPECT_VALUE, EXPECT_SEP):
                        state_stack.pop()
                        break
                if state_stack and state_stack[-1] == EXPECT_VALUE:
                    state_stack[-1] = EXPECT_SEP
                i += 1
                continue
            if tok == ",":
                # 期望 value 或 sep 时遇到 , -> 忽略 / 或作为 sep
                if out and out[-1] in ("{", "[", ":", ","):
                    i += 1
                    continue
                out.append(",")
                # 翻到父级状态
                for _ in range(max(1, len(state_stack))):
                    top = state_stack[-1]
                    if top == EXPECT_SEP:
                        state_stack[-1] = EXPECT_KEY if out.count("{") > out.count("}") else EXPECT_VALUE
                        break
                    elif top in (EXPECT_KEY, EXPECT_VALUE):
                        break
                    else:
                        state_stack.pop()
                # 简化：根据 depth（未闭合的 {}）决定下一个期待
                unclosed_obj = (sum(1 for x in out if x == "{") - sum(1 for x in out if x == "}"))
                unclosed_arr = (sum(1 for x in out if x == "[") - sum(1 for x in out if x == "]"))
                if unclosed_obj > 0 and unclosed_arr == 0:
                    state_stack[-1] = EXPECT_KEY
                elif unclosed_arr > 0 and unclosed_obj == 0:
                    state_stack[-1] = EXPECT_VALUE
                i += 1
                continue
            if tok == ":":
                if st == EXPECT_COLON:
                    out.append(":")
                    state_stack[-1] = EXPECT_VALUE
                    i += 1
                    continue
                # 意外冒号：丢
                i += 1
                continue

            # --- 值/key token ---
            if st == EXPECT_KEY:
                # 必须是字符串或裸标识符 -> 双引号 key
                if tok.startswith('"'):
                    # 特殊：模型把 key+value 拼成一个字符串，例如 "need_clarification false"
                    inner = tok[1:-1].strip()
                    mm = re.match(r"^([A-Za-z_][\w]*)\s+(true|false|null|-?\d+(?:\.\d+)?)$", inner)
                    if mm:
                        key_tok = '"' + mm.group(1) + '"'
                        val_tok = mm.group(2)
                        out.append(key_tok); out.append(":"); out.append(val_tok)
                        # value 完 -> 期待 , / }
                        state_stack[-1] = EXPECT_SEP
                        i += 1
                        continue
                    key = tok
                else:
                    # 特殊：裸的 "need_clarification false" （没引号但内部有空格已经被切分成两个 token 了 -> 此分支不会命中）
                    key = '"' + tok + '"'
                out.append(key)
                # next should be ':'
                j = i + 1
                nxt = tokens[j] if j < n else None
                if nxt == ":":
                    out.append(":")
                    state_stack[-1] = EXPECT_VALUE
                    i = j + 1
                else:
                    out.append(":")
                    state_stack[-1] = EXPECT_VALUE
                    i += 1
                continue

            if st in (EXPECT_VALUE, EXPECT_SEP):
                if st == EXPECT_SEP and out and out[-1] not in ("{", "[", ":", ","):
                    out.append(",")
                    # 刚加完逗号，如果是在对象里 -> 期望 key；数组里 -> 期望 value
                    unclosed_obj = (sum(1 for x in out if x == "{") - sum(1 for x in out if x == "}"))
                    unclosed_arr = (sum(1 for x in out if x == "[") - sum(1 for x in out if x == "]"))
                    if unclosed_obj > 0 and (unclosed_arr == 0 or unclosed_obj >= unclosed_arr):
                        state_stack[-1] = EXPECT_KEY
                    else:
                        state_stack[-1] = EXPECT_VALUE
                    # 重新 loop 同一个 tok（现在 EXPECT_KEY/VALUE）
                    continue
                # EXPECT_VALUE: 直接把 tok 当值
                out.append(tok)
                state_stack[-1] = EXPECT_SEP
                i += 1
                continue

            # 任何其他状态：跳过
            i += 1

        # 最后一层尾逗号/未闭合
        recon = "".join(out)
        # 补齐未闭合的 {} 和 []
        opens_obj = recon.count("{") - recon.count("}")
        opens_arr = recon.count("[") - recon.count("]")
        if opens_obj > 0: recon += "}" * opens_obj
        if opens_arr > 0: recon += "]" * opens_arr
        try:
            return json.loads(recon)
        except json.JSONDecodeError:
            return None
    return None


def _fallback_to_rule(text: str, msg: str) -> Dict[str, Any]:
    """本地 LLM 任何异常都 fallback 到 rule_parse，保持返回 schema 不变，只是 method=llm 让 main.py 能区分"""
    rule = rule_parse_intent(text)
    intent = rule.get("intent")
    if isinstance(intent, dict):
        intent = dict(intent)
        intent.pop("raw_text", None)
        intent.pop("control_mode", None)
        intent.pop("delta", None)
    return {
        "success": bool(rule.get("success") and intent is not None),
        "msg": msg or rule.get("msg", ""),
        "method": "llm",
        "confidence": float(rule.get("confidence", 0.0) or 0.0),
        "need_clarification": bool(rule.get("need_clarification", False)),
        "intent": intent,
    }


def _parse_output_and_wrap(raw: Optional[str], original_text: str) -> Dict[str, Any]:
    """把本地 LLM 输出 → parse_intent_llm 对外契约的 dict。"""
    if raw is None:
        return _fallback_to_rule(original_text, "本地模型生成失败，已降级为 rule 解析")

    data = _extract_json(raw)
    if data is None:
        return _fallback_to_rule(
            original_text,
            f"模型输出无法解析为 JSON，前200字符: {raw[:200]}",
        )

    # 字段容错：兼容 LLM 多种命名风格
    device_id = str(data.get("device_id") or data.get("device") or data.get("slot", {}).get("ac") or "")
    device_name = str(data.get("device_name") or device_id)
    slot_action = str((data.get("slot") or {}).get("action") or "")
    action = str(data.get("action") or slot_action or "")
    # 动作归一化：把中文/英文别名（用户 slot 里的 open/close，或者中文"打开"/"关闭"）统一成 rule 体系的 on/off/increase/decrease/set
    ACTION_NORM = {
        "open": "on", "打开": "on", "开启": "on", "开": "on", "on": "on",
        "close": "off", "关闭": "off", "关": "off", "off": "off", "关掉": "off", "熄灭": "off",
        "调高": "increase", "升高": "increase", "increase": "increase",
        "调低": "decrease", "降低": "decrease", "decrease": "decrease",
        "调": "set", "设置": "set", "设置为": "set", "set": "set",
    }
    if not action:
        intent_str = str(data.get("intent") or "")
        # 兼容 test 截图里的 {"intent":"control_ac","slot":{"ac":"卧室空调","action":"open"},...} 老 schema
        action = ACTION_NORM.get(intent_str, intent_str) or "unknown"
    # 如果 slot.action 是独立字段（比如 open），但 action 没被归一化，再做一次归一
    if action in ACTION_NORM:
        action = ACTION_NORM[action]

    value = data.get("value")
    if value is None and isinstance(data.get("attrs"), dict):
        value = data["attrs"].get("temperature") or data["attrs"].get("brightness") or data["attrs"].get("channel")
    if value is None and isinstance(data.get("slot"), dict):
        slot = data["slot"]
        for k in ("temperature", "brightness", "channel", "target_temperature", "target_brightness"):
            if slot.get(k) is not None and value is None:
                value = slot[k]
        # slot.target=open/close 也可当作 action（见 test 截图 control_ac slot）
        slot_action_norm = {
            "open": "on", "打开": "on", "on": "on", "close": "off", "关闭": "off", "off": "off",
        }
        if slot.get("target") and str(slot["target"]) in slot_action_norm:
            # action 比 intent 字符串（如 set_ac / control_ac）更靠谱，有就覆盖
            action = slot_action_norm[str(slot["target"])]
    if value is not None and isinstance(value, str):
        # 温度 value="26" 这种字符串 int 化；浮点也兼容
        try:
            if re.fullmatch(r"-?\d+", value):
                value = int(value)
            elif re.fullmatch(r"-?\d+\.\d+", value):
                value = float(value)
        except Exception:
            pass
    # attrs.temperature / slot.temperature 也转成 int
    if isinstance(value, int) or isinstance(value, float):
        pass

    # 归一化 intent 语义字符串（比如 set_ac / control_light 这种）为 action
    #    只有当 action 还不是 on/off/set/... 这种 rule 规范 action 时，才覆盖。
    FINAL_ACTIONS = {"on", "off", "toggle", "increase", "decrease", "set",
                     "set_temperature", "set_mode", "set_brightness", "set_color",
                     "set_channel", "set_volume", "open_curtain", "close_curtain"}
    if action not in FINAL_ACTIONS:
        raw_intent = str(data.get("intent") or "")
        if raw_intent in ACTION_NORM:
            action = ACTION_NORM[raw_intent]
        elif raw_intent == "toggle":
            action = "toggle"
        elif raw_intent.startswith("set_") or raw_intent.startswith("control_"):
            # set_ac / control_light / ... 这种泛意图：根据槽位反推
            if isinstance(data.get("slot"), dict):
                sl = data["slot"]
                if any(k in sl for k in ("mode",)):
                    action = "set_mode"
                elif any(k in sl for k in ("temperature",)):
                    action = "set_temperature" if not value else "set"
                elif any(k in sl for k in ("brightness",)):
                    action = "set_brightness"
                else:
                    # slot 有 target=open/close 时上面已经覆盖过 action，这里兜底按 value 有无
                    action = "set" if value is not None else "on"
            elif value is not None:
                action = "set"
            else:
                action = "on"


    value_type = str(data.get("value_type") or "")
    confidence = float(data.get("confidence", 0.8))
    need_clarification = bool(data.get("need_clarification", False))

    # 设备 ID 归一化：如果模型输出了"卧室空调"/"客厅窗帘"之类的中文名，翻译成英文 ID
    RAW_TO_ID = {raw: dev for raw, dev in DEVICE_MAP}
    if device_id in RAW_TO_ID:
        device_id = RAW_TO_ID[device_id]
    known_ids = [d[1] for d in DEVICE_MAP]
    action_ok = bool(action) and action != "unknown"
    dev_ok = bool(device_id) and device_id in known_ids
    success = bool(data.get("success", dev_ok and action_ok)) and dev_ok and action_ok
    if not success and (not dev_ok or not action_ok):
        return _fallback_to_rule(
            original_text,
            f"模型输出不合法(device_id={device_id!r},action={action!r})，降级为 rule",
        )

    msg = str(data.get("msg") or "LLM 解析成功（本地）")

    # 业务层强制：高风险设备（窗帘）没明确房间 → 必须 need_clarification
    curtain_ids = ("curtain", "bedroom_curtain", "living_room_curtain")
    if device_id in curtain_ids and device_id == "curtain":
        need_clarification = True
        device_name = device_name or "窗帘"

    attrs = data.get("attrs", {}) or {}
    if not isinstance(attrs, dict):
        attrs = {}
    # 把 "slot": {"temperature": 26} 兼容成 attrs.temperature（用户截图里的老 schema）
    if isinstance(data.get("slot"), dict):
        slot = data["slot"]
        if "temperature" in slot and "temperature" not in attrs:
            attrs["temperature"] = slot["temperature"]

    intent = {
        "device_id": device_id,
        "device_name": device_name,
        "action": action,
        "value": value,
        "value_type": value_type,
        "attrs": attrs,
    }

    return {
        "success": success,
        "msg": msg,
        "method": "llm",
        "confidence": confidence,
        "need_clarification": need_clarification,
        "intent": intent,
    }


# ==============================================================
# 对外入口：parse_intent_llm(text) （签名 / 返回字段契约保持 100% 不变）
# ==============================================================
def parse_intent_llm(text: str) -> Dict[str, Any]:
    """
    本地 LoRA 推理版 parse_intent_llm。
    返回 schema 与 nlu_parser.parse_intent() / 原在线版完全一致：
        {
            "success": bool,
            "msg": str,
            "method": "llm",
            "confidence": float,
            "need_clarification": bool,
            "intent": {device_id, device_name, action, value, value_type, attrs} | None
        }
    """
    # 0) 先统一走预处理管道（洗标点/剥填充词/热词/同音）
    text = preprocess_voice_text(text)

    # 0.1) 空指令早退：避免误触/背景噪音触发 40 秒模型加载 + 推理
    if is_effective_empty(text):
        return {
            "success": False,
            "msg": "未检测到有效语音指令，请重新表达",
            "method": "llm",
            "confidence": 0.0,
            "need_clarification": True,
            "intent": None,
        }

    # 1) ASR 同音纠错（这里再跑一次 asr_phonetic_correct 成本可忽略，保持历史行为不变）
    text = asr_phonetic_correct(text)

    # 2) 跑本地 LLM（模型内部懒加载）
    raw = _call_local_llm(text)
    parsed = _parse_output_and_wrap(raw, text)
    parsed["method"] = "llm"

    # 3) 后端最后一道业务修正：明确房间 → 锁定 device_id + need_clarification=false
    explicit_room = detect_explicit_room(text, asr_corrected=True)
    if explicit_room and isinstance(parsed.get("intent"), dict):
        dev_id = parsed["intent"].get("device_id", "") or ""
        fixed = resolve_device_id(dev_id, explicit_room=explicit_room)
        if fixed != dev_id:
            print(f"[Local-NLU] 房间锁定: text='{text}' room={explicit_room} "
                  f"device_id {dev_id!r} → {fixed!r}")
            parsed["intent"]["device_id"] = fixed
            room_cn = {v: k for k, v in ROOM_CN_TO_SLUG.items()}.get(explicit_room, "")
            suffix_to_type = {"ac": "空调", "light": "灯", "curtain": "窗帘", "tv": "电视"}
            tail = fixed.rsplit("_", 1)[-1]
            type_cn = suffix_to_type.get(tail, "")
            if room_cn and type_cn:
                parsed["intent"]["device_name"] = room_cn + type_cn
            parsed["intent"]["_explicit_room_slug"] = explicit_room
            parsed["need_clarification"] = False

    return parsed


# ==============================================================
# 兼容保留：老代码如果直接调用 _call_llm / _parse_llm_output / _fallback_no_parse
# 仍能工作（全部代理到新实现，保持签名大致一致）
# ==============================================================
def _call_llm(text: str) -> Optional[str]:
    return _call_local_llm(text)


def _parse_llm_output(raw: str, original_text: str) -> Dict[str, Any]:
    return _parse_output_and_wrap(raw, original_text)


def _fallback_no_parse(raw: str, original_text: str) -> Dict[str, Any]:
    return _fallback_to_rule(original_text, f"LLM 输出无法解析，前200字符: {raw[:200]}")
