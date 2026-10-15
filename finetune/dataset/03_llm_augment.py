"""
【微调 · 阶段 A-3】LLM 自然口语扩写（可选，edge case 覆盖）

用途：
  - 阶段 A-1/A-2 做出来的样本"结构化模板味"还是偏重；
  - 真实语音里用户会用很多 LLM 才想得出来的自然表达（例如"我老婆嫌冷"、"要睡觉了把灯关了"、"给娃念书调暗点"…）。
  - 这个脚本用 SiliconFlow（和你现在 main.py 里用的同一个 key/base_url）批量合成 200~300 条"自然化 edge case"。

用法：
    Agent_env\Scripts\python.exe finetune/dataset/03_llm_augment.py --count 300
或
    Agent_env\Scripts\python.exe finetune/dataset/03_llm_augment.py --dry  # 只打印提示词模板，不实际调 API（省 token）

输出：
    finetune/outputs/llm_augment_samples.json  # 和 01/02 同 schema，可直接被 04_build_jsonl 合并
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    SILICONFLOW_MODEL,
    LLM_TIMEOUT,
    LLM_MAX_RETRIES,
)
from llm_nlu import SYSTEM_PROMPT

random.seed(44)

REQUESTED_CAPABILITIES = [
    "ASR 同音错别字（卧试空调/克厅灯/窗连/空条/关毕/大开…）",
    "环境感受式模糊指令（客厅有点热/卧室太黑/晃眼/闷）",
    "多意图复合（打开客厅空调调到26度、卧室灯和客厅窗帘都关了）",
    "同一设备多参数（空调26度、风速3档、扫风打开）",
    "礼貌口语/啰嗦表达（麻烦你帮我一下、可以吗、谢谢、拜托）",
    "语序倒装 / 方言口语（客厅空条开下嘛、空调给我搞到二十六度撒）",
    "数字口语化（二十六度、一档风、半开窗帘）",
    "情境指令（要睡觉了把卧室灯关了、吃饭了客厅开亮点、我走了把所有灯关一下）",
    "间接请求（好刺眼把窗帘拉上、有点闷空调开下）",
    "指代式（'那个'/'刚才那个'/'客厅那个' 这种代词 → 强制用设备全名代替，不要真的留代词）",
]

# 为了让 LLM 输出"可 parse"的 JSON，我们给它几个 1-shot 正例作为锚点
FEW_SHOT = [
    {
        "scenario": "ASR 同音错词 + 多参数",
        "input": "克厅空条开一下，二十六度，制冷，分速二档",
        "output_json": {
            "device_id": "living_room_ac", "action": "on",
            "value": 26, "value_type": "temperature",
            "attrs": {"temperature": 26, "mode": "cooling", "fan_speed": 2},
        },
    },
    {
        "scenario": "环境感受 + 卧室",
        "input": "唉，小孩房间太黑，你给把卧试的灯大开一下好不好？",
        "output_json": {
            "device_id": "bedroom_light", "action": "on",
            "value": None, "value_type": None, "attrs": {},
        },
    },
]


def build_user_prompt(n_requested: int) -> str:
    caps = "\n".join(f"  - {c}" for c in REQUESTED_CAPABILITIES)
    shots = "\n\n".join(
        f"## 示例 {i+1}（{ex['scenario']}）\n"
        f"Input: {ex['input']}\n"
        f"Output: {json.dumps(ex['output_json'], ensure_ascii=False)}"
        for i, ex in enumerate(FEW_SHOT)
    )
    return f"""你是"智能家居 NLU 数据集合成专家"。请按照以下要求，生成 {n_requested} 条 **diverse（多样）** 的 NLU 训练样本。

## 能力覆盖（每条至少命中 1~2 项，尽量跨能力覆盖）
{caps}

## 输出格式（必须严格）
输出一个 JSON array，每个元素形如：
{{
  "input": "用户给语音助手说的中文口语文本（带错字/自然表达）",
  "output_json": {{
    "success": true,
    "msg": "解析成功",
    "device_id": "living_room_ac / bedroom_ac / living_room_light / bedroom_light / living_room_curtain / bedroom_curtain / living_room_tv / bedroom_tv 之一【必须是带房间的完整 device_id，禁止输出通用 curtain/ac/light/tv】",
    "device_name": "客厅空调 / 卧室灯 这种中文显示名",
    "action": "on/off/set/set_temperature/increase/decrease 之一",
    "value": 数值或字符串或 null（例如 26、"制冷"、"全开"）,
    "value_type": "temperature/brightness/hvac_mode/fan_mode/swing_mode/curtain_state/channel/null",
    "attrs": {{"temperature":26, "mode":"cooling", "fan_speed":2, "brightness":80, "swing":"off", "state":"全开", "channel": 5}} 或 {{}},
    "confidence": 1.0,
    "need_clarification": false
  }}
}}

## 示例（作为格式锚点，你生成的内容不能和示例重复）
{shots}

## 质量红线（违反一条整批重生成）
1. output_json.device_id **必须是 8 种标准 device_id**（bedroom_* / living_room_*），任何通用 ID（ac / light / curtain / tv）一律算错。
2. need_clarification 必须为 false（所有样本都必须是可直接执行的确定性指令，不能含糊）。
3. 禁止输出 ```json 代码块，只输出纯 JSON array。
4. 输入文本尽量贴近真实口语（错字、礼貌词、啰嗦都行），但不能用代词（这个/那个）让意图在语言外。
5. 20% 样本可以是 多参数 / 多意图复合（比如 "开XX + 调到Y度"）。

现在开始，只输出 JSON array（长度 = {n_requested}）。"""


def call_siliconflow(prompt: str, timeout: int = 180, retries: int = 4) -> Optional[str]:
    try:
        import requests
    except Exception:
        return None
    url = f"{SILICONFLOW_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SILICONFLOW_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n你现在的任务不是解析，而是合成训练对。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
        "max_tokens": 8000,
        "stream": False,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                print(f"  [LLM HTTP {resp.status_code}] attempt {attempt+1}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
                continue
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [LLM exception] attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return None


def strip_code_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # 去掉 ```json ... ```
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def normalize_batch(raw: Any) -> List[Dict[str, Any]]:
    """把 LLM 返回的 （可能脏的） JSON array 清洗成与 01/02 同 schema 的 list。"""
    from device_controller import DEVICE_STATES, DEVICE_TYPE_NAMES
    legal_devices = set(DEVICE_STATES.keys())
    legal_actions = {"on", "off", "set", "set_temperature", "increase", "decrease"}
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        inp = item.get("input")
        o = item.get("output_json") or {}
        if not isinstance(inp, str) or not inp.strip():
            continue
        if not isinstance(o, dict):
            continue
        # 1) 设备 ID 修正/校验
        dev_id = o.get("device_id") or ""
        if dev_id not in legal_devices:
            # 允许从 device_name 里通过子串近似回写（"客厅空调" → living_room_ac）
            dn = str(o.get("device_name") or "")
            for cand in sorted(legal_devices, key=len, reverse=True):
                st = DEVICE_STATES.get(cand) or {}
                room = st.get("room") or ""
                t = st.get("type") or ""
                cn = room + DEVICE_TYPE_NAMES.get(t, t)
                if cn and cn in dn:
                    dev_id = cand
                    break
            if dev_id not in legal_devices:
                continue  # 修不好就丢
        # 2) action 校验
        act = o.get("action")
        if act not in legal_actions:
            if isinstance(act, str):
                act_l = act.lower()
                if act_l in ("turn_on", "turnoff"): act = "on"
                elif act_l in ("turn_off", "turnoff"): act = "off"
                elif "temp" in act_l: act = "set_temperature"
                else: act = "set"
            else:
                act = "set"
        st = DEVICE_STATES.get(dev_id) or {}
        room_cn = st.get("room") or ""
        t = st.get("type") or ""
        dev_cn = room_cn + DEVICE_TYPE_NAMES.get(t, t)
        normalized = {
            "success": True,
            "msg": "解析成功",
            "device_id": dev_id,
            "device_name": o.get("device_name") or dev_cn,
            "action": act,
            "value": o.get("value"),
            "value_type": o.get("value_type"),
            "attrs": o.get("attrs") if isinstance(o.get("attrs"), dict) else {},
            "confidence": 1.0,
            "need_clarification": False,
        }
        out.append({
            "input": inp.strip(),
            "output_json": normalized,
            "meta": {"source": "llm_augment"},
        })
    return out


# ================================================================
# ★ Fallback：离线 rule-based 自然口语合成器
#   - 当 SiliconFlow 全部超时/失败时，用这份纯本地模板补够 --count 条
#   - 目标：覆盖"情境指令 + 间接请求 + 礼貌口语 + 错词/方言 + 复合指令"这些
#     01/02 模板味较重没覆盖到的 edge case。
# ================================================================

_POLITE_PREFIX = [
    "唉，", "那个，", "嗯，", "我说啊，", "我说，", "你看啊，",
    "能不能", "能不", "你给", "你给我", "帮我", "麻烦", "麻烦你", "请", "请你", "给我",
    "我说能不能", "我说你能不能", "我说帮我",
]
_POLITE_SUFFIX = [
    "哦", "啊", "吧", "呗", "嘛", "哈", "谢谢", "好吗", "行吗", "行不行", "可以吗", "好不好",
    "麻烦了", "感谢", "辛苦啦",
]
_DIALECT_PARTICLES = [
    "撒", "噻", "咯", "嘛", "啦", "哈", "撒", "咩",
]

_SITUATIONS: List[Tuple[str, Dict[str, Any], str]] = [
    # (自然口语模板, output_json 的关键槽, 场景标签)
    # —— 热 / 冷 感受 ——
    ("{prefix}{room}有点热啊{suffix}", {"room": "客厅", "device_id": "living_room_ac",  "action": "on", "value_type": None, "attrs": {"mode": "cooling"}}, "间接请求"),
    ("{prefix}{room}太热了{suffix}",       {"room": "客厅", "device_id": "living_room_ac",  "action": "on", "value_type": None, "attrs": {"mode": "cooling"}}, "间接请求"),
    ("{prefix}{room}有点闷开个空调{suffix}", {"room": "客厅", "device_id": "living_room_ac",  "action": "on", "value_type": None, "attrs": {"mode": "cooling"}}, "间接请求"),
    ("{prefix}{room}好闷啊{suffix}",       {"room": "卧室", "device_id": "bedroom_ac",       "action": "on", "value_type": None, "attrs": {"mode": "cooling"}}, "间接请求"),
    ("{prefix}{room}有点冷{suffix}",       {"room": "客厅", "device_id": "living_room_ac",  "action": "on", "value_type": None, "attrs": {"mode": "heating"}}, "间接请求"),
    ("{prefix}{room}太冷了{suffix}",       {"room": "卧室", "device_id": "bedroom_ac",       "action": "on", "value_type": None, "attrs": {"mode": "heating"}}, "间接请求"),
    # —— 光 / 晃眼 ——
    ("{prefix}{room}太黑了{suffix}",       {"room": "卧室", "device_id": "bedroom_light",    "action": "on",  "value_type": None, "attrs": {}}, "间接请求"),
    ("{prefix}{room}有点暗{suffix}",       {"room": "客厅", "device_id": "living_room_light", "action": "on",  "value_type": None, "attrs": {}}, "间接请求"),
    ("{prefix}{room}太亮了晃眼{suffix}",    {"room": "客厅", "device_id": "living_room_light", "action": "off", "value_type": None, "attrs": {}}, "间接请求"),
    ("{prefix}{room}晃眼{suffix}",         {"room": "卧室", "device_id": "bedroom_light",    "action": "off", "value_type": None, "attrs": {}}, "间接请求"),
    ("{prefix}{room}好刺眼把窗连拉一下{suffix}", {"room": "客厅", "device_id": "living_room_curtain", "action": "set", "value": "全关", "value_type": "curtain_state", "attrs": {"state": "全关"}}, "间接请求"),
    ("{prefix}{room}太阳太大了把窗帘拉上{suffix}", {"room": "卧室", "device_id": "bedroom_curtain", "action": "set", "value": "全关", "value_type": "curtain_state", "attrs": {"state": "全关"}}, "间接请求"),
    # —— 情境指令（睡觉 / 吃饭 / 离家）——
    ("{prefix}要睡觉了把{room}灯关掉{suffix}",       {"room": "卧室", "device_id": "bedroom_light",    "action": "off", "value_type": None, "attrs": {}}, "情境指令"),
    ("{prefix}我走了把{room}灯关一下{suffix}",        {"room": "客厅", "device_id": "living_room_light", "action": "off", "value_type": None, "attrs": {}}, "情境指令"),
    ("{prefix}吃饭了{room}开亮一点{suffix}",          {"room": "客厅", "device_id": "living_room_light", "action": "set", "value": 100, "value_type": "brightness", "attrs": {"brightness": 100}}, "情境指令"),
    ("{prefix}要睡觉了{room}灯调暗点{suffix}",        {"room": "卧室", "device_id": "bedroom_light",    "action": "set", "value": 30,  "value_type": "brightness", "attrs": {"brightness": 30}},  "情境指令"),
    ("{prefix}看电视了把{room}灯暗一点{suffix}",       {"room": "客厅", "device_id": "living_room_light", "action": "set", "value": 30,  "value_type": "brightness", "attrs": {"brightness": 30}},  "情境指令"),
    # —— 间接请求 + 温度 ——
    ("{prefix}{room}空调再高一点{suffix}",   {"room": "卧室", "device_id": "bedroom_ac",      "action": "increase", "value_type": "temperature", "attrs": {}}, "间接请求"),
    ("{prefix}{room}空调再低一度{suffix}",   {"room": "客厅", "device_id": "living_room_ac",   "action": "decrease", "value_type": "temperature", "attrs": {}}, "间接请求"),
    ("{prefix}娃要洗澡，{room}空调开个{temp}度{suffix}", {"room": "卧室", "device_id": "bedroom_ac",    "action": "on", "value": "<TEMP>", "value_type": "temperature", "attrs": {"temperature": "<TEMP>", "mode": "heating"}}, "情境指令"),
    ("{prefix}我妈嫌热，{room}空调再凉一点{suffix}", {"room": "客厅", "device_id": "living_room_ac", "action": "decrease", "value_type": "temperature", "attrs": {}}, "间接请求"),
    ("{prefix}好刺眼，{room}那个空条能不能开下{suffix}", {"room": "客厅", "device_id": "living_room_ac", "action": "on", "value_type": None, "attrs": {"mode": "cooling"}}, "代词+错词"),
    # —— 错词 + 方言口语 ——
    ("{prefix}克厅那个卧试风格的空条关毕下{suffix}", {"room": "客厅", "device_id": "living_room_ac", "action": "off", "value_type": None, "attrs": {}}, "错词堆叠"),
    ("{prefix}卧试的窗连关毕{suffix}",             {"room": "卧室", "device_id": "bedroom_curtain", "action": "set", "value": "全关", "value_type": "curtain_state", "attrs": {"state": "全关"}}, "错词堆叠"),
    ("{prefix}客听的灯大开一下嘛{suffix}",          {"room": "客厅", "device_id": "living_room_light", "action": "on", "value_type": None, "attrs": {}}, "错词+方言"),
    ("{prefix}卧世的电世关一下{suffix}",            {"room": "卧室", "device_id": "bedroom_tv", "action": "off", "value_type": None, "attrs": {}}, "错词堆叠"),
    ("{prefix}克厅窗连拉成半开撒{suffix}",          {"room": "客厅", "device_id": "living_room_curtain", "action": "set", "value": "半开", "value_type": "curtain_state", "attrs": {"state": "半开"}}, "错词+方言"),
]


def _out_json_from_slot(slot: Dict[str, Any], temp_override: Optional[int] = None) -> Dict[str, Any]:
    from device_controller import DEVICE_STATES, DEVICE_TYPE_NAMES
    dev_id = slot["device_id"]
    action = slot["action"]
    st = DEVICE_STATES.get(dev_id) or {}
    room_cn = st.get("room") or ""
    t = st.get("type") or ""
    dev_cn = room_cn + DEVICE_TYPE_NAMES.get(t, t)

    value = slot.get("value")
    value_type = slot.get("value_type")
    attrs = dict(slot.get("attrs") or {})
    if value == "<TEMP>" and temp_override is not None:
        value = temp_override
        value_type = "temperature"
        if "temperature" in attrs:
            attrs["temperature"] = temp_override
    return {
        "success": True,
        "msg": "解析成功",
        "device_id": dev_id,
        "device_name": dev_cn,
        "action": action,
        "value": value,
        "value_type": value_type,
        "attrs": attrs,
        "confidence": 1.0,
        "need_clarification": False,
    }


def _decompose_room(slot: Dict[str, Any]) -> Dict[str, Any]:
    """把带 '客厅/卧室' 二选一的 slot 展开成 2 条（客厅/卧室各一条），用来丰富分布。"""
    # 如果 slot["room"] 固定，就原样返回单条
    return [slot]


def local_rule_augment(target: int) -> List[Dict[str, Any]]:
    """离线 rule-based 合成 target 条样本。"""
    random.seed(44 + target)
    temps = [22, 24, 25, 26, 27, 28]
    out: List[Dict[str, Any]] = []
    seen: set = set()
    safety = 0
    max_loops = target * 30

    # 加载 02_perturb_asr（不是 package 的子模块，按绝对路径 import）
    import importlib.util
    p2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_perturb_asr.py")
    spec = importlib.util.spec_from_file_location("p2", p2_path)
    p2 = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(p2)

    while len(out) < target and safety < max_loops:
        safety += 1
        tmpl, slot, _tag = random.choice(_SITUATIONS)
        # 10% 概率把房间从 slot 反一下（比如客厅→卧室，卧室→客厅），扩充房间分布
        slot_var = dict(slot)
        if random.random() < 0.5:
            # flip room
            room_var = random.choice(["客厅", "卧室"])
            slot_var["room"] = room_var
            slot_var["device_id"] = room_to_device(room_var, slot_var["device_id"])
        # 温度占位补真实值
        temp_override: Optional[int] = None
        if slot_var.get("value") == "<TEMP>":
            temp_override = random.choice(temps)
        # 填模板
        prefix = random.choice(_POLITE_PREFIX + [""] * 4)  # 60% 不加前缀
        suffix = random.choice(_POLITE_SUFFIX + _DIALECT_PARTICLES + [""] * 6)  # 不加后缀 60%
        # 错词替换：以 ~40% 概率把 room/device 相关正确词替换成错词（复用 02_perturb_asr 的 CORRECT_TO_WRONG）
        text = tmpl.format(prefix=prefix, room=slot_var["room"], suffix=suffix,
                           temp=temp_override if temp_override is not None else 26)
        # 做轻量 ASR 错词（p=0.4）+ 再轻度扰动语序
        if random.random() < 0.4:
            text = p2.apply_phonetic(text, p_per_slot=0.5)
        if random.random() < 0.2:
            text = p2.apply_oral(text, p_prefix=0.0, p_suffix=0.6)
        text = re.sub(r"\s+", "", text).strip("，。、？？！!")
        if not text:
            continue
        if text in seen:
            continue
        out_json = _out_json_from_slot(slot_var, temp_override=temp_override)
        # 强校验：拼出来的 device_id / device_name 必须合法，否则丢弃（防止 room_to_device 偶尔拼错进入数据集）
        from device_controller import DEVICE_STATES
        dev_id = out_json.get("device_id")
        if dev_id not in DEVICE_STATES or not out_json.get("device_name"):
            continue
        seen.add(text)
        out.append({
            "input": text,
            "output_json": out_json,
            "meta": {"source": "local_rule_augment", "scenario": _tag},
        })
    random.shuffle(out)
    return out


def room_to_device(room_cn: str, device_id_template: str) -> str:
    # 把一个 device_id（不管原先是客厅还是卧室）强制替换成 room_cn 对应房间
    # device_id_template 形如 bedroom_ac / living_room_curtain —— 下划线分三段：<room_prefix>_<room_suffix>_<type>
    # 所以要 split("_", 2)[2] 取最后那段 type（ac/light/curtain/tv），再拼 slug_type
    parts = device_id_template.split("_")
    if len(parts) >= 3:
        suffix = parts[-1]  # ac / light / curtain / tv
    elif len(parts) == 2:
        suffix = parts[1]
    else:
        suffix = device_id_template
    slug = {"客厅": "living_room", "卧室": "bedroom"}.get(room_cn, "living_room")
    new_id = f"{slug}_{suffix}"
    # 最终合法性强校验：如果拼出来不在 8 种合法 ID 里就不换，直接返回模板原样（下游 normalize 里会再丢）
    from device_controller import DEVICE_STATES
    if new_id in DEVICE_STATES:
        return new_id
    return device_id_template




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=300, help="目标条数（每条 batch 会拆成 60 条一次，拼够为止）")
    ap.add_argument("--dry", action="store_true", help="只打印 prompt，不调 API")
    ap.add_argument("--output", type=str, default="")
    ap.add_argument("--timeout", type=int, default=180, help="LLM HTTP 单次调用超时（秒），默认 180")
    ap.add_argument("--retries", type=int, default=4, help="LLM 每次调用的重试次数，默认 4")
    ap.add_argument("--batch", type=int, default=60, help="每次请求的 batch 条数（过小会太慢过大会爆 max_tokens）")
    ap.add_argument("--force-local", action="store_true",
                    help="不调 SiliconFlow，直接用离线 rule-based 合成器补够 --count 条（保证今天一定拿得到数据）")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "finetune", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, "llm_augment_samples.json")

    if args.dry:
        print(build_user_prompt(5))
        return

    BATCH = max(1, args.batch)
    all_items: List[Dict[str, Any]] = []
    seen_inp: set = set()
    budget = args.count

    # ============== 1) 优先：LLM 远端合成（除非 --force-local 或 API 不可用）==============
    llm_skipped = False
    if args.force_local:
        print("[03] --force-local 开启，跳过 LLM，直接用本地合成器。")
        llm_skipped = True
    elif not (SILICONFLOW_API_KEY and SILICONFLOW_BASE_URL and SILICONFLOW_MODEL):
        print("[03] 环境变量里缺少 SILICONFLOW_*，跳过 LLM。")
        llm_skipped = True

    if not llm_skipped:
        round_idx = 0
        llm_rounds_max = max(3, budget // max(1, BATCH) + 2)
        while len(all_items) < budget and round_idx < llm_rounds_max:
            round_idx += 1
            need = min(BATCH, max(BATCH, budget - len(all_items)))
            prompt = build_user_prompt(need)
            print(f"[round {round_idx}] 向 LLM 请求 {need} 条...（当前已累计 {len(all_items)}）")
            content = call_siliconflow(prompt, timeout=args.timeout, retries=args.retries)
            if not content:
                print("  · 本轮 LLM 多次重试仍无响应，停止继续调远端，剩余用本地合成器补。")
                break
            cleaned = strip_code_block(content)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError as e:
                print(f"  · JSON 解析失败：{e}，尝试截断修复")
                fixed = None
                for end in range(len(cleaned), 1, -1):
                    try:
                        fixed = json.loads(cleaned[:end] + "]")
                        break
                    except Exception:
                        continue
                data = fixed
            batch_norm = normalize_batch(data)
            added = 0
            for it in batch_norm:
                key = it["input"]
                if key in seen_inp:
                    continue
                seen_inp.add(key)
                all_items.append(it)
                added += 1
            print(f"  · 本轮新增 {added} 条。")
            time.sleep(0.5)

    # ============== 2) 保底：本地 rule-based 合成器补够预算（任何情况下都能凑齐 --count）==============
    if len(all_items) < budget:
        need = budget - len(all_items)
        print(f"[03] 远端 LLM 仅产出 {len(all_items)} 条，本地合成器再补 {need} 条...")
        local = local_rule_augment(need)
        for it in local:
            key = it["input"]
            if key in seen_inp:
                continue
            seen_inp.add(key)
            all_items.append(it)
        if len(all_items) < budget:
            # 还是不够就做更多扰动重复：对已有 local 再做 apply_phonetic + apply_oral 扩增
            print(f"[03] 本地一次采样仍不够（{len(all_items)} < {budget}），对已产出样本做轻扰动扩增补量。")
            extra_needed = budget - len(all_items)
            import importlib.util as _ilu
            _p2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_perturb_asr.py")
            _spec = _ilu.spec_from_file_location("p2_fallback", _p2_path)
            _p2 = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
            assert _spec is not None and _spec.loader is not None
            _spec.loader.exec_module(_p2)
            base_pool = list(all_items)
            guard = 0
            while len(all_items) < budget and guard < extra_needed * 10:
                guard += 1
                base = random.choice(base_pool)
                inp2 = _p2.apply_phonetic(base["input"], p_per_slot=0.6)
                inp2 = _p2.apply_oral(inp2, p_prefix=0.4, p_suffix=0.5)
                inp2 = _p2.apply_punct(inp2, p=0.6)
                inp2 = _p2.apply_word_order(inp2, p=0.3)
                inp2 = re.sub(r"\s+", "", inp2).strip("，。、？？！!")
                if not inp2 or inp2 in seen_inp:
                    continue
                seen_inp.add(inp2)
                all_items.append({
                    "input": inp2,
                    "output_json": deepcopy(base["output_json"]),
                    "meta": {**base.get("meta", {}), "source": "local_rule_augment_perturb"},
                })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    print(f"[llm_augment] 共 {len(all_items)} 条 → {out_path}")
    # 打印场景分布
    from collections import Counter
    tags = Counter((s.get("meta") or {}).get("source") for s in all_items)
    print(" · source 分布：", tags)
    scen = Counter((s.get("meta") or {}).get("scenario") for s in all_items if (s.get("meta") or {}).get("scenario"))
    if scen:
        print(" · scenario 分布（local 合成部分）：")
        for k, v in scen.most_common():
            print(f"    · {k:<10s} {v:>4d}")


if __name__ == "__main__":
    main()
