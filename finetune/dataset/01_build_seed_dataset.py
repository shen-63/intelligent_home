"""
【微调 · 阶段 A-1】种子数据集构建（Seed Dataset）

目标：基于本项目真实注册的设备 + 合法动作枚举，用"模板 → 自然语言 Input"的方式，
生成 ~500 条高质量"用户指令文本 → 标准 JSON 输出"对（seed_samples.json）。

这些样本都**不带 ASR 错词 / 不带口语扰动**（这些放在 02_perturb_asr.py 里做），
保证基座模型先学到"标准指令应该对应什么 JSON"的主分布。

用法：
    cd <project_root>
    Agent_env\Scripts\python.exe finetune/dataset/01_build_seed_dataset.py

输出：
    finetune/outputs/seed_samples.json   # list[dict], each={input:str, output_json:dict, meta:{...}}
"""
from __future__ import annotations

import json
import os
import random
import sys
from itertools import product
from typing import Any, Dict, List, Tuple

# 允许直接从项目根 import 我们真实的 NLU 模块（保证输出 JSON schema 一致）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from device_controller import (
    DEVICE_STATES,
    DEVICE_TYPE_NAMES,
    AC_MODE_NAMES,
    AC_FAN_NAMES,
    AC_SWING_NAMES,
)
from nlu_parser import ROOM_CN_TO_SLUG

random.seed(42)

# ================================================================
# 1. 合法枚举（必须与 nlu_parser/device_controller 完全对齐）
# ================================================================

# 注册的 8 台设备
DEVICES: List[Tuple[str, str, str]] = []  # (device_id, room_cn, device_cn)
for dev_id, st in DEVICE_STATES.items():
    room_cn = st.get("room") or ""
    t = st.get("type") or ""
    DEVICES.append((dev_id, room_cn, DEVICE_TYPE_NAMES.get(t, t)))

# 通用动作 → action 枚举（只保留我们现有管线会理解的那些）
ACTIONS_ON_OFF = ["on", "off"]
ACTIONS_SET = ["set_temperature", "set_brightness", "set_fan_mode", "set_hvac_mode", "set_swing_mode", "set_curtain"]
ACTIONS_INC_DEC = ["increase", "decrease"]

# 参数候选值（真实允许范围）
TEMP_RANGE = [16, 18, 20, 22, 24, 25, 26, 27, 28, 30]
BRIGHT_RANGE = [10, 30, 50, 80, 100]
# 取每个 AC_*_NAMES 的**合法右值**（用 set 去重）
HVAC_MODES = sorted({v for v in AC_MODE_NAMES.values()})  # 制冷/制热/除湿/送风/自动
FAN_MODES  = sorted({v for v in AC_FAN_NAMES.values() if isinstance(v, str)})  # 1档~5档/静音档/自动风
SWING_MODES = sorted({v for v in AC_SWING_NAMES.values()})  # 上下扫风/左右扫风/关闭扫风
CURTAIN_STATES = ["全开", "全关", "半开"]

# 中文 → 英文归一化（让 output JSON 里 value_type 里的字符串和 rule NLU 输出完全一致）
MODE_CN_TO_SLUG: Dict[str, str] = {
    "制冷": "cooling", "制热": "heating", "除湿": "dehumidify",
    "送风": "fan", "自动": "auto",
}
FAN_CN_TO_SLUG: Dict[str, str] = {
    "静音档": "quiet", "1档": 1, "2档": 2, "3档": 3, "4档": 4, "5档": 5, "自动风": "auto",
}
SWING_CN_TO_SLUG: Dict[str, str] = {
    "上下扫风": "vertical", "左右扫风": "horizontal", "关闭扫风": "off",
}

# ================================================================
# 2. 指令模板（每种动作配多套自然表达，避免模板味太重）
# ================================================================

TEMPLATES_ON_OFF = {
    "on": [
        "打开{device}",
        "开启{device}",
        "{device}打开",
        "把{device}打开",
        "帮我开一下{device}",
        "开{device}",
    ],
    "off": [
        "关闭{device}",
        "关掉{device}",
        "把{device}关闭",
        "{device}关掉",
        "{device}关一下",
        "关{device}",
    ],
}

TEMPLATES_TEMP = [
    "把{device}调到{temp}度",
    "{device}设为{temp}度",
    "{device}调到{temp}度",
    "{device}温度{temp}度",
    "将{device}温度设置为{temp}度",
    "把{device}开{temp}度",
]

TEMPLATES_BRIGHT = [
    "把{device}调到{bright}%亮度",
    "{device}亮度设为{bright}%",
    "{device}亮度{bright}%",
]

TEMPLATES_HVAC = [
    "{device}切到{mode}",
    "把{device}调到{mode}模式",
    "{device}模式{mode}",
]

TEMPLATES_FAN = [
    "{device}风速设为{fan}",
    "把{device}风速调到{fan}",
    "{device}{fan}",
]

TEMPLATES_SWING = [
    "{device}扫风设为{swing}",
    "把{device}调成{swing}",
]

TEMPLATES_CURTAIN = [
    "把{device}拉成{state}",
    "{device}{state}",
]

TEMPLATES_INC_DEC = {
    "increase": [
        "{device}调高一度",
        "{device}温度再高一点",
        "{device}再加一点温度",
    ],
    "decrease": [
        "{device}调低一度",
        "{device}温度再低一点",
        "{device}降一点温度",
    ],
}

# 环境感知类指令（纯语义，没有显式设备/动作词，ASR 对这类也容易错）
ENV_PROMPTS: List[Tuple[str, Dict[str, Any]]] = [
    ("客厅有点热",   {"device_id": "living_room_ac",  "action": "on", "attrs": {"mode": "cooling"}, "value_type": None}),
    ("客厅太热了",   {"device_id": "living_room_ac",  "action": "on", "attrs": {"mode": "cooling"}, "value_type": None}),
    ("客厅有点闷",   {"device_id": "living_room_ac",  "action": "on", "attrs": {"mode": "cooling"}, "value_type": None}),
    ("客厅有点冷",   {"device_id": "living_room_ac",  "action": "on", "attrs": {"mode": "heating"}, "value_type": None}),
    ("客厅太冷了",   {"device_id": "living_room_ac",  "action": "on", "attrs": {"mode": "heating"}, "value_type": None}),
    ("客厅有点暗",   {"device_id": "living_room_light", "action": "on", "attrs": {}, "value_type": None}),
    ("客厅太黑了",   {"device_id": "living_room_light", "action": "on", "attrs": {}, "value_type": None}),
    ("客厅太亮了",   {"device_id": "living_room_light", "action": "off", "attrs": {}, "value_type": None}),
    ("客厅晃眼",     {"device_id": "living_room_light", "action": "off", "attrs": {}, "value_type": None}),
    ("卧室有点热",   {"device_id": "bedroom_ac",       "action": "on", "attrs": {"mode": "cooling"}, "value_type": None}),
    ("卧室太热了",   {"device_id": "bedroom_ac",       "action": "on", "attrs": {"mode": "cooling"}, "value_type": None}),
    ("卧室有点冷",   {"device_id": "bedroom_ac",       "action": "on", "attrs": {"mode": "heating"}, "value_type": None}),
    ("卧室太冷了",   {"device_id": "bedroom_ac",       "action": "on", "attrs": {"mode": "heating"}, "value_type": None}),
    ("卧室有点暗",   {"device_id": "bedroom_light",    "action": "on", "attrs": {}, "value_type": None}),
    ("卧室太黑了",   {"device_id": "bedroom_light",    "action": "on", "attrs": {}, "value_type": None}),
    ("卧室太亮了",   {"device_id": "bedroom_light",    "action": "off", "attrs": {}, "value_type": None}),
    ("卧室晃眼",     {"device_id": "bedroom_light",    "action": "off", "attrs": {}, "value_type": None}),
]


def _base_output_json(device_id: str, action: str) -> Dict[str, Any]:
    """按照 llm_nlu.py SYSTEM_PROMPT 里"输出格式（严格 JSON）"严格对齐。"""
    st = DEVICE_STATES.get(device_id) or {}
    room_cn = st.get("room") or ""
    t = st.get("type") or ""
    device_name_cn = room_cn + DEVICE_TYPE_NAMES.get(t, t)

    return {
        "success": True,
        "msg": "解析成功",
        "device_id": device_id,
        "device_name": device_name_cn,
        "action": action,
        "value": None,
        "value_type": None,
        "attrs": {},
        "confidence": 1.0,
        "need_clarification": False,
    }


def _is_ac(did: str) -> bool:
    return did.endswith("_ac")

def _is_light(did: str) -> bool:
    return did.endswith("_light")

def _is_curtain(did: str) -> bool:
    return did.endswith("_curtain")

def _is_tv(did: str) -> bool:
    return did.endswith("_tv")


# ================================================================
# 3. 生成器
# ================================================================

SAMPLES: List[Dict[str, Any]] = []


def _add(inp: str, out: Dict[str, Any], meta: Dict[str, Any]) -> None:
    SAMPLES.append({
        "input": inp,
        "output_json": out,
        "meta": {"source": "seed_template", **meta},
    })


def build_on_off() -> None:
    for (dev_id, room_cn, dev_cn), action in product(DEVICES, ACTIONS_ON_OFF):
        device_display = room_cn + dev_cn
        for tmpl in TEMPLATES_ON_OFF[action]:
            out = _base_output_json(dev_id, action)
            _add(tmpl.format(device=device_display), out, {"kind": f"onoff_{action}", "device_id": dev_id})


def build_temperature() -> None:
    acs = [(d, r, c) for d, r, c in DEVICES if _is_ac(d)]
    # 2 条/台/度：选 10 个 temp × 2 台 × 2 模板（不×所有模板，避免 over-rep 温度）
    for (dev_id, room_cn, dev_cn), temp in product(acs, TEMP_RANGE):
        out = _base_output_json(dev_id, "set_temperature")
        out["value"] = temp
        out["value_type"] = "temperature"
        out["attrs"]["temperature"] = temp
        for tmpl in random.sample(TEMPLATES_TEMP, 2):
            _add(tmpl.format(device=room_cn + dev_cn, temp=temp), out,
                 {"kind": "temp_set", "device_id": dev_id, "temp": temp})


def build_brightness() -> None:
    lights = [(d, r, c) for d, r, c in DEVICES if _is_light(d)]
    for (dev_id, room_cn, dev_cn), bright in product(lights, BRIGHT_RANGE):
        out = _base_output_json(dev_id, "set")
        out["value"] = bright
        out["value_type"] = "brightness"
        out["attrs"]["brightness"] = bright
        for tmpl in TEMPLATES_BRIGHT[:2]:
            _add(tmpl.format(device=room_cn + dev_cn, bright=bright), out,
                 {"kind": "bright_set", "device_id": dev_id, "bright": bright})


def build_ac_modes() -> None:
    acs = [(d, r, c) for d, r, c in DEVICES if _is_ac(d)]
    # 空调模式
    for (dev_id, room_cn, dev_cn), mode_cn in product(acs, HVAC_MODES):
        out = _base_output_json(dev_id, "set")
        out["value"] = mode_cn
        out["value_type"] = "hvac_mode"
        out["attrs"]["mode"] = MODE_CN_TO_SLUG[mode_cn]
        for tmpl in TEMPLATES_HVAC[:2]:
            _add(tmpl.format(device=room_cn + dev_cn, mode=mode_cn), out,
                 {"kind": "hvac_set", "device_id": dev_id, "mode_cn": mode_cn})
    # 风速
    for (dev_id, room_cn, dev_cn), fan_cn in product(acs, FAN_MODES):
        out = _base_output_json(dev_id, "set")
        out["value"] = fan_cn
        out["value_type"] = "fan_mode"
        out["attrs"]["fan_speed"] = FAN_CN_TO_SLUG[fan_cn]
        for tmpl in TEMPLATES_FAN[:2]:
            _add(tmpl.format(device=room_cn + dev_cn, fan=fan_cn), out,
                 {"kind": "fan_set", "device_id": dev_id, "fan_cn": fan_cn})
    # 扫风
    for (dev_id, room_cn, dev_cn), swing_cn in product(acs, SWING_MODES):
        out = _base_output_json(dev_id, "set")
        out["value"] = swing_cn
        out["value_type"] = "swing_mode"
        out["attrs"]["swing"] = SWING_CN_TO_SLUG[swing_cn]
        for tmpl in TEMPLATES_SWING[:2]:
            _add(tmpl.format(device=room_cn + dev_cn, swing=swing_cn), out,
                 {"kind": "swing_set", "device_id": dev_id, "swing_cn": swing_cn})


def build_curtain_state() -> None:
    curtains = [(d, r, c) for d, r, c in DEVICES if _is_curtain(d)]
    for (dev_id, room_cn, dev_cn), state_cn in product(curtains, CURTAIN_STATES):
        out = _base_output_json(dev_id, "set")
        out["value"] = state_cn
        out["value_type"] = "curtain_state"
        out["attrs"]["state"] = state_cn
        for tmpl in TEMPLATES_CURTAIN[:2]:
            _add(tmpl.format(device=room_cn + dev_cn, state=state_cn), out,
                 {"kind": "curtain_set", "device_id": dev_id, "state_cn": state_cn})


def build_tv_channel() -> None:
    # TV 只有频道这个显式参数（现有 schema）
    tvs = [(d, r, c) for d, r, c in DEVICES if _is_tv(d)]
    channels = [1, 5, 12, 23, 36]
    for (dev_id, room_cn, dev_cn), ch in product(tvs, channels):
        out = _base_output_json(dev_id, "set")
        out["value"] = ch
        out["value_type"] = "channel"
        out["attrs"]["channel"] = ch
        for tmpl in [
            "把{device}调到{ch}频道",
            "{device}频道{ch}",
            "{device}切到{ch}台",
        ]:
            _add(tmpl.format(device=room_cn + dev_cn, ch=ch), out,
                 {"kind": "tv_channel", "device_id": dev_id, "channel": ch})


def build_inc_dec_ac() -> None:
    acs = [(d, r, c) for d, r, c in DEVICES if _is_ac(d)]
    for (dev_id, room_cn, dev_cn), action in product(acs, ACTIONS_INC_DEC):
        out = _base_output_json(dev_id, action)
        out["value_type"] = "temperature"
        out["attrs"] = {}
        for tmpl in TEMPLATES_INC_DEC[action]:
            _add(tmpl.format(device=room_cn + dev_cn), out,
                 {"kind": f"incdec_{action}", "device_id": dev_id})


def build_env_prompts() -> None:
    for text, slot in ENV_PROMPTS:
        out = _base_output_json(slot["device_id"], slot["action"])
        out["value_type"] = slot.get("value_type")
        out["attrs"] = slot.get("attrs") or {}
        _add(text, out, {"kind": "env_prompt", "device_id": slot["device_id"]})


def build_compound_multi_attr() -> None:
    """复合指令：'打开客厅空调调到26度'这种同设备多槽位，之前图一里失败的场景。"""
    acs = [(d, r, c) for d, r, c in DEVICES if _is_ac(d)]
    TEMPS_PICK = [24, 25, 26, 28]
    MODES_PICK = list(HVAC_MODES)[:3]
    for (dev_id, room_cn, dev_cn), temp, mode_cn in product(acs, TEMPS_PICK, MODES_PICK):
        out = _base_output_json(dev_id, "on")
        out["value"] = temp
        out["value_type"] = "temperature"
        out["attrs"] = {"temperature": temp, "mode": MODE_CN_TO_SLUG[mode_cn]}
        samples_inp = [
            f"打开{room_cn}{dev_cn}调到{temp}度切到{mode_cn}",
            f"把{room_cn}{dev_cn}打开调到{temp}度并切到{mode_cn}",
            f"{room_cn}{dev_cn}打开，{temp}度，{mode_cn}",
        ]
        for inp in samples_inp:
            _add(inp, out, {"kind": "compound_ac_on_temp_mode", "device_id": dev_id})

    # 同设备：空调 set_temperature + 风速（没有 on/off）
    for (dev_id, room_cn, dev_cn), temp, fan_cn in product(acs, TEMPS_PICK, ["2档", "3档", "自动风"]):
        out = _base_output_json(dev_id, "set_temperature")
        out["value"] = temp
        out["value_type"] = "temperature"
        out["attrs"] = {"temperature": temp, "fan_speed": FAN_CN_TO_SLUG[fan_cn]}
        for inp in [
            f"{room_cn}{dev_cn}调到{temp}度，风速{fan_cn}",
            f"把{room_cn}{dev_cn}调到{temp}度再设{fan_cn}",
        ]:
            _add(inp, out, {"kind": "compound_ac_temp_fan", "device_id": dev_id})


def main() -> None:
    build_on_off()
    build_temperature()
    build_brightness()
    build_ac_modes()
    build_curtain_state()
    build_tv_channel()
    build_inc_dec_ac()
    build_env_prompts()
    build_compound_multi_attr()

    random.shuffle(SAMPLES)

    out_dir = os.path.join(ROOT, "finetune", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "seed_samples.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(SAMPLES, f, ensure_ascii=False, indent=2)

    print(f"[seed] 共生成 {len(SAMPLES)} 条样本 → {out_path}")
    # 打个覆盖率：每类 kind 数量
    from collections import Counter
    c = Counter(s["meta"].get("kind") for s in SAMPLES)
    for k, v in c.most_common():
        print(f"  · {k:<28s} {v:>4d}")


if __name__ == "__main__":
    main()
