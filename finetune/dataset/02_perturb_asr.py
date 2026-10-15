"""
【微调 · 阶段 A-2】ASR 扰动 & 口语扩写

输入：finetune/outputs/seed_samples.json（来自 01_build_seed_dataset.py）
动作：
  1) 同音错词替换（卧试/克厅/空条/窗连/关毕/大开/… 与 nlu_parser._PHONETIC_REPLACE_RULES_UNIQUE 严格对齐，方向是"正确词→错词"）
  2) 口语前缀/后缀插入（"帮我"、"麻烦你"、"一下"、"好不好"、"吧"、"啊"…）
  3) 语序变换（把 XX 打开 → XX 打开 / 打开XX 等；标点变化）
  4) 数字口语化（"26度" ↔ "二十六度"）
  5) 每条种子 → 随机扰动 2~3 份变体（p=0.9 应用扰动，p=0.1 保留原句，保证有干净样本）
输出：finetune/outputs/perturbed_samples.json

注意：所有扰动都只改 input，不改 output_json（保证监督信号不变）。
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from copy import deepcopy
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nlu_parser import (
    _PHONETIC_REPLACE_RULES_UNIQUE,  # 错→正 的规则，这里要反过来用
)

random.seed(43)

# 把 nlu_parser 的错→正规则翻转成"正词→常见错词候选"
# 注意：同一个"正词"可能对应多个错词变体，按原列表顺序收集并去重。
CORRECT_TO_WRONG: Dict[str, List[str]] = {}
for wrong, correct in _PHONETIC_REPLACE_RULES_UNIQUE:
    if not correct or not wrong or correct == wrong:
        continue
    CORRECT_TO_WRONG.setdefault(correct, [])
    if wrong not in CORRECT_TO_WRONG[correct]:
        CORRECT_TO_WRONG[correct].append(wrong)

# 对 ASR 高频词再手工补一些（nlu_parser 里只列了 2 字，我们这里可以加 3 字的错）
EXTRA: Dict[str, List[str]] = {
    "客厅空调": ["客听空调", "克厅空调", "客厅空条", "克厅空条"],
    "卧室空调": ["卧试空调", "卧世空调", "卧室空条", "卧试空条", "我是空调"],
    "客厅灯": ["客听灯", "克厅灯"],
    "卧室灯": ["卧试灯", "卧世灯"],
    "客厅窗帘": ["客听窗帘", "克厅窗连", "客厅窗连", "克厅窗帘"],
    "卧室窗帘": ["卧试窗帘", "卧世窗帘", "卧室窗连", "卧试窗连"],
    "客厅电视": ["客听电视", "克厅电世", "客厅电试", "克厅电视"],
    "卧室电视": ["卧试电视", "卧世电试", "卧室电是"],
    "温度": ["问度", "温渡", "问渡"],
    "风速": ["分速", "风束", "风诉"],
    "亮度": ["粮度", "明渡"],
}
for k, vs in EXTRA.items():
    CORRECT_TO_WRONG.setdefault(k, [])
    for v in vs:
        if v not in CORRECT_TO_WRONG[k]:
            CORRECT_TO_WRONG[k].append(v)

# 按长度从长→短，保证"客厅空调"先替换，再替换"客厅"，不会被拆开。
PHONETIC_POSITIONS = sorted(CORRECT_TO_WRONG.keys(), key=lambda s: -len(s))

ORAL_PREFIXES = ["帮我", "麻烦", "麻烦你", "请", "请你", "给我", "请问", "唉", "那个"]
ORAL_SUFFIXES = ["一下", "哦", "啊", "好不好", "行不", "呗", "吧", "哈", "谢谢", "麻烦了"]

_CN_DIGIT = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}


def num_to_chinese(n: int) -> str:
    if 0 <= n <= 9:
        return _CN_DIGIT[str(n)]
    if 10 <= n <= 19:
        return "十" + ("" if n == 10 else _CN_DIGIT[str(n % 10)])
    if 20 <= n <= 39:
        s = _CN_DIGIT[str(n // 10)] + "十"
        if n % 10 != 0:
            s += _CN_DIGIT[str(n % 10)]
        return s
    return str(n)  # 超出范围就不换了，保底


def apply_phonetic(text: str, p_per_slot: float = 0.35) -> str:
    """对 text 做同音错词替换：每个可替换的词，按 p_per_slot 概率替换成错词。"""
    for key in PHONETIC_POSITIONS:
        if key not in text:
            continue
        wrongs = CORRECT_TO_WRONG[key]
        if not wrongs:
            continue
        # 按每个出现位置独立决策（避免把所有"客厅"一次性换掉）
        chunks = text.split(key)
        new_chunks: List[str] = [chunks[0]]
        for c in chunks[1:]:
            if random.random() < p_per_slot:
                new_chunks.append(random.choice(wrongs))
            else:
                new_chunks.append(key)
            new_chunks.append(c)
        text = "".join(new_chunks)
    return text


def apply_oral(text: str, p_prefix: float = 0.3, p_suffix: float = 0.3) -> str:
    if random.random() < p_prefix:
        text = random.choice(ORAL_PREFIXES) + text
    if random.random() < p_suffix:
        text = text + random.choice(ORAL_SUFFIXES)
    return text


def apply_punct(text: str, p: float = 0.4) -> str:
    # 随机替换/加逗号，句号，空格
    if random.random() < p:
        # 把已经有的逗号换成顿号，或在 打开/关闭/调到 后随机加逗号
        text = text.replace("，", ",").replace(",", random.choice(["，", "、", " "] * 3 + [","]))
        # 把 "，" 再放回大部分
        text = text.replace(",", "，")
    if random.random() < 0.2:
        text = text + random.choice(["。", "？", ""])
    # 去掉多余空字符
    text = re.sub(r"\s+", "", text)
    return text


def apply_num_chinese(text: str, p: float = 0.4) -> str:
    """把"26度" → "二十六度"，其它地方的数字不换。"""
    m = re.search(r"(\d+)度", text)
    if not m:
        return text
    if random.random() > p:
        return text
    n = int(m.group(1))
    return text[:m.start(1)] + num_to_chinese(n) + text[m.end(1):]


def apply_word_order(text: str, p: float = 0.2) -> str:
    """轻度语序扰动：'把X打开'↔'打开X'↔'X打开'，以及'把X调到Y度'↔'X调到Y度'。"""
    if random.random() > p:
        return text
    # 把XXX打开 / 帮我把XXX打开 / 把XXX关闭
    m = re.match(r"^(帮我|请|麻烦|麻烦你|给我)?把(.+?)(打开|开启|关掉|关闭|关一下)$", text)
    if m:
        x = m.group(2)
        v = m.group(3)
        forms = [
            f"{v}{x}",
            f"{x}{v}",
            f"把{x}{v}",
        ]
        # 如果有前缀，概率保留前缀
        if m.group(1) and random.random() < 0.5:
            forms = [m.group(1) + f for f in forms]
        return random.choice(forms)
    m = re.match(r"^把(.+?)调到(\d+度?|.+度?)$", text)
    if m:
        x = m.group(1)
        y = m.group(2)
        return random.choice([f"{x}调到{y}", f"把{x}调到{y}", f"{x}{y}"])
    return text


def perturb_one(text: str) -> str:
    """对一条 input 做一次完整扰动。"""
    t = apply_phonetic(text)
    t = apply_num_chinese(t)
    t = apply_word_order(t)
    t = apply_oral(t)
    t = apply_punct(t)
    return t.strip()


def main() -> None:
    in_path = os.path.join(ROOT, "finetune", "outputs", "seed_samples.json")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"没找到 {in_path}，先跑 01_build_seed_dataset.py。")

    with open(in_path, "r", encoding="utf-8") as f:
        seeds: List[Dict[str, Any]] = json.load(f)

    variants: List[Dict[str, Any]] = []
    for s in seeds:
        inp = s["input"]
        outp = s["output_json"]
        meta = {**s.get("meta", {}), "source": "perturbed", "original_input": inp}

        # 变体 0：原样保留一份（p=0.1 不加扰动，直接塞）
        if random.random() < 0.6:  # 60% 的种子保留一条"干净原句"
            variants.append({
                "input": inp,
                "output_json": deepcopy(outp),
                "meta": {**meta, "perturb_level": 0},
            })

        # 变体 1~k：做 k 份扰动（k=2 或 3，按随机）
        k = random.choice([2, 2, 3])
        seen = {inp}
        for i in range(k):
            pp = perturb_one(inp)
            tries = 0
            while pp in seen and tries < 3:
                pp = perturb_one(inp)
                tries += 1
            if pp == inp:
                continue
            seen.add(pp)
            variants.append({
                "input": pp,
                "output_json": deepcopy(outp),
                "meta": {**meta, "perturb_level": i + 1},
            })

    random.shuffle(variants)

    out_dir = os.path.join(ROOT, "finetune", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "perturbed_samples.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(variants, f, ensure_ascii=False, indent=2)

    print(f"[perturb] 种子 {len(seeds)} 条 → 扰动后 {len(variants)} 条 → {out_path}")


if __name__ == "__main__":
    main()
