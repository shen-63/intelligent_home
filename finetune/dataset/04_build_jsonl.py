"""
【微调 · 阶段 A-4】合并 → 去重 → 切分 → 产出 ShareGPT JSONL

输入（只要存在就合并，缺少就跳过）：
  - finetune/outputs/seed_samples.json         （01）
  - finetune/outputs/perturbed_samples.json    （02）
  - finetune/outputs/llm_augment_samples.json  （03，可选）

产出（ShareGPT / LLaMA-Factory 通用）：
  - finetune/outputs/nlu_train.jsonl  ~1500 条
  - finetune/outputs/nlu_eval.jsonl   ~150 条（固定 seed=42，保证每次切分一致）

每条 JSONL 形如：
{
  "conversations": [
    {"from": "system", "value": "<和 llm_nlu.SYSTEM_PROMPT 完全一致>"},
    {"from": "user",   "value": "用户指令文本"},
    {"from": "gpt",    "value": "<标准 JSON 字符串，不带 ```json>"}
  ]
}

ShareGPT 格式是 LLaMA-Factory / Unsloth 两个框架都原生支持的，后面阶段 B 直接
`dataset_info: {dataset: nlu_train, template: qwen, file_name: ...}` 就能训练。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from llm_nlu import SYSTEM_PROMPT  # 关键：训练时的 system prompt = 推理时的 system prompt，绝对不能改

random.seed(42)


def load_json_safe(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except Exception as e:
        print(f"[warn] 解析 {path} 失败：{e}")
    return []


def dedup(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        inp = (s.get("input") or "").strip()
        if not inp:
            continue
        # 同一个 input 有多条不同 output 时，保留优先级：
        #   seed > perturbed(p_level 低) > llm_augment
        src = (s.get("meta") or {}).get("source")
        p_lvl = (s.get("meta") or {}).get("perturb_level", 99)
        if not isinstance(p_lvl, int):
            p_lvl = 99
        rank = {"seed_template": 0, "perturbed": 1, "llm_augment": 2}.get(src or "", 9)
        key = inp
        if key in seen:
            old = seen[key]
            old_meta = old.get("meta") or {}
            old_rank = {"seed_template": 0, "perturbed": 1, "llm_augment": 2}.get(old_meta.get("source"), 9)
            old_plvl = old_meta.get("perturb_level", 99)
            if (rank, p_lvl) < (old_rank, old_plvl):
                seen[key] = s
        else:
            seen[key] = s
    return list(seen.values())


def sample_balanced(samples: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """按 (device_id, action) 分层抽样，避免训练集被 80% 温度样本占满。"""
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        out = s.get("output_json") or {}
        key = (str(out.get("device_id") or "?"), str(out.get("action") or "?"))
        buckets[key].append(s)

    keys = list(buckets.keys())
    per_bucket_target = max(1, n // max(1, len(keys)))
    picked: List[Dict[str, Any]] = []
    for k in keys:
        lst = buckets[k]
        random.shuffle(lst)
        take = min(len(lst), per_bucket_target)
        picked.extend(lst[:take])

    # 不够再从剩余里随机补
    rest = []
    picked_keys = {id(p) for p in picked}
    for lst in buckets.values():
        rest.extend([x for x in lst if id(x) not in picked_keys])
    random.shuffle(rest)
    remain = n - len(picked)
    if remain > 0:
        picked.extend(rest[:remain])
    random.shuffle(picked)
    return picked


def to_sharegpt(sample: Dict[str, Any]) -> Dict[str, Any]:
    out = sample.get("output_json") or {}
    gpt_value = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "user", "value": (sample.get("input") or "").strip()},
            {"from": "gpt", "value": gpt_value},
        ]
    }


def coverage_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    device_c = Counter()
    action_c = Counter()
    vt_c = Counter()
    for s in samples:
        out = s.get("output_json") or {}
        device_c[out.get("device_id") or "?"] += 1
        action_c[out.get("action") or "?"] += 1
        vt = out.get("value_type")
        if vt:
            vt_c[vt] += 1
    return {"device": device_c, "action": action_c, "value_type": vt_c}


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--eval", type=int, default=150)
    ap.add_argument("--skip_llm_augment", action="store_true",
                    help="即使有 03_llm_augment_samples.json 也不合并（防止 LLM 合成数据引入未知幻觉）")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "finetune", "outputs")
    seed = load_json_safe(os.path.join(out_dir, "seed_samples.json"))
    pert = load_json_safe(os.path.join(out_dir, "perturbed_samples.json"))
    aug = load_json_safe(os.path.join(out_dir, "llm_augment_samples.json")) if not args.skip_llm_augment else []

    print(f"[build] seed={len(seed)}  perturbed={len(pert)}  llm_augment={len(aug)}")
    merged = seed + pert + aug
    print(f"[build] 合并后 {len(merged)} 条")

    merged = dedup(merged)
    print(f"[build] 去重后 {len(merged)} 条")

    if len(merged) < max(args.train, 500):
        print("[build] ⚠️ 样本不够（建议至少 500 条再训，至少 1200 条稳妥）。"
              "可以再跑 02_perturb_asr.py 或 03_llm_augment.py 补量。")

    # —— 切分 train/eval：按 (device_id, action) 分层，保证 eval 每种组合至少有例子 ——
    train_pool = sample_balanced(merged, max(args.train + args.eval * 3, 1600)) if len(merged) >= args.train + args.eval else merged
    # 从 train_pool 里拿最后 args.eval 条做 eval（但先按 (dev,act) 分层抽 eval，更科学）
    by_bucket: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for s in train_pool:
        out = s.get("output_json") or {}
        key = (str(out.get("device_id") or "?"), str(out.get("action") or "?"))
        by_bucket[key].append(s)
    eval_samples: List[Dict[str, Any]] = []
    train_samples: List[Dict[str, Any]] = []
    eval_per_bucket = max(1, args.eval // max(1, len(by_bucket)))
    for k, lst in by_bucket.items():
        random.shuffle(lst)
        eval_samples.extend(lst[:eval_per_bucket])
        train_samples.extend(lst[eval_per_bucket:])
    # 若 eval 不够，再从 train_samples 末尾补
    if len(eval_samples) < args.eval:
        random.shuffle(train_samples)
        need = args.eval - len(eval_samples)
        eval_samples.extend(train_samples[:need])
        train_samples = train_samples[need:]
    # train 不够 1500，就从 merged 的剩余里补（但 eval 已经拿出来了，所以用 merged - eval_samples 的差集）
    eval_keys = {s["input"] for s in eval_samples}
    rest = [s for s in merged if s["input"] not in eval_keys and s not in train_samples]
    random.shuffle(rest)
    if len(train_samples) < args.train:
        need = args.train - len(train_samples)
        train_samples.extend(rest[:need])
    random.shuffle(train_samples)
    random.shuffle(eval_samples)

    print(f"[build] 切分后 train={len(train_samples)}  eval={len(eval_samples)}")

    # 覆盖率统计
    for name, ss in [("train", train_samples), ("eval", eval_samples)]:
        stats = coverage_stats(ss)
        print(f"\n== {name} 覆盖率 ==")
        print(" device_id:")
        for k, v in stats["device"].most_common():
            print(f"   · {k:<24s} {v:>4d}")
        print(" action:")
        for k, v in stats["action"].most_common():
            print(f"   · {k:<16s} {v:>4d}")
        print(" value_type:")
        for k, v in stats["value_type"].most_common():
            print(f"   · {k:<16s} {v:>4d}")

    # 写成 JSONL
    train_sg = [to_sharegpt(s) for s in train_samples]
    eval_sg = [to_sharegpt(s) for s in eval_samples]

    train_path = os.path.join(out_dir, "nlu_train.jsonl")
    eval_path = os.path.join(out_dir, "nlu_eval.jsonl")
    write_jsonl(train_path, train_sg)
    write_jsonl(eval_path, eval_sg)

    # spot-check：打印前 3 条 train，便于肉眼检查
    print("\n== train spot-check (first 3) ==")
    for s in train_samples[:3]:
        print(" USER:", s["input"])
        print("  GPT:", json.dumps(s["output_json"], ensure_ascii=False, indent=2).replace("\n", "\n       "))
        print()

    print(f"\n[DONE] 训练集 {len(train_sg)} → {train_path}")
    print(f"[DONE] 评估集 {len(eval_sg)}  → {eval_path}")


if __name__ == "__main__":
    main()
