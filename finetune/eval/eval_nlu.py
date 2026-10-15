"""
阶段 C：智能家居 NLU 模型评估脚本
=================================
对 finetune/outputs/nlu_eval.jsonl（150 条）跑 4 组推理后端，对比：
 1. Rule    (rule_based_nlu.rule_parse, 本地确定性)
 2. SF-API  (SiliconFlow OpenAI 兼容远端: api.siliconflow.cn/v1/chat/completions)
 3. Base    (本地基座: Ollama / SiliconFlow 同 API 协议，本地部署 Qwen2.5-1.5B-Instruct)
 4. SFT     (SFT 后: Ollama / SiliconFlow 同 API 协议，本地部署 qwen2.5-1.5b-nlu-q4)

输出 3 个主业务指标：
  - JSON Parse Rate: 模型输出能 json.loads 且含必要字段的比例
  - Intent Accuracy: (device_id == gold) AND (command == gold) 的准确率
  - Slot F1 (macro): 对 value / value_type / attrs 做 token 级 F1 并 macro 平均

Usage:
  # 1. 仅跑 Rule 基线：
  python -m finetune.eval.eval_nlu --backends rule --limit 50

  # 2. 跑全部 4 组：
  python -m finetune.eval.eval_nlu --backends rule,sfapi,base,sft --limit 150

  # 3. 指定 Ollama / SiliconFlow 的 base_url & model 名：
  python -m finetune.eval.eval_nlu \
      --backends rule,sfapi,base,sft \
      --sfapi-base-url "https://api.siliconflow.cn/v1" \
      --sfapi-model "Qwen/Qwen2.5-7B-Instruct" \
      --ollama-base-url "http://127.0.0.1:11434/v1" \
      --base-model "qwen2.5:1.5b-instruct-q4_0" \
      --sft-model  "qwen2.5-1.5b-nlu:q4_k_m" \
      --limit 150

结果打印到控制台，同时落盘 finetune/outputs/eval_nlu_report.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import statistics
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

# --- 允许作为 package 或直接脚本运行 ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 真实 rule NLU 接口：
#     nlu_parser.parse_intent(text) 返回包装层 {success, method, intent:{device_id,...}}
#     我们取 intent 内部字段作为“gold 的兄弟字段”对齐。
from nlu_parser import parse_intent as rule_parse_intent

def rule_parse(text: str) -> dict:
    """把 parse_intent 的外层包装剥成和 SFT 对齐的 JSON（gpt 输出的 dict 就是 intent 本身 + success/msg/confidence/need_clarification）。"""
    wrap = rule_parse_intent(text)
    intent = wrap.get("intent") or {}
    out = dict(intent)
    out["success"] = wrap.get("success", False)
    out["msg"] = wrap.get("msg", "")
    out["confidence"] = wrap.get("confidence", 0.0)
    out["need_clarification"] = wrap.get("need_clarification", False)
    # intent 里带了 raw_text 不是我们的输出格式，去掉避免干扰对比
    out.pop("raw_text", None)
    out.pop("control_mode", None)
    out.pop("delta", None)
    return out

NECESSARY_FIELDS = {"device_id", "device_name", "action"}  # success/msg/confidence 等也要求，但 rule/LLM 都给，核心 intent 字段就按这三项判


# ==============================================================
# 数据加载
# ==============================================================
def load_eval_rows(limit: int) -> List[Dict[str, Any]]:
    """读取 nlu_eval.jsonl，抽 gold 的 input + output_json。"""
    p = PROJECT_ROOT / "finetune" / "outputs" / "nlu_eval.jsonl"
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            conv = row["conversations"]
            inp = next(m["value"] for m in conv if m["from"] == "user")
            gpt = next(m["value"] for m in conv if m["from"] == "gpt")
            gold = json.loads(gpt)
            out.append({"input": inp, "gold": gold})
            if len(out) >= limit:
                break
    return out


# ==============================================================
# OpenAI 协议 POST（同一份代码打 SiliconFlow / Ollama）
# ==============================================================
def chat_completion_post(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = 60,
    temperature: float = 0.0,
) -> Tuple[Optional[str], Optional[str]]:
    """返回 (content_str_or_None, error_msg_or_None)。"""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 512,
    }
    data = json.dumps(body).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            choices = obj.get("choices") or []
            if not choices:
                return None, f"empty choices; raw={raw[:300]}"
            msg = choices[0].get("message") or {}
            c = msg.get("content")
            return (c.strip() if isinstance(c, str) else None), None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        return None, f"http_err={type(e).__name__}: {e}"
    except json.JSONDecodeError as e:
        return None, f"json_decode: {e}"


def try_parse_json_output(content: Optional[str]) -> Optional[Dict[str, Any]]:
    """抽取模型输出里的 JSON：优先整串 parse，失败就提取第一个 {...}。"""
    if not content:
        return None
    s = content.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            return None
    return None


def has_necessary_fields(o: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(o, dict):
        return False
    return NECESSARY_FIELDS.issubset(o.keys()) and all(isinstance(o[k], str) and o[k] for k in NECESSARY_FIELDS)


# ==============================================================
# 指标：JSON Parse Rate / Intent Acc / Slot F1 (macro)
# ==============================================================
# 真实 schema 里没有 "command"（SFT gpt 输出字段是 device_id/device_name/action/value/...）
# Intent Acc 判为：(device_id 相等) AND (action 相等)
def intent_match(pred: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    return (
        str(pred.get("device_id")) == str(gold.get("device_id"))
        and str(pred.get("action")) == str(gold.get("action"))
    )


def f1_precision_recall(tp: int, pred_pos: int, gold_pos: int) -> Tuple[float, float, float]:
    p = tp / pred_pos if pred_pos else 0.0
    r = tp / gold_pos if gold_pos else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return f, p, r


def slot_macro_f1(preds: List[Dict[str, Any]], golds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对 5 个业务槽位分别算 ExactMatch F1，再 macro 平均。"""
    slot_names = ["value", "value_type", "attrs", "action", "need_clarification"]
    results_per_slot: Dict[str, Dict[str, float]] = {}
    for sname in slot_names:
        tp = pp = gp = 0
        for pred, gold in zip(preds, golds):
            pv = pred.get(sname)
            gv = gold.get(sname)
            # None / 空串 / 空 dict 都视为"该槽位 absent"；attrs 用 sort_keys 序列化比较
            if sname == "attrs":
                pv_s = json.dumps(pv, ensure_ascii=False, sort_keys=True) if isinstance(pv, dict) else ""
                gv_s = json.dumps(gv, ensure_ascii=False, sort_keys=True) if isinstance(gv, dict) else ""
                pv, gv = pv_s, gv_s
            p_pos = pv is not None and not (isinstance(pv, str) and pv == "")
            g_pos = gv is not None and not (isinstance(gv, str) and gv == "")
            if p_pos:
                pp += 1
            if g_pos:
                gp += 1
            if p_pos and g_pos and pv == gv:
                tp += 1
        f, p, r = f1_precision_recall(tp, pp, gp)
        results_per_slot[sname] = {
            "f1": round(f, 4), "p": round(p, 4), "r": round(r, 4),
            "tp": tp, "pp": pp, "gp": gp,
        }
    return {
        "macro_f1": round(statistics.mean(v["f1"] for v in results_per_slot.values()), 4),
        "macro_p":  round(statistics.mean(v["p"]  for v in results_per_slot.values()), 4),
        "macro_r":  round(statistics.mean(v["r"]  for v in results_per_slot.values()), 4),
        "per_slot": results_per_slot,
    }


# ==============================================================
# 四个后端
# ==============================================================
# LLM 后端的 system prompt 必须和 SFT 训练/eval 数据里用的保持一致（04_build_jsonl.py 里是 from llm_nlu import SYSTEM_PROMPT）
from llm_nlu import SYSTEM_PROMPT  # noqa: E402


def run_rule(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    preds, errs = [], 0
    t0 = time.perf_counter()
    for r in rows:
        try:
            out = rule_parse(r["input"])
        except Exception as e:
            errs += 1
            out = {"parse_error": str(e)}
        preds.append(out)
    return _summarize(rows, preds, time.perf_counter() - t0, errs)


def run_openai_like(
    rows: List[Dict[str, Any]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    label: str,
) -> Dict[str, Any]:
    preds: List[Dict[str, Any]] = []
    errs = 0
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        content, err = chat_completion_post(base_url, api_key, model, SYSTEM_PROMPT, r["input"])
        if err:
            errs += 1
            preds.append({"_err": err})
            continue
        parsed = try_parse_json_output(content)
        if parsed is None:
            preds.append({"_raw": content})
        else:
            preds.append(parsed)
        if (i + 1) % 20 == 0:
            print(f"  [{label}] {i+1}/{len(rows)} done, elapsed {time.perf_counter()-t0:.1f}s", flush=True)
    return _summarize(rows, preds, time.perf_counter() - t0, errs)


def _summarize(rows: List[Dict[str, Any]], preds: List[Dict[str, Any]], elapsed: float, http_errors: int) -> Dict[str, Any]:
    assert len(rows) == len(preds)
    total = len(rows)
    json_ok = [p for p in preds if isinstance(p, dict) and has_necessary_fields(p)]
    parse_rate = len(json_ok) / total if total else 0.0

    valid_idx = [i for i, p in enumerate(preds) if isinstance(p, dict) and has_necessary_fields(p)]
    intent_correct = 0
    for i in valid_idx:
        if intent_match(preds[i], rows[i]["gold"]):
            intent_correct += 1
    intent_acc = intent_correct / total if total else 0.0  # 分母 = total（含 parse 失败即扣分）

    # Slot F1 只在 JSON parse 通过的样本上统计（否则全是错，会拉低但避免 0 除）
    sub_preds = [preds[i] for i in valid_idx]
    sub_golds = [rows[i]["gold"] for i in valid_idx]
    slot = slot_macro_f1(sub_preds, sub_golds) if sub_preds else {"macro_f1": 0.0, "macro_p": 0.0, "macro_r": 0.0, "per_slot": {}}

    return {
        "total": total,
        "elapsed_s": round(elapsed, 2),
        "http_errors": http_errors,
        "json_parse_rate": round(parse_rate, 4),
        "intent_acc": round(intent_acc, 4),
        "slot_macro_f1": round(slot["macro_f1"], 4),
        "slot_macro_p": round(slot["macro_p"], 4),
        "slot_macro_r": round(slot["macro_r"], 4),
        "slot_per_slot": slot["per_slot"],
        "samples": _sample_errors(rows, preds, k=3),
    }


def _sample_errors(rows: List[Dict[str, Any]], preds: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    """挑 parse 失败 / intent 错的样例各几条供人工排查。"""
    failures: List[Dict[str, Any]] = []
    for i, (r, p) in enumerate(zip(rows, preds)):
        ok = isinstance(p, dict) and has_necessary_fields(p)
        im = ok and intent_match(p, r["gold"])
        if not ok or not im:
            failures.append({"idx": i, "input": r["input"], "gold": r["gold"], "pred": p})
        if len(failures) >= k:
            break
    return failures


# ==============================================================
# CLI
# ==============================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="rule", help="逗号分隔：rule,sfapi,base,sft")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--sfapi-base-url", default=os.environ.get("SFAPI_BASE_URL", "https://api.siliconflow.cn/v1"))
    ap.add_argument("--sfapi-api-key", default=os.environ.get("SILICONFLOW_API_KEY", ""))
    ap.add_argument("--sfapi-model", default=os.environ.get("SFAPI_MODEL", "Qwen/Qwen2.5-7B-Instruct"))

    ap.add_argument("--ollama-base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"))
    ap.add_argument("--ollama-api-key", default=os.environ.get("OLLAMA_API_KEY", "ollama"))  # Ollama 可不填
    ap.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "qwen2.5:1.5b-instruct-q4_0"))
    ap.add_argument("--sft-model", default=os.environ.get("SFT_MODEL", "qwen2.5-1.5b-nlu:q4_k_m"))
    args = ap.parse_args()

    backends = [x.strip() for x in args.backends.split(",") if x.strip()]
    rows = load_eval_rows(args.limit)
    print(f"[eval_nlu] eval rows={len(rows)}; backends={backends}", flush=True)

    report: Dict[str, Any] = {"meta": {"limit": args.limit, "backends": backends, "eval_file": "finetune/outputs/nlu_eval.jsonl"}, "results": {}}

    if "rule" in backends:
        print(f"\n==== Backend: rule (rule_parse) ====", flush=True)
        r = run_rule(rows)
        _print_block("rule", r)
        report["results"]["rule"] = r

    if "sfapi" in backends:
        print(f"\n==== Backend: SF-API (SiliconFlow) model={args.sfapi_model} ====", flush=True)
        if not args.sfapi_api_key:
            print("  [!] skip sfapi: SILICONFLOW_API_KEY / --sfapi-api-key not set")
        else:
            r = run_openai_like(
                rows, base_url=args.sfapi_base_url, api_key=args.sfapi_api_key,
                model=args.sfapi_model, label="SF-API",
            )
            _print_block("SF-API", r)
            report["results"]["sfapi"] = r

    if "base" in backends:
        print(f"\n==== Backend: Base (Ollama base) model={args.base_model} ====", flush=True)
        r = run_openai_like(
            rows, base_url=args.ollama_base_url, api_key=args.ollama_api_key,
            model=args.base_model, label="Ollama-Base",
        )
        _print_block("Base", r)
        report["results"]["base"] = r

    if "sft" in backends:
        print(f"\n==== Backend: SFT (Ollama merged) model={args.sft_model} ====", flush=True)
        r = run_openai_like(
            rows, base_url=args.ollama_base_url, api_key=args.ollama_api_key,
            model=args.sft_model, label="Ollama-SFT",
        )
        _print_block("SFT", r)
        report["results"]["sft"] = r

    out_path = PROJECT_ROOT / "finetune" / "outputs" / "eval_nlu_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval_nlu] report written: {out_path}")
    return 0


def _print_block(label: str, r: Dict[str, Any]) -> None:
    print(f"  {label:8s} | parse_rate={r['json_parse_rate']:.2%}  "
          f"intent_acc={r['intent_acc']:.2%}  slot_f1_macro={r['slot_macro_f1']:.2%}  "
          f"elapsed={r['elapsed_s']}s  http_err={r['http_errors']}")
    for sname, m in (r.get("slot_per_slot") or {}).items():
        print(f"     slot {sname:11s}: F1={m['f1']:.2%}  P={m['p']:.2%}  R={m['r']:.2%}  "
              f"(TP={m['tp']} pred={m['pp']} gold={m['gp']})")
    errs = r.get("samples") or []
    for e in errs:
        print(f"   - sample idx={e['idx']} input={e['input']!r}")
        print(f"       gold={json.dumps(e['gold'], ensure_ascii=False)}")
        print(f"       pred={json.dumps(e['pred'], ensure_ascii=False, default=str)}")


if __name__ == "__main__":
    raise SystemExit(main())
