"""
本地 NLU Wrapper：Ollama (OpenAI 兼容协议) 版本
================================================
给 smart_home.py 的 NLU 控制器用，直接替换原先的 SiliconFlow 调用：

    from finetune.deploy.llm_nlu_local_ollama import OllamaNLUClient
    client = OllamaNLUClient(model="qwen2.5-1.5b-nlu:q4_k_m")
    out = client.parse("麻烦客厅空调开26度制冷呗")

如果 Ollama 没启动或出错，会自动 fallback 回 rule_parse（不影响主流程跑）。
支持带 cache（默认打开），避免同一句反复问模型。
"""
from __future__ import annotations

import functools
import json
import os
import pathlib
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- fallback: 复用真实 rule NLU（保持和 smart_home.py 的 parse_intent 一致）---
from nlu_parser import parse_intent as rule_parse_intent

def rule_parse(text: str) -> dict:
    """把 parse_intent 的外层包装剥成 SFT 输出 JSON（和 Ollama 的 JSON 输出同 schema，便于比较）。"""
    wrap = rule_parse_intent(text)
    intent = wrap.get("intent") or {}
    out = dict(intent)
    out["success"] = wrap.get("success", False)
    out["msg"] = wrap.get("msg", "")
    out["confidence"] = wrap.get("confidence", 0.0)
    out["need_clarification"] = wrap.get("need_clarification", False)
    out.pop("raw_text", None)
    out.pop("control_mode", None)
    out.pop("delta", None)
    return out

# --- system prompt 直接复用 SFT 训练时用的（llm_nlu.SYSTEM_PROMPT）——
#     训练/推理 system 完全对齐，避免 domain drift
from llm_nlu import SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT  # noqa: E402


def _post_chat(base_url: str, api_key: str, model: str, system: str, user: str, timeout: int = 60) -> Optional[str]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
            c = ((obj.get("choices") or [{}])[0].get("message") or {}).get("content")
            return c.strip() if isinstance(c, str) else None
    except Exception:
        return None


def _extract_json(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    try:
        return json.loads(s.strip())
    except Exception:
        pass
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(s[a : b + 1])
        except Exception:
            return None
    return None


def _validate(o: Dict[str, Any]) -> bool:
    if not isinstance(o, dict):
        return False
    for k in ("device_id", "device_name", "command", "action"):
        if not isinstance(o.get(k), str) or not o[k]:
            return False
    return True


class OllamaNLUClient:
    """Ollama NLU 客户端；失败自动 fallback 到 rule_parse。"""

    def __init__(
        self,
        model: str = "qwen2.5-1.5b-nlu:q4_k_m",
        base_url: str | None = None,
        api_key: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        use_cache: bool = True,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY", "ollama")
        self.system_prompt = system_prompt
        self.timeout = timeout
        self._cache: Dict[str, Dict[str, Any]] = {} if use_cache else None  # type: ignore[assignment]
        self.stats = {"llm_hit": 0, "llm_miss": 0, "llm_fallback": 0, "fallback_reasons": []}

    # --- Public API（和 smart_home_nlu.SiliconFlowNLUClient 保持一致）---
    def parse(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return rule_parse("")
        if self._cache is not None and text in self._cache:
            self.stats["llm_hit"] += 1
            return self._cache[text]
        out = self._call_llm(text)
        if out is None:
            self.stats["llm_fallback"] += 1
            out = rule_parse(text)
            if self._cache is not None:
                self._cache[text] = out
            self.stats["llm_miss"] += 1
            return out
        self.stats["llm_miss"] += 1
        if self._cache is not None:
            self._cache[text] = out
        return out

    # --- Internals ---
    def _call_llm(self, text: str) -> Optional[Dict[str, Any]]:
        content = _post_chat(self.base_url, self.api_key, self.model, self.system_prompt, text, timeout=self.timeout)
        parsed = _extract_json(content)
        if _validate(parsed):  # type: ignore[arg-type]
            return parsed  # type: ignore[return-value]
        self.stats["fallback_reasons"].append(
            "parse_fail" if parsed is None else "schema_missing"
        )
        if len(self.stats["fallback_reasons"]) > 200:
            self.stats["fallback_reasons"] = self.stats["fallback_reasons"][-100:]
        return None


# ==============================================================
# 自测：直接运行本文件
#   python -m finetune.deploy.llm_nlu_local_ollama
# ==============================================================
if __name__ == "__main__":
    c = OllamaNLUClient()
    for q in [
        "客厅空调开26度制冷",
        "卧室窗帘拉开一半",
        "把客厅灯调暗点",
        "哎呀卧室的空条关一下",
    ]:
        t0 = time.perf_counter()
        r = c.parse(q)
        dt = time.perf_counter() - t0
        print(f"Q: {q!r}")
        print(f"  -> {json.dumps(r, ensure_ascii=False)}  ({dt*1000:.0f}ms)")
    print("stats:", c.stats)
