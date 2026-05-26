"""Shared API-budget tracker and per-call JSONL logger.

Usage:
    from scripts.runs.api_budget import BudgetLogger
    log = BudgetLogger(run_name="bw_p1_o4mini")
    # wrap an OpenRouter or Anthropic call:
    log.record(model, prompt_tokens, completion_tokens, status=..., latency=..., extra={"pid":...})
    log.live_remaining()    # hits OpenRouter for the latest limit_remaining
    log.summary()           # prints aggregate
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "logs" / "api_runs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


_PRICE_PER_M_TOKENS = {
    "openai/gpt-4o":                       (2.50, 10.00),
    "openai/o4-mini":                      (1.10, 4.40),
    "anthropic/claude-sonnet-4":           (3.00, 15.00),
    "meta-llama/llama-3.1-8b-instruct":    (0.05, 0.10),
    "google/gemini-2.5-flash":             (0.30, 2.50),
}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class BudgetLogger:
    """Append-only JSONL per-call log; thread-safe.

    The file is sized at one call per line. A periodic `live_remaining()`
    is appended whenever invoked so we can later reconstruct exact budget
    consumption from logs alone.
    """

    def __init__(self, run_name: str) -> None:
        self.run_name = run_name
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = LOG_DIR / f"{run_name}_{ts}.jsonl"
        self.lock = threading.Lock()
        self.tot_prompt = 0
        self.tot_completion = 0
        self.tot_dollars = 0.0
        self.tot_calls = 0
        self.tot_errors = 0
        self._write({"event": "run_start", "run": run_name, "ts": _ts()})

    def _write(self, rec: dict) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        status: str = "ok",
        latency_s: float = 0.0,
        extra: dict | None = None,
    ) -> None:
        pt = int(prompt_tokens or 0)
        ct = int(completion_tokens or 0)
        in_price, out_price = _PRICE_PER_M_TOKENS.get(model, (0.0, 0.0))
        dollars = (pt / 1_000_000) * in_price + (ct / 1_000_000) * out_price
        with self.lock:
            self.tot_prompt += pt
            self.tot_completion += ct
            self.tot_dollars += dollars
            self.tot_calls += 1
            if status != "ok":
                self.tot_errors += 1
        rec = {
            "event": "call",
            "ts": _ts(),
            "model": model,
            "status": status,
            "latency_s": round(latency_s, 3),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "dollars": round(dollars, 6),
        }
        if extra:
            rec["extra"] = extra
        self._write(rec)

    def live_remaining(self) -> float | None:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            return None
        try:
            r = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            if r.status_code != 200:
                return None
            d = r.json()["data"]
            rec = {
                "event": "budget",
                "ts": _ts(),
                "limit_remaining": d.get("limit_remaining"),
                "usage": d.get("usage"),
            }
            self._write(rec)
            return d.get("limit_remaining")
        except Exception:
            return None

    def summary(self) -> dict:
        rec = {
            "event": "run_summary",
            "ts": _ts(),
            "run": self.run_name,
            "calls": self.tot_calls,
            "errors": self.tot_errors,
            "prompt_tokens": self.tot_prompt,
            "completion_tokens": self.tot_completion,
            "dollars_estimated": round(self.tot_dollars, 4),
        }
        self._write(rec)
        return rec
