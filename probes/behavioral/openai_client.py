"""
OpenRouter client covering GPT-4o, o3, and other 
closed models behind one key. Dormant until OPENROUTER_API_KEY is set in .env.
Interface identical to MockClient.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from probes.behavioral.sampling import DEFAULT_TEMPERATURE


# Optional shared budget logger. If RVC_RUN_NAME is set, every API call is
# appended to a JSONL log under logs/api_runs/.
_BUDGET = None
if os.environ.get("RVC_RUN_NAME"):
    try:
        _RVC_ROOT = Path(__file__).resolve().parents[2]
        if str(_RVC_ROOT) not in sys.path:
            sys.path.insert(0, str(_RVC_ROOT))
        from scripts.runs.api_budget import BudgetLogger
        _BUDGET = BudgetLogger(run_name=os.environ["RVC_RUN_NAME"])
    except Exception:
        _BUDGET = None


class OpenRouterClient:
    def __init__(
        self,
        model: str = "openai/gpt-4o",
        *,
        max_tokens: int | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int | None = None,
    ) -> None:
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Add it to .env before running the sweep.")
        
        self.model = model
        self.temperature = float(temperature)
        self.seed = seed
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        # Reasoning models default to very high max_tokens on OpenRouter; cap to avoid 402.
        if max_tokens is not None:
            self.max_tokens = max_tokens
        elif any(x in model.lower() for x in ("o1", "o3", "o4")):
            self.max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "4096"))
        else:
            self.max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "8192"))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True
    )
    def _make_api_call(self, prompt: str) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        if not response.ok:
            try:
                body = response.json()
                detail = body.get("error", {}).get("message") or str(body)[:400]
            except Exception:
                detail = (response.text or "")[:400]
            raise requests.HTTPError(
                f"{response.status_code} {response.reason} for {response.url}: {detail}",
                response=response,
            )
        return response.json()

    def complete(self, problem_id: str, prompt: str, **kwargs: Any) -> dict:
        t0 = time.time()
        try:
            data = self._make_api_call(prompt)

            choices = data.get("choices", [])
            response_text = ""
            if choices and isinstance(choices, list):
                content = choices[0].get("message", {}).get("content", "")
                response_text = content if content is not None else ""

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            if _BUDGET is not None:
                _extra = {"problem_id": problem_id, "len_resp": len(response_text)}
                if os.environ.get("RVC_LOG_RAW", "0") == "1":
                    _extra["raw_response"] = response_text[:8000]
                _BUDGET.record(
                    self.model, prompt_tokens, completion_tokens,
                    status="ok", latency_s=time.time() - t0,
                    extra=_extra,
                )
            return {
                "response": response_text,
                "model": self.model,
                "problem_id": problem_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "temperature": self.temperature,
                "seed": self.seed,
            }
        except Exception as e:
            if _BUDGET is not None:
                _BUDGET.record(
                    self.model, 0, 0,
                    status="error", latency_s=time.time() - t0,
                    extra={"problem_id": problem_id, "err": str(e)[:240]},
                )
            return {
                "response": f"ERROR: {str(e)}",
                "model": self.model,
                "problem_id": problem_id,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "temperature": self.temperature,
                "seed": self.seed,
            }

    def complete_batch(self, problems: list[dict], **kwargs: Any) -> list[dict]:
        # TODO: replace with async batching when throughput matters.
        results = []
        for prob in problems:
            pid = prob.get("problem_id", "unknown")
            prompt = prob.get("prompt", "")
            results.append(self.complete(pid, prompt, **kwargs))
        return results
