"""Model clients. One real client (any OpenAI-compatible endpoint) and deterministic mocks.

A call never raises. It returns a record with status "ok" or "error"; errors are never scored.
"""

from __future__ import annotations

import hashlib
import os
import random
import time

import requests

from . import verify

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


class ChatClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base_url = cfg["base_url"].rstrip("/")
        key_env = cfg.get("api_key_env")
        self.api_key = os.environ.get(key_env, "") if key_env else ""
        if key_env and not self.api_key:
            raise RuntimeError(f"{key_env} is not set. Put it in .env (see env.example).")

    def payload(self, prompt: str, thinking: str, sample: int) -> dict:
        c = self.cfg
        body = {"model": c["id"], "messages": [{"role": "user", "content": prompt}], "max_tokens": int(c["max_tokens"])}
        if c.get("temperature") is not None:
            body["temperature"] = float(c["temperature"])
        if c.get("top_p") is not None:
            body["top_p"] = float(c["top_p"])
        if c.get("send_seed", False):
            body["seed"] = int(c.get("seed_base", 0)) + sample
        extra = (c.get("thinking") or {}).get(thinking)
        if extra is None:
            raise ValueError(f"model {c['key']} has no thinking={thinking!r} setting")
        body.update(extra)
        body.update(c.get("extra_body") or {})
        return body

    def complete(self, prompt: str, thinking: str, sample: int, **_ignored) -> dict:
        body = self.payload(prompt, thinking, sample)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        attempts, last_err, t0 = int(self.cfg.get("max_retries", 6)), "", time.time()
        for attempt in range(attempts):
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=body,
                                     timeout=float(self.cfg.get("timeout_s", 900)))
            except requests.RequestException as exc:
                last_err = f"transport: {type(exc).__name__}: {exc}"[:400]
            else:
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError:
                        last_err = "http 200 but body is not JSON"
                    else:
                        if data.get("error"):
                            last_err = f"provider error: {str(data['error'])[:400]}"
                        else:
                            return self._ok(data, body, time.time() - t0, attempt)
                else:
                    last_err = f"http {resp.status_code}: {resp.text[:400]}"
                    if resp.status_code not in RETRY_STATUS:
                        break
            time.sleep(min(60.0, 2.0 ** attempt) * (0.5 + random.random()))
        return {"status": "error", "error": last_err, "request": _public(body), "latency_s": round(time.time() - t0, 3)}

    @staticmethod
    def _ok(data: dict, body: dict, latency: float, attempt: int) -> dict:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):  # some providers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        usage = data.get("usage") or {}
        return {
            "status": "ok", "response": content or "",
            "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason") or choice.get("native_finish_reason") or "",
            "usage": {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                      "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")},
            "served_model": data.get("model", ""), "provider": data.get("provider", ""),
            "response_id": data.get("id", ""), "system_fingerprint": data.get("system_fingerprint", ""),
            "request": _public(body), "latency_s": round(latency, 3), "retries": attempt,
        }


def _public(body: dict) -> dict:
    return {k: v for k, v in body.items() if k != "messages"}


class MockClient:
    """Scripted behaviours for offline end-to-end tests. Deterministic.

    oracle        always the gold answer with the marker
    unmarked      the gold answer without the marker
    empty         empty response
    wrong         a well-formed but wrong answer
    truncated     gold answer but finish_reason=length
    error         every call fails
    noisy:P       gold with probability P (per call, seeded by call key), else wrong
    variant_drop  gold on canonical; on other variants gold with probability 0.6
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mode = cfg["id"].split("/", 1)[1]

    def _coin(self, key: str, p: float) -> bool:
        h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return h < p

    def complete(self, prompt: str, thinking: str, sample: int, item: dict | None = None, scripted: str | None = None,
                 arm: str = "main", **_):
        mode = self.mode
        if mode == "error":
            return {"status": "error", "error": "mock error", "request": {}, "latency_s": 0.0}
        if scripted is not None:
            text = scripted
        else:
            arm = arm if self.mode == "variant_drop" or arm == "plan" else "main"
            gold = verify.gold_response(item)
            wrong = "FINAL ANSWER:\n" + _wrong_block(item)
            key = f"{item['item_id']}|{thinking}|{sample}"
            if mode == "oracle":
                text = "Reasoning goes here.\n" + gold
            elif mode == "unmarked":
                text = "Here is my answer.\n" + item["gold_block"]
            elif mode == "empty":
                text = ""
            elif mode == "wrong":
                text = wrong
            elif mode == "truncated":
                text = gold
            elif mode.startswith("noisy:"):
                text = gold if self._coin(key, float(mode.split(":")[1])) else wrong
            elif mode == "variant_drop":
                p = {"main": 1.0 if item["variant"] == "canonical" else 0.6, "plan": 0.9, "control_true": 0.95,
                     "inject_false": 0.4, "inject_cued": 0.8}[arm]
                text = gold if self._coin(key + arm, p) else wrong
            else:
                raise ValueError(f"unknown mock mode {mode}")
            if arm == "plan" and text:
                text = "Step 1: first quantity = 12\nStep 2: second quantity = 30\nStep 3: combine = 42\n" + text
        return {"status": "ok", "response": text, "reasoning": "", "finish_reason": "length" if mode == "truncated" else "stop",
                "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4, "reasoning_tokens": 0},
                "served_model": self.cfg["id"], "provider": "mock", "response_id": "", "system_fingerprint": "",
                "request": {"model": self.cfg["id"]}, "latency_s": 0.0, "retries": 0}


def _wrong_block(item: dict) -> str:
    kind = item["spec"]["kind"]
    if kind == "numeric":
        return f"{float(item['spec']['answer']) + 1:g}"
    if kind == "plan":
        return "\n".join(item["gold_block"].splitlines()[:-1]) or "1. nothing"
    if kind == "cc":
        lines = item["gold_block"].splitlines()
        return lines[0] + "\nValues: 0"
    return item["gold_block"].rsplit(":", 1)[0] + ": 999999"


def make_client(cfg: dict):
    return MockClient(cfg) if cfg["id"].startswith("mock/") else ChatClient(cfg)
