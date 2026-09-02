#!/usr/bin/env python3
"""I4: Distinguish OpenRouter expired/invalid key from zero balance.

Completions can return 401 for both. Auth/key + credits bodies differ.
Does not run a model sweep. At most one max_tokens=1 ping.
Does not write results/raw/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

DERIVED = REPO_ROOT / "results/derived"
OUT = DERIVED / "I4_openrouter_401.csv"
OUT_JSON = DERIVED / "I4_openrouter_401.json"

AUTH = "https://openrouter.ai/api/v1/auth/key"
CREDITS = "https://openrouter.ai/api/v1/credits"
CHAT = "https://openrouter.ai/api/v1/chat/completions"


def _trim(text: str, n: int = 500) -> str:
    s = " ".join(str(text or "").split())
    return s[:n]


def _err_blob(status: int, text: str) -> dict:
    out: dict = {"status": status, "body_preview": _trim(text)}
    try:
        j = json.loads(text)
    except json.JSONDecodeError:
        return out
    err = j.get("error") if isinstance(j, dict) else None
    if isinstance(err, dict):
        out["error_message"] = str(err.get("message") or "")
        out["error_code"] = err.get("code")
        out["error_type"] = str(err.get("type") or "")
    return out


def classify(*, key_present: bool, auth: dict, credits: dict, chat: dict | None) -> tuple[str, str]:
    if not key_present:
        return "missing_key", "OPENROUTER_API_KEY is not set after loading .env"
    a_st = int(auth.get("status") or 0)
    c_st = int(credits.get("status") or 0)
    ch_st = int((chat or {}).get("status") or 0)
    a_msg = str(auth.get("error_message") or auth.get("body_preview") or "").lower()
    c_msg = str(credits.get("error_message") or credits.get("body_preview") or "").lower()
    ch_msg = str((chat or {}).get("error_message") or (chat or {}).get("body_preview") or "").lower()
    joined = " ".join([a_msg, c_msg, ch_msg])

    expired_markers = (
        "invalid api key",
        "expired",
        "user not found",
        "no cookie auth",
        "missing authentication",
        "unauthorized",
        "key not found",
        "disabled",
        "revoked",
    )
    balance_markers = (
        "credits",
        "afford",
        "insufficient",
        "payment required",
        "can only afford",
        "max_tokens",
        "balance",
        "overdrawn",
    )

    wallet = credits.get("account_wallet")
    if a_st == 401 or (a_st != 200 and any(m in a_msg for m in expired_markers)):
        return (
            "expired_or_invalid_key",
            f"auth/key HTTP {a_st}; completions 401 here is the key, not the wallet",
        )
    if any(m in joined for m in expired_markers) and a_st != 200:
        return "expired_or_invalid_key", f"auth/key HTTP {a_st}"

    if c_st == 200 and wallet is not None and float(wallet) <= 0:
        cause = "zero_or_negative_balance"
        extra = f"wallet={wallet}"
        if ch_st == 402:
            return cause, f"{extra}; chat completions 402 (payment), not 401"
        if ch_st == 401:
            return cause, (
                f"{extra}; chat completions also 401 — same status as a bad key; "
                "credits endpoint is the discriminator"
            )
        return cause, extra

    if ch_st == 402 or any(m in ch_msg for m in balance_markers):
        return "zero_or_negative_balance", f"chat HTTP {ch_st}; {ch_msg[:180]}"

    if a_st == 200 and ch_st == 200:
        return "ok", "auth/key and a 1-token ping both succeeded"

    if ch_st == 401 and a_st == 200:
        return (
            "zero_or_negative_balance_or_key_scope",
            "auth/key 200 but chat 401 — not an expired key; inspect credits/wallet and chat body",
        )
    return "unknown", f"auth={a_st} credits={c_st} chat={ch_st}"


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    if load_dotenv:
        load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    auth_r = requests.get(AUTH, headers=headers, timeout=15) if key else None
    cred_r = requests.get(CREDITS, headers=headers, timeout=15) if key else None

    auth = _err_blob(auth_r.status_code, auth_r.text) if auth_r is not None else {"status": None, "body_preview": "no key"}
    credits = _err_blob(cred_r.status_code, cred_r.text) if cred_r is not None else {"status": None, "body_preview": "no key"}

    wallet = total_credits = total_usage = None
    key_limit = key_limit_remaining = None
    if auth_r is not None and auth_r.status_code == 200:
        try:
            kd = auth_r.json().get("data") or {}
            key_limit = kd.get("limit")
            key_limit_remaining = kd.get("limit_remaining")
        except Exception:
            pass
    if cred_r is not None and cred_r.status_code == 200:
        try:
            cd = cred_r.json().get("data") or {}
            total_credits = cd.get("total_credits")
            total_usage = cd.get("total_usage")
            if total_credits is not None and total_usage is not None:
                wallet = float(total_credits) - float(total_usage)
        except Exception:
            pass
    credits["account_wallet"] = wallet
    credits["total_credits"] = total_credits
    credits["total_usage"] = total_usage

    chat = None
    if key:
        ping = requests.post(
            CHAT,
            headers={**headers, "Content-Type": "application/json"},
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=15,
        )
        chat = _err_blob(ping.status_code, ping.text)

    cause, note = classify(key_present=bool(key), auth=auth, credits=credits, chat=chat)
    rec = {
        "cause": cause,
        "note": note,
        "auth_status": auth.get("status"),
        "auth_error_message": auth.get("error_message", ""),
        "credits_status": credits.get("status"),
        "credits_error_message": credits.get("error_message", ""),
        "account_wallet": wallet,
        "total_credits": total_credits,
        "total_usage": total_usage,
        "key_limit": key_limit,
        "key_limit_remaining": key_limit_remaining,
        "chat_status": (chat or {}).get("status"),
        "chat_error_message": (chat or {}).get("error_message", ""),
        "chat_body_preview": (chat or {}).get("body_preview", ""),
        "key_present": bool(key),
        "key_len": len(key),
        "key_prefix": (key[:7] + "…") if len(key) >= 8 else ("set" if key else ""),
    }
    payload = {
        "cause": cause,
        "note": note,
        "auth": {k: v for k, v in auth.items() if k != "raw"},
        "credits": credits,
        "chat": chat,
        "key_present": bool(key),
        "key_len": len(key),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    import pandas as pd

    pd.DataFrame([rec]).to_csv(OUT, index=False)
    print(f"cause={cause}")
    print(note)
    print(f"auth={auth.get('status')} credits={credits.get('status')} chat={(chat or {}).get('status')} wallet={wallet}")
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
