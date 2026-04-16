"""
scripts/test_api_keys.py

Phase 0, Step 4 — verify API access before proceeding.
Run this once after setting up your .env file.

Usage:
    python scripts/test_api_keys.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()  # reads from .env in project root

PASS = "✓"
FAIL = "✗"


def test_openai():
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Reply with the single word: hello"}],
            max_tokens=5,
        )
        reply = resp.choices[0].message.content.strip()
        print(f"  {PASS} OpenAI (GPT-4o): got '{reply}'")
        return True
    except KeyError:
        print(f"  {FAIL} OpenAI: OPENAI_API_KEY not set in .env")
    except Exception as e:
        print(f"  {FAIL} OpenAI: {e}")
    return False


def test_anthropic():
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=5,
            messages=[{"role": "user", "content": "Reply with the single word: hello"}],
        )
        reply = resp.content[0].text.strip()
        print(f"  {PASS} Anthropic (Claude): got '{reply}'")
        return True
    except KeyError:
        print(f"  {FAIL} Anthropic: ANTHROPIC_API_KEY not set in .env")
    except Exception as e:
        print(f"  {FAIL} Anthropic: {e}")
    return False


def test_infinigram():
    """Infini-gram is a free public API — no key needed. Just check reachability."""
    try:
        import requests
        payload = {
            "index": "v4_pileval_gpt2",  # The Pile index
            "query_type": "count",
            "query": "the quick brown fox",
        }
        resp = requests.post("https://api.infini-gram.io/", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        count = data.get("count", "?")
        print(f"  {PASS} Infini-gram API: reachable, test query count={count}")
        return True
    except Exception as e:
        print(f"  {FAIL} Infini-gram API: {e}")
    return False


if __name__ == "__main__":
    print("\n=== Phase 0 API Key Verification ===\n")

    results = {
        "openai": test_openai(),
        "anthropic": test_anthropic(),
        "infinigram": test_infinigram(),
    }

    print()
    all_ok = all(results.values())
    if all_ok:
        print("All checks passed. Gate 0 condition (API keys) met.\n")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"Failed: {failed}")
        print("Fix these before proceeding to Phase 1.\n")
        sys.exit(1)
