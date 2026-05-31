"""Verify the prompt-aware target-token resolver picks the token Qwen would
ACTUALLY predict, across all four prompt modes.

Runs ONLY the tokenizer (no GPU, no model weights) — safe to run locally.
"""

from __future__ import annotations
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B"  # base tokenizer (identical to -Instruct for our purposes)


def make_resolver(tokenizer):
    def _hf_encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    def resolve(prompt: str, answer: str) -> tuple[int, str, str]:
        if not answer.strip():
            return -1, "", ""
        prompt_ids = _hf_encode(prompt)
        candidates = []
        for sep in ("", " "):
            joint = _hf_encode(prompt + sep + answer)
            if len(joint) <= len(prompt_ids):
                continue
            if joint[: len(prompt_ids)] != prompt_ids:
                continue
            tid = int(joint[len(prompt_ids)])
            candidates.append((sep, tid, len(joint)))
        if not candidates:
            bare = _hf_encode(answer)
            if not bare:
                return -1, "", ""
            return int(bare[0]), tokenizer.decode([bare[0]]), "FALLBACK"
        candidates.sort(key=lambda c: c[2])
        sep, tid, _ = candidates[0]
        return tid, tokenizer.decode([tid]), repr(sep)

    return resolve


def main():
    tk = AutoTokenizer.from_pretrained(MODEL)
    resolve = make_resolver(tk)

    # The three families' representative examples
    cases = [
        ("gsm", "Alice has 3 apples. Bob gives her 2 more. How many apples does Alice have?", "5"),
        ("gsm", "A call costs $0.50/min. How much for 86 minutes?", "43.00"),
        ("algo", "Find min coins for 11 with [1,5,10].", "Take coin 10"),
        ("algo", "Shortest path from A to D.", "A B D"),
        ("bw", "Initial: A on table, B on A. Goal: A on B.", "Unstack B from A"),
    ]

    prompt_modes = {
        "raw": lambda text: text,
        "raw-qa": lambda text: f"Problem: {text}\n\nAnswer:\n",   # trailing newline = clean BPE boundary
        # chat templates would normally apply, but for this offline test we
        # simulate the trailing context they produce.
        "chat-direct-tail": lambda text: f"{text}<|im_end|>\n<|im_start|>assistant\n",
    }

    print(f"Tokenizer: {MODEL}")
    print(f"{'family':<6} {'mode':<22} {'answer':<22} {'tok_id':>7}  decoded  | sep")
    print("-" * 110)
    for fam, text, ans in cases:
        for mode_name, mode_fn in prompt_modes.items():
            prompt = mode_fn(text)
            tid, dec, sep = resolve(prompt, ans)
            print(f"{fam:<6} {mode_name:<22} {ans[:20]:<22} {tid:>7}  {dec!r:<10} | sep={sep}")
        print()


if __name__ == "__main__":
    main()
