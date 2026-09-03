#!/usr/bin/env python3
"""Build o16_open_model_calibration.ipynb (Colab T4) — O16 Part B.

Scores every canonical problem with Pythia-2.8B (Pile) and OLMo-2-1B (Dolma)
using both O5 (teacher-forced gold under Probe-1 prompt) and O15 (statement
surprisal / min-k%) measures. Output feeds O16 Part C calibration.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "_rvc_build_notebooks", OUT / "_build_notebooks.py"
)
assert _spec is not None and _spec.loader is not None
_bn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bn)
SETUP_PIP = _bn.SETUP_PIP
SETUP_REPO = _bn.SETUP_REPO
md = _bn.md
code = _bn.code
nb = _bn.nb


NB_O16 = [
    md("""# O16 Part B — Open-corpus model scores for GT calibration (Colab T4)

Part A searched **The Pile** and **Dolma** for exact/near-exact matches of every
canonical problem. Here we score those same instances with the models trained
on those corpora:

| Model | Corpus GT | dtype |
|-------|-----------|-------|
| `EleutherAI/pythia-2.8b` | Pile (`v4_piletrain_llama`) | fp16 |
| `allenai/OLMo-2-0425-1B` | Dolma (`v4_dolma-v1_7_llama`) | fp16 |

### Measures (canonical only)
1. **O15 statement surprisal:** mean per-token NLL of the problem statement; min-k% (k=20).
2. **O5 teacher-forced gold:** mean logprob / NLL of the bank gold under the Appendix-N Probe-1 prompt.

**T4:** fp16, `attn_implementation="sdpa"`.

**Output:** `colab_out/O16_open_model_scores.csv` → `results/raw/O16_open_model_scores.csv`

Then run:
```bash
python scripts/consolidate/o16_calibrate_proxies.py
```

**Paper caveat:** this GT calibration cannot be done for Claude, GPT-4o, Gemini,
o4-mini, or DeepSeek — a permanent limitation of contamination research on closed models."""),
    code(SETUP_PIP),
    code(
        SETUP_REPO
        + r'''

MIN_K_PCT = 20
'''
    ),
    md("""## Canonical item queue"""),
    code(r'''
from __future__ import annotations

import csv
import gc
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes.common.clones import algo_cluster_map

PROBE1_TEMPLATE = (
    "Solve the following problem exactly and provide only the final answer "
    "in the required output format. Problem: {problem}. Format instruction: "
    "{family_specific_output_format}."
)
FAMILY_FORMAT = {
    "GSM": (
        "Write the final numerical answer on its own line as #### <number>. "
        "No other text after that tag."
    ),
    "ALGO": (
        "Follow the problem's required output format exactly "
        "(Path: / Count: / Selected: or Total: / Scoops:). No explanation."
    ),
    "BW": (
        "A numbered list of actions only. Each action must be one of the "
        "permitted operators with their arguments. No explanation."
    ),
}

MODELS = [
    ("EleutherAI/pythia-2.8b", "pile", "Pythia/Pile"),
    ("allenai/OLMo-2-0425-1B", "dolma", "OLMo/Dolma"),
]

O16_CSV = OUT_DIR / "O16_open_model_scores.csv"
OUT_COLUMNS = [
    "family", "problem_id", "variant", "model", "corpus_lineage",
    "clone_family",
    # O15
    "o15_n_tokens", "o15_mean_nll", "o15_sum_nll", "o15_residual_mean_nll",
    "o15_min_k_mean_logprob", "o15_min_k_mean_nll", "o15_n_min_k_tokens",
    # O5
    "o5_prompt_n_tokens", "o5_n_gold_tokens",
    "o5_mean_logprob", "o5_sum_logprob", "o5_mean_nll_gold",
    "o5_gold_first_token_logprob", "o5_gold_first_token_rank",
]


def _norm_vt(v: str) -> str:
    v = str(v).strip()
    return "canonical" if v.lower() == "canonical" else v.upper()


def _strip_csv_quotes(text: str) -> str:
    s = str(text)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s


def load_canonicals(limit: int | None) -> list[dict[str, Any]]:
    specs = [
        ("GSM", REPO_ROOT / "data/problems/question_bank_gsm.csv"),
        ("ALGO", REPO_ROOT / "data/problems/question_bank_algo.csv"),
        ("BW", REPO_ROOT / "data/problems/question_bank_bw.csv"),
    ]
    cmap = algo_cluster_map()
    items: list[dict[str, Any]] = []
    for family, path in specs:
        df = pd.read_csv(path, dtype=str).fillna("")
        df["variant_type"] = df["variant_type"].map(_norm_vt)
        can = df[df["variant_type"] == "canonical"]
        for _, row in can.iterrows():
            pid = str(row["problem_id"]).strip()
            items.append(
                {
                    "family": family,
                    "problem_id": pid,
                    "variant": "canonical",
                    "problem_text": _strip_csv_quotes(str(row["problem_text"])).strip(),
                    "gold": _strip_csv_quotes(str(row["correct_answer"])),
                    "clone_family": (
                        cmap.get(pid, f"SINGLETON_{pid}")
                        if family == "ALGO"
                        else f"SINGLETON_{pid}"
                    ),
                }
            )
        if limit is not None:
            fam_ids = sorted({x["problem_id"] for x in items if x["family"] == family})[:limit]
            keep = set(fam_ids)
            items = [x for x in items if x["family"] != family or x["problem_id"] in keep]
    return items


ITEMS = load_canonicals(LIMIT)
print(f"[queue] {len(ITEMS)} canonical items")
print(pd.DataFrame(ITEMS).groupby("family").size().to_string())
'''),
    md("""## Scoring functions (O15 statement + O5 teacher-forced gold)"""),
    code(r'''
def append_rows(path: Path, rows: list[dict]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in OUT_COLUMNS})


def _done_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, dtype=str).fillna("")
    return {
        (str(r.family), str(r.problem_id), str(r.model))
        for r in df.itertuples(index=False)
    }


def wrap_chat(tokenizer, user_text: str) -> str:
    """Best-effort chat wrap; base models fall back to raw user text."""
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return user_text


def build_prompt(problem_text: str, family: str) -> str:
    return PROBE1_TEMPLATE.format(
        problem=problem_text.strip(),
        family_specific_output_format=FAMILY_FORMAT[family],
    )


@torch.inference_mode()
def statement_surprisal(model, tokenizer, device, text: str) -> dict[str, Any]:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    n = int(input_ids.shape[1])
    if n < 2:
        return {
            "o15_n_tokens": n, "o15_mean_nll": float("nan"), "o15_sum_nll": float("nan"),
            "o15_min_k_mean_logprob": float("nan"), "o15_min_k_mean_nll": float("nan"),
            "o15_n_min_k_tokens": 0,
        }
    if DRY_RUN or model is None:
        return {
            "o15_n_tokens": n - 1, "o15_mean_nll": 2.5, "o15_sum_nll": 2.5 * (n - 1),
            "o15_min_k_mean_logprob": -3.0, "o15_min_k_mean_nll": 3.0,
            "o15_n_min_k_tokens": max(1, int(np.ceil((n - 1) * MIN_K_PCT / 100))),
        }
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0, :-1].float()
    targets = input_ids[0, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    tok_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
    n_scored = len(tok_lp)
    mean_nll = float((-tok_lp).mean())
    k = max(1, int(np.ceil(n_scored * MIN_K_PCT / 100.0)))
    lowest = np.sort(tok_lp)[:k]
    min_k_lp = float(lowest.mean())
    del out, logits, input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "o15_n_tokens": n_scored,
        "o15_mean_nll": round(mean_nll, 6),
        "o15_sum_nll": round(float((-tok_lp).sum()), 6),
        "o15_min_k_mean_logprob": round(min_k_lp, 6),
        "o15_min_k_mean_nll": round(-min_k_lp, 6),
        "o15_n_min_k_tokens": k,
    }


def resolve_continuation(tokenizer, prompt: str, answer: str):
    enc = lambda t: tokenizer(t, add_special_tokens=False)["input_ids"]
    prompt_ids = enc(prompt)
    candidates = []
    for sep in ("", "\n", " "):
        joint = enc(prompt + sep + answer)
        if joint[: len(prompt_ids)] != prompt_ids:
            continue
        rest = joint[len(prompt_ids):]
        candidates.append((sep, rest, len(joint)))
    if not candidates:
        return prompt_ids, enc(answer)
    candidates.sort(key=lambda c: c[2])
    return prompt_ids, candidates[0][1]


@torch.inference_mode()
def teacher_forced_gold(model, tokenizer, device, user_text: str, gold: str) -> dict[str, Any]:
    prompt = wrap_chat(tokenizer, user_text)
    prompt_ids, gold_ids = resolve_continuation(tokenizer, prompt, gold)
    n_prompt, n_gold = len(prompt_ids), len(gold_ids)
    if n_gold == 0:
        return {
            "o5_prompt_n_tokens": n_prompt, "o5_n_gold_tokens": 0,
            "o5_mean_logprob": float("nan"), "o5_sum_logprob": float("nan"),
            "o5_mean_nll_gold": float("nan"),
            "o5_gold_first_token_logprob": float("nan"), "o5_gold_first_token_rank": -1,
        }
    if DRY_RUN or model is None:
        return {
            "o5_prompt_n_tokens": n_prompt, "o5_n_gold_tokens": n_gold,
            "o5_mean_logprob": -1.5, "o5_sum_logprob": -1.5 * n_gold,
            "o5_mean_nll_gold": 1.5,
            "o5_gold_first_token_logprob": -1.2, "o5_gold_first_token_rank": 3,
        }
    input_ids = torch.tensor([prompt_ids + gold_ids], dtype=torch.long, device=device)
    out = model(input_ids=input_ids, use_cache=False)
    gold_logits = out.logits[0, n_prompt - 1 : n_prompt + n_gold - 1].float()
    log_probs = F.log_softmax(gold_logits, dim=-1)
    gold_t = torch.tensor(gold_ids, device=device, dtype=torch.long)
    tok_lp = log_probs.gather(1, gold_t.unsqueeze(1)).squeeze(1)
    sum_lp = float(tok_lp.sum().item())
    mean_lp = sum_lp / n_gold
    first_logits = gold_logits[0]
    first_tid = int(gold_ids[0])
    first_lp = float(F.log_softmax(first_logits, dim=-1)[first_tid].item())
    rank = int((first_logits > first_logits[first_tid]).sum().item()) + 1
    del out, input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "o5_prompt_n_tokens": n_prompt,
        "o5_n_gold_tokens": n_gold,
        "o5_mean_logprob": round(mean_lp, 6),
        "o5_sum_logprob": round(sum_lp, 6),
        "o5_mean_nll_gold": round(-mean_lp, 6),
        "o5_gold_first_token_logprob": round(first_lp, 6),
        "o5_gold_first_token_rank": rank,
    }


def load_model(model_id: str):
    assert torch.cuda.is_available() or DRY_RUN, "GPU required unless DRY_RUN"
    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN or True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if DRY_RUN:
        print(f"[model] DRY_RUN skip {model_id}")
        return tok, None, torch.device("cpu")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        token=HF_TOKEN or True,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    mdl.eval()
    device = next(mdl.parameters()).device
    print(f"[model] {model_id} fp16 device={device}")
    return tok, mdl, device


def unload(mdl):
    if mdl is None:
        return
    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
'''),
    md("""## Run"""),
    code(r'''
done = _done_keys(O16_CSV) if RESUME else set()
print(f"[resume] {len(done)} rows in {O16_CSV}")

for model_id, corpus, lineage in MODELS:
    tok, mdl, device = load_model(model_id)
    buf: list[dict] = []
    todo = [it for it in ITEMS if (it["family"], it["problem_id"], model_id) not in done]
    print(f"[run] {model_id}: {len(todo)} remaining")
    for it in tqdm(todo, desc=lineage):
        o15 = statement_surprisal(mdl, tok, device, it["problem_text"])
        user = build_prompt(it["problem_text"], it["family"])
        o5 = teacher_forced_gold(mdl, tok, device, user, it["gold"])
        row = {
            "family": it["family"],
            "problem_id": it["problem_id"],
            "variant": "canonical",
            "model": model_id,
            "corpus_lineage": corpus,
            "clone_family": it["clone_family"],
            **o15,
            **o5,
        }
        buf.append(row)
        if len(buf) >= 16:
            append_rows(O16_CSV, buf)
            buf = []
    if buf:
        append_rows(O16_CSV, buf)
    unload(mdl)
    del tok
    gc.collect()

# Length residual within model (O15)
if O16_CSV.exists():
    df = pd.read_csv(O16_CSV)
    parts = []
    for _, g in df.groupby("model"):
        sub = g.copy()
        x = pd.to_numeric(sub["o15_n_tokens"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub["o15_mean_nll"], errors="coerce").to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        resid = np.full(len(sub), np.nan)
        if m.sum() >= 3 and np.unique(x[m]).size >= 2:
            b, a = np.polyfit(x[m], y[m], 1)
            resid[m] = y[m] - (a + b * x[m])
        sub["o15_residual_mean_nll"] = resid
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    # Ensure column exists in schema for downstream
    if "o15_residual_mean_nll" not in OUT_COLUMNS:
        pass
    out.to_csv(O16_CSV, index=False)
    print(out.groupby(["model", "family"]).size().unstack(fill_value=0).to_string())
    print(f"[done] {O16_CSV} ({len(out)} rows)")
'''),
    md("""## Download → `results/raw/O16_open_model_scores.csv`"""),
    code(r'''
_out_files = [O16_CSV]
_drive_dir = Path("/content/drive/MyDrive/rvc_colab_out")
if Path("/content/drive").exists():
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
    except Exception as exc:
        print("[drive] skipped:", exc)
try:
    import shutil as _shutil
    _drive_dir.mkdir(parents=True, exist_ok=True)
    for p in _out_files:
        if p.exists():
            _shutil.copy2(p, _drive_dir / p.name)
            print(f"[backup] {p.name}")
except Exception as exc:
    print("[backup] skipped:", exc)
try:
    from google.colab import files as _colab_files  # type: ignore
    for p in _out_files:
        if p.exists():
            _colab_files.download(str(p))
except Exception as exc:
    print("[download] skipped:", exc)
'''),
]


def build() -> Path:
    path = OUT / "o16_open_model_calibration.ipynb"
    # Add residual column to saved CSV schema note — written in-run.
    path.write_text(
        json.dumps(nb(NB_O16, "o16_open_model_calibration.ipynb"), indent=1) + "\n"
    )
    print(f"wrote {path} ({len(NB_O16)} cells)")
    return path


if __name__ == "__main__":
    build()
