#!/usr/bin/env python3
"""Build o15_surprisal_contamination.ipynb (Colab T4).

Also importable from ``_build_notebooks.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _split(src)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split(src),
    }


def _split(src: str) -> list[str]:
    src = src.strip("\n") + "\n"
    if not src:
        return []
    return src.splitlines(keepends=True)


def nb(cells: list[dict], name: str) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "colab": {"provenance": [], "gpuType": "T4", "name": name},
        },
        "cells": cells,
    }


# Load shared SETUP_PIP / SETUP_REPO from the sibling builder (avoid package import).
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_rvc_build_notebooks", OUT / "_build_notebooks.py"
)
assert _spec is not None and _spec.loader is not None
_bn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bn)
SETUP_PIP = _bn.SETUP_PIP
SETUP_REPO = _bn.SETUP_REPO


NB_O15 = [
    md("""# O15 — Independent surprisal contamination measure (Colab T4)

Probe 3 currently rests on a **single** proxy (Infini-gram n-gram overlap) with
**incomparable** windows across families (GSM 8, ALGO/BW 13). This notebook adds
an independent second measure: **causal LM surprisal of the problem statement**
(not the solution).

### Models (T4)
| Model | dtype | flag |
|-------|-------|------|
| `EleutherAI/pythia-2.8b` (The Pile) | fp16 | primary |
| `allenai/OLMo-2-0425-1B` (Dolma / OLMo-mix) | fp16 | primary |
| `Qwen/Qwen2.5-1.5B` | fp16 | primary |
| `EleutherAI/pythia-6.9b` | **4-bit NF4** | optional scale (`RUN_OPTIONAL_SCALE`) |
| `allenai/OLMo-2-1124-7B` | **4-bit NF4** | optional scale |

**T4 hard constraints:** fp16 only (no bf16), `attn_implementation="sdpa"`, no FlashAttention-2.

### Per problem statement (canonical **and** every bank variant)
1. **Mean per-token NLL** (negative log-likelihood).
2. Length control is applied **downstream** (`scripts/consolidate/o15_surprisal_vs_infinigram.py`): OLS residual of mean NLL on token count within family×model.
3. **Min-k%** (k=20): mean log-prob of the k% lowest-probability tokens (Carlini-style membership statistic).

### Output
`colab_out/O15_surprisal_contamination.csv` → copy to `results/raw/O15_surprisal_contamination.csv`, then run the consolidate script for residuals + Infini-gram Spearman.

**Pre-registered reading:** agreement with Infini-gram validates Probe 3; disagreement shows the field's standard proxy is unreliable. Report which.

**Secrets:** optional `HF_TOKEN` / `GITHUB_TOKEN` (Colab 🔑)."""),
    code(SETUP_PIP),
    code(
        SETUP_REPO
        + r'''

# O15-specific knobs (override shared LIMIT/DRY_RUN/RESUME above if needed)
RUN_OPTIONAL_SCALE = True   # Pythia-6.9B + OLMo-2-7B in 4-bit
MIN_K_PCT = 20
'''
    ),
    md("""## Item queue — every bank problem statement (canonical + W*)

No Probe-1 chat wrapper: we score the **raw `problem_text`** under each base LM.
Infini-gram correlations use existing P3 scores (canonical texts); variants are
still scored here so length residuals see the full statement distribution."""),
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from probes.common.clones import algo_cluster_map

VARIANTS = ("canonical", "W1", "W2", "W3", "W4", "W5", "W6")

# (hf_id, quant, optional_scale, corpus_note, short_name)
MODELS: list[tuple[str, str, bool, str, str]] = [
    ("EleutherAI/pythia-2.8b", "fp16", False, "The Pile", "pythia-2.8b"),
    ("allenai/OLMo-2-0425-1B", "fp16", False, "Dolma/OLMo-mix", "olmo2-1b"),
    ("Qwen/Qwen2.5-1.5B", "fp16", False, "Qwen2.5 pretrain", "qwen2.5-1.5b"),
    ("EleutherAI/pythia-6.9b", "nf4", True, "The Pile", "pythia-6.9b"),
    ("allenai/OLMo-2-1124-7B", "nf4", True, "Dolma/OLMo-mix", "olmo2-7b"),
]

O15_CSV = OUT_DIR / "O15_surprisal_contamination.csv"

OUT_COLUMNS = [
    "family",
    "problem_id",
    "variant",
    "model",
    "model_short",
    "quantized",
    "dtype_label",
    "corpus_note",
    "n_tokens",
    "sum_nll",
    "mean_nll",
    "min_k_pct",
    "min_k_mean_logprob",
    "min_k_mean_nll",
    "n_min_k_tokens",
    "clone_family",
    "whitespace_n_tokens",
]


def _norm_vt(v: str) -> str:
    v = str(v).strip()
    return "canonical" if v.lower() == "canonical" else v.upper()


def _strip_csv_quotes(text: str) -> str:
    s = str(text)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s


def _load_bank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["problem_id"] = df["problem_id"].astype(str).str.strip()
    df["variant_type"] = df["variant_type"].map(_norm_vt)
    df["problem_text"] = df["problem_text"].map(_strip_csv_quotes)
    return df


def clone_family_for(family: str, problem_id: str, cmap: dict[str, str]) -> str:
    if family == "ALGO":
        return cmap.get(problem_id, f"SINGLETON_{problem_id}")
    return f"SINGLETON_{problem_id}"


def load_items(limit: int | None) -> list[dict[str, Any]]:
    specs = [
        ("GSM", REPO_ROOT / "data/problems/question_bank_gsm.csv"),
        ("ALGO", REPO_ROOT / "data/problems/question_bank_algo.csv"),
        ("BW", REPO_ROOT / "data/problems/question_bank_bw.csv"),
    ]
    cmap = algo_cluster_map()
    items: list[dict[str, Any]] = []
    for family, path in specs:
        df = _load_bank(path)
        for _, row in df.iterrows():
            vt = str(row["variant_type"])
            if vt not in VARIANTS:
                continue
            text = str(row["problem_text"]).strip()
            if not text:
                continue
            pid = str(row["problem_id"])
            items.append(
                {
                    "family": family,
                    "problem_id": pid,
                    "variant": vt,
                    "problem_text": text,
                    "clone_family": clone_family_for(family, pid, cmap),
                    "whitespace_n_tokens": len(text.split()),
                }
            )
    if limit is not None:
        keep: set[tuple[str, str]] = set()
        for fam in ("GSM", "ALGO", "BW"):
            ids = sorted({x["problem_id"] for x in items if x["family"] == fam})[:limit]
            keep |= {(fam, pid) for pid in ids}
        items = [x for x in items if (x["family"], x["problem_id"]) in keep]
    return items


ITEMS = load_items(LIMIT)
print(f"[queue] {len(ITEMS)} problem statements (LIMIT={LIMIT})")
print(pd.DataFrame(ITEMS).groupby(["family", "variant"]).size().unstack(fill_value=0).to_string())
'''),
    md("""## Surprisal metrics on the problem statement

Causal LM forward pass on the tokenized statement. Token NLLs are for positions
`1..n-1` (token `t_i` predicted from `t_<i`). Min-k% uses the lowest-probability
`ceil(0.20 * n)` of those tokens."""),
    code(r'''
def append_rows(path: Path, rows: list[dict]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in OUT_COLUMNS})


def _done_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, dtype=str).fillna("")
    keys = set()
    for _, r in df.iterrows():
        keys.add(
            (
                str(r["family"]),
                str(r["problem_id"]),
                str(r["variant"]),
                str(r["model"]),
            )
        )
    return keys


@torch.inference_mode()
def statement_surprisal(
    model,
    tokenizer,
    device,
    text: str,
    *,
    min_k_pct: int = 20,
) -> dict[str, Any]:
    """Mean NLL + min-k% on the problem statement (no chat template, no gold)."""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    n = int(input_ids.shape[1])
    if n < 2:
        return {
            "n_tokens": n,
            "sum_nll": float("nan"),
            "mean_nll": float("nan"),
            "min_k_mean_logprob": float("nan"),
            "min_k_mean_nll": float("nan"),
            "n_min_k_tokens": 0,
        }
    if DRY_RUN or model is None:
        # Deterministic placeholder from length (pipeline check only).
        fake = -2.0 - 0.001 * n
        return {
            "n_tokens": n,
            "sum_nll": round((-fake) * (n - 1), 6),
            "mean_nll": round(-fake, 6),
            "min_k_mean_logprob": round(fake - 0.5, 6),
            "min_k_mean_nll": round(-(fake - 0.5), 6),
            "n_min_k_tokens": max(1, int(np.ceil((n - 1) * min_k_pct / 100.0))),
        }

    out = model(input_ids=input_ids, use_cache=False)
    # logits[i] predicts token i+1 → score tokens 1..n-1
    logits = out.logits[0, :-1].float()
    targets = input_ids[0, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    tok_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    tok_lp_np = tok_lp.detach().cpu().numpy()
    n_scored = int(tok_lp_np.shape[0])
    mean_lp = float(tok_lp_np.mean())
    sum_nll = float((-tok_lp_np).sum())
    mean_nll = float(-mean_lp)

    k = max(1, int(np.ceil(n_scored * float(min_k_pct) / 100.0)))
    lowest = np.sort(tok_lp_np)[:k]  # most surprising (lowest logprob)
    min_k_mean_lp = float(lowest.mean())

    del out, logits, input_ids, log_probs, tok_lp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "n_tokens": n_scored,  # scored continuation tokens (n_input - 1)
        "sum_nll": round(sum_nll, 6),
        "mean_nll": round(mean_nll, 6),
        "min_k_mean_logprob": round(min_k_mean_lp, 6),
        "min_k_mean_nll": round(-min_k_mean_lp, 6),
        "n_min_k_tokens": k,
    }


def load_model(model_id: str, quant: str):
    assert torch.cuda.is_available() or DRY_RUN, "GPU required (Colab T4) unless DRY_RUN."
    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN or True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if DRY_RUN:
        print(f"[model] DRY_RUN skip load: {model_id} ({quant})")
        return tok, None, torch.device("cpu")

    common = dict(
        device_map="auto",
        token=HF_TOKEN or True,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if quant == "fp16":
        mdl = AutoModelForCausalLM.from_pretrained(model_id, **common)
        label = "fp16"
    elif quant == "nf4":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, **common
        )
        label = "nf4_4bit"
    else:
        raise ValueError(quant)
    mdl.eval()
    device = next(mdl.parameters()).device
    print(f"[model] {model_id}  {label}  device={device}")
    return tok, mdl, device


def unload(mdl):
    if mdl is None:
        return
    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_item(model, tokenizer, device, item: dict, meta: dict) -> dict[str, Any]:
    m = statement_surprisal(
        model, tokenizer, device, item["problem_text"], min_k_pct=MIN_K_PCT
    )
    return {
        "family": item["family"],
        "problem_id": item["problem_id"],
        "variant": item["variant"],
        "model": meta["model_id"],
        "model_short": meta["short"],
        "quantized": str(meta["quant"] != "fp16").lower(),
        "dtype_label": meta["quant"],
        "corpus_note": meta["corpus"],
        "n_tokens": m["n_tokens"],
        "sum_nll": m["sum_nll"],
        "mean_nll": m["mean_nll"],
        "min_k_pct": MIN_K_PCT,
        "min_k_mean_logprob": m["min_k_mean_logprob"],
        "min_k_mean_nll": m["min_k_mean_nll"],
        "n_min_k_tokens": m["n_min_k_tokens"],
        "clone_family": item["clone_family"],
        "whitespace_n_tokens": item["whitespace_n_tokens"],
    }
'''),
    md("""## Run all models (resume-safe)"""),
    code(r'''
active_models = [
    m for m in MODELS if (not m[2]) or RUN_OPTIONAL_SCALE
]
print("[models]", [f"{s} ({q}{' optional' if opt else ''})" for _, q, opt, _, s in active_models])

done = _done_keys(O15_CSV) if RESUME else set()
print(f"[resume] {len(done)} rows already in {O15_CSV}")

for model_id, quant, _opt, corpus, short in active_models:
    tok, mdl, device = load_model(model_id, quant)
    meta = {"model_id": model_id, "quant": quant, "corpus": corpus, "short": short}
    buf: list[dict] = []
    todo = [
        it for it in ITEMS
        if (it["family"], it["problem_id"], it["variant"], model_id) not in done
    ]
    print(f"[run] {short}: {len(todo)} remaining / {len(ITEMS)} total")
    for it in tqdm(todo, desc=short):
        buf.append(score_item(mdl, tok, device, it, meta))
        if len(buf) >= 32:
            append_rows(O15_CSV, buf)
            buf = []
    if buf:
        append_rows(O15_CSV, buf)
    unload(mdl)
    del tok
    gc.collect()

print(f"\n[done] wrote {O15_CSV}")
if O15_CSV.exists():
    out_df = pd.read_csv(O15_CSV)
    print(out_df.groupby(["model_short", "family"]).size().unstack(fill_value=0).to_string())
    print(out_df.groupby("model_short")[["mean_nll", "min_k_mean_logprob"]].mean().round(4).to_string())
'''),
    md("""## Download

Copy `O15_surprisal_contamination.csv` into the repo as
`results/raw/O15_surprisal_contamination.csv`, then run:

```bash
python scripts/consolidate/o15_surprisal_vs_infinigram.py
```

That script residualizes mean NLL on length, writes
`results/derived/O15_surprisal_contamination.csv`, and produces
`results/derived/O15_surprisal_vs_infinigram.csv` (Spearman × Infini-gram,
cluster-bootstrap CIs, agreement verdict)."""),
    code(r'''
_out_files = [O15_CSV]
_drive_dir = Path("/content/drive/MyDrive/rvc_colab_out")
if Path("/content/drive").exists():
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
    except Exception as exc:
        print("[drive] mount skipped:", exc)
try:
    import shutil as _shutil
    _drive_dir.mkdir(parents=True, exist_ok=True)
    for p in _out_files:
        if p.exists():
            _shutil.copy2(p, _drive_dir / p.name)
            print(f"[backup] {p.name} -> {_drive_dir / p.name}")
except Exception as exc:
    print("[backup] skipped:", exc)
try:
    from google.colab import files as _colab_files  # type: ignore
    for p in _out_files:
        if p.exists():
            _colab_files.download(str(p))
            print(f"[download] {p.name}")
except Exception as exc:
    print("[download] skipped (not Colab or download blocked):", exc)
'''),
]


def build() -> Path:
    path = OUT / "o15_surprisal_contamination.ipynb"
    path.write_text(json.dumps(nb(NB_O15, "o15_surprisal_contamination.ipynb"), indent=1) + "\n")
    print(f"wrote {path} ({len(NB_O15)} cells)")
    return path


if __name__ == "__main__":
    build()
