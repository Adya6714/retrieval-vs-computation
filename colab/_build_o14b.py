#!/usr/bin/env python3
"""Build colab/o14b_naming_likelihood.ipynb (Colab T4).

O14 API generation is blocked (no OpenRouter key). This notebook runs the same
controlled naming intervention with O5 teacher-forced mean_logprob on
Qwen2.5-1.5B/3B-Instruct, plus greedy accuracy (expected BINARY_DEGENERATE on BW).
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


NB_O14B = [
    md("""# O14b — Controlled Blocksworld naming intervention (teacher-forced likelihood)

**Colab T4 · GPU required · no OpenRouter.**

The API O14 eval is blocked (`OPENROUTER_API_KEY` 401). This notebook runs the
**same controlled n=120 × 3-arm bank** using **teacher-forced gold-plan
`mean_logprob`**, which remains informative even when open models are at floor
on Blocksworld accuracy.

### Bank
`results/derived/O14_naming_bank.jsonl` — 120 pairs × 3 arms = **360 rows**.

| Arm | Naming |
|-----|--------|
| `A_sequential` | `a, b, c, …` |
| `B_scattered` | scattered letters |
| `C_indexed` | `b1, b2, …` |

Byte-level `assert_only_block_ids_differ` was already validated when the bank was built.

### Models (T4 hard constraints)
| Model | load |
|-------|------|
| `Qwen/Qwen2.5-1.5B-Instruct` | fp16, `attn_implementation="sdpa"` |
| `Qwen/Qwen2.5-3B-Instruct` | fp16, `attn_implementation="sdpa"` |

**Reuse O5 teacher-forcing exactly** (`wrap_chat` / `resolve_continuation` /
`teacher_forced_metrics` / Appendix-N `PROBE1_TEMPLATE` + BW `FAMILY_FORMAT`).
Do not reimplement.

### Outputs
- `colab_out/O14b_naming_likelihood.csv` → `results/raw/O14b_naming_likelihood.csv`
- `colab_out/O14b_naming_analysis.csv` → `results/derived/O14b_naming_analysis.csv`

---

## SCOPE CAVEAT (notebook + paper)

This measures **belief shift (gold-plan likelihood) in small open models**, **not**
accuracy in the six frontier models where L1 was observed. It is a controlled
**n=120** intervention replacing an **n=13** observational stratification — a
genuine upgrade — but **it is not the same experiment**.

---

## PRE-REGISTERED INTERPRETATION (do not revise after seeing results)

Write this **before** the run. Both outcomes are reported.

1. **Substantial naming effect on `mean_logprob`** (paired arm contrast; 95% CI
   for Δ excludes 0 under pair_id cluster bootstrap) → block naming is a real
   **lexical** variable; L1 is reframed as partly lexical; this becomes a
   **primary finding**.
2. **Null-compatible** (CI includes 0 on primary A−B contrast for both models) →
   L1 **survives** the naming confound with a controlled n=120 on open models,
   with the explicit caveat that this does **not** test the frontier models
   where L1 was observed.

Greedy accuracy is recorded for completeness. If it is at floor, flag
`BINARY_DEGENERATE` so nobody mistakes a floor for a null on the likelihood
endpoint.
"""),
    code(SETUP_PIP),
    code(
        SETUP_REPO
        + r'''

# O14b knobs (override LIMIT for a smoke test, e.g. 4 pairs → 12 rows)
LIMIT_PAIRS = None   # None = all 120 pairs
MAX_NEW_TOKENS = 512  # BW greedy (same ballpark as other BW Colab notebooks)
N_BOOT = 5000
SEED = 42
FLOOR_ACC = 0.05      # greedy mean acc ≤ this → BINARY_DEGENERATE
'''
    ),
    md("""## Pre-registration gate (must print before any scoring)

Confirm the interpretation and scope caveat are frozen. Edit nothing below after
this cell has run on real data."""),
    code(r'''
PREREG = {
    "endpoint": "mean_logprob of gold plan under Probe-1 prompt (O5 path)",
    "primary_contrast": "A_sequential − B_scattered (paired within pair_id)",
    "secondary_contrasts": ["A_sequential − C_indexed", "B_scattered − C_indexed"],
    "inference": "cluster bootstrap on pair_id, B=5000, seed=42; CI excludes 0",
    "substantial_naming_effect": (
        "primary Δ mean_logprob 95% CI excludes 0 → primary finding / L1 partly lexical"
    ),
    "null_compatible": (
        "primary CI includes 0 for both models → L1 survives naming confound on open "
        "models; does NOT test frontier L1 models"
    ),
    "scope_caveat": (
        "belief shift in Qwen-1.5B/3B, not frontier accuracy; controlled n=120 "
        "replaces n=13 observational stratification — upgrade, not the same experiment"
    ),
    "greedy": "record accuracy; BINARY_DEGENERATE if mean acc ≤ 0.05",
}
for k, v in PREREG.items():
    print(f"{k}:\n  {v}\n")
assert N_BOOT == 5000 and SEED == 42
print("[prereg] frozen — proceed to score")
'''),
    md("""## Load O14 naming bank (360 rows)"""),
    code(r'''
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes.contamination.verify import verify_answer

BANK_CANDIDATES = [
    REPO_ROOT / "results/derived/O14_naming_bank.jsonl",
    Path("/content/drive/MyDrive/rvc_colab_out/O14_naming_bank.jsonl"),
]
BANK_PATH = next((p for p in BANK_CANDIDATES if p.is_file()), None)
assert BANK_PATH is not None, f"Missing O14 bank; tried {BANK_CANDIDATES}"

rows_bank: list[dict[str, Any]] = []
with BANK_PATH.open() as f:
    for line in f:
        line = line.strip()
        if line:
            rows_bank.append(json.loads(line))

bank_df = pd.DataFrame(rows_bank)
assert len(bank_df) == 360, f"expected 360 rows, got {len(bank_df)}"
assert bank_df["pair_id"].nunique() == 120
assert set(bank_df["arm"]) == {"A_sequential", "B_scattered", "C_indexed"}

if LIMIT_PAIRS is not None:
    keep = sorted(bank_df["pair_id"].unique())[: int(LIMIT_PAIRS)]
    bank_df = bank_df[bank_df["pair_id"].isin(keep)].copy()
    print(f"[smoke] LIMIT_PAIRS={LIMIT_PAIRS} → {len(bank_df)} rows")

print(f"[bank] {BANK_PATH}")
print(bank_df.groupby("arm").size().to_string())
print("num_blocks dist:\n", bank_df.drop_duplicates("pair_id")["num_blocks"].value_counts().sort_index().to_string())
'''),
    md("""## O5 teacher-forcing primitives (copied verbatim — do not edit)

Appendix-N Probe-1 prompt + chat wrap + joint-encode gold continuation.
Identical to `colab/o5_teacher_forced_likelihood.ipynb` / `_build_notebooks.py` NB3."""),
    code(r'''
# Appendix N Probe-1 template (paper/appendix.tex \label{app:prompts})
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

MODELS: list[tuple[str, str]] = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "fp16"),
    ("Qwen/Qwen2.5-3B-Instruct", "fp16"),
]

O14B_CSV = OUT_DIR / "O14b_naming_likelihood.csv"
O14B_ANALYSIS = OUT_DIR / "O14b_naming_analysis.csv"

COLUMNS = [
    "pair_id",
    "arm",
    "problem_id",
    "model",
    "model_short",
    "num_blocks",
    "plan_length",
    "n_gold_tokens",
    "sum_logprob",
    "mean_logprob",
    "gold_first_token_rank",
    "gold_first_token_logprob",
    "prompt_n_tokens",
    "sep_note",
    "greedy_response",
    "greedy_correct",
    "binary_flag",
]


def build_prompt(problem_text: str, family: str) -> str:
    """Identical Probe-1 user string construction as the behavioural Colab notebooks."""
    return PROBE1_TEMPLATE.format(
        problem=problem_text.strip(),
        family_specific_output_format=FAMILY_FORMAT[family],
    )


def wrap_chat(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        tokenize=False,
    )


def resolve_continuation(
    tokenizer,
    prompt: str,
    answer: str,
) -> tuple[list[int], list[int], str]:
    """Prompt-aware gold token ids (joint encode; try '' then ' ' separator)."""

    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    prompt_ids = enc(prompt)
    answer = str(answer)
    if not answer:
        return prompt_ids, [], "EMPTY"
    candidates: list[tuple[str, list[int], int]] = []
    for sep in ("", " "):
        joint = enc(prompt + sep + answer)
        if len(joint) <= len(prompt_ids):
            continue
        if joint[: len(prompt_ids)] != prompt_ids:
            continue
        rest = joint[len(prompt_ids) :]
        candidates.append((sep, rest, len(joint)))
    if not candidates:
        bare = enc(answer)
        return prompt_ids, bare, "FALLBACK"
    candidates.sort(key=lambda c: c[2])
    sep, rest, _ = candidates[0]
    return prompt_ids, rest, repr(sep)


@torch.inference_mode()
def teacher_forced_metrics(
    model,
    tokenizer,
    device,
    user_text: str,
    gold_text: str,
) -> dict[str, Any]:
    prompt = wrap_chat(tokenizer, user_text)
    prompt_ids, gold_ids, sep_note = resolve_continuation(tokenizer, prompt, gold_text)
    n_prompt = len(prompt_ids)
    n_gold = len(gold_ids)
    if n_gold == 0:
        return {
            "n_gold_tokens": 0,
            "sum_logprob": float("nan"),
            "mean_logprob": float("nan"),
            "gold_first_token_rank": -1,
            "gold_first_token_logprob": float("nan"),
            "prompt_n_tokens": n_prompt,
            "sep_note": sep_note,
        }
    if DRY_RUN or model is None:
        return {
            "n_gold_tokens": n_gold,
            "sum_logprob": 0.0,
            "mean_logprob": 0.0,
            "gold_first_token_rank": 1,
            "gold_first_token_logprob": 0.0,
            "prompt_n_tokens": n_prompt,
            "sep_note": "DRY_RUN",
        }

    input_ids = torch.tensor([prompt_ids + gold_ids], dtype=torch.long, device=device)
    out = model(input_ids=input_ids, use_cache=False)
    # logits[t] predicts token t+1
    logits = out.logits[0]  # [seq, vocab]
    # gold token at absolute index n_prompt + i is predicted by position n_prompt + i - 1
    gold_logits = logits[n_prompt - 1 : n_prompt + n_gold - 1]
    log_probs = F.log_softmax(gold_logits.float(), dim=-1)
    gold_t = torch.tensor(gold_ids, device=device, dtype=torch.long)
    tok_lp = log_probs.gather(1, gold_t.unsqueeze(1)).squeeze(1)
    sum_lp = float(tok_lp.sum().item())
    mean_lp = sum_lp / n_gold

    first_logits = gold_logits[0].float()
    first_tid = int(gold_ids[0])
    first_lp = float(F.log_softmax(first_logits, dim=-1)[first_tid].item())
    rank = int((first_logits > first_logits[first_tid]).sum().item()) + 1

    del out, logits, input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "n_gold_tokens": n_gold,
        "sum_logprob": round(sum_lp, 6),
        "mean_logprob": round(mean_lp, 6),
        "gold_first_token_rank": rank,
        "gold_first_token_logprob": round(first_lp, 6),
        "prompt_n_tokens": n_prompt,
        "sep_note": sep_note,
    }


def load_model(model_id: str, quant: str):
    assert torch.cuda.is_available() or DRY_RUN, "GPU required (Colab T4) unless DRY_RUN."
    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN or True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if DRY_RUN:
        print(f"[model] DRY_RUN skip load: {model_id} ({quant})")
        return tok, None, torch.device("cpu")

    common = dict(
        device_map="auto",
        token=HF_TOKEN or True,
        attn_implementation="sdpa",  # T4: no FlashAttention-2
        torch_dtype=torch.float16,  # T4: fp16 only, no bf16
    )
    if quant == "fp16":
        mdl = AutoModelForCausalLM.from_pretrained(model_id, **common)
        label = "fp16 unquantized + sdpa"
    else:
        raise ValueError(f"O14b allows fp16 only, got {quant}")
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


@torch.inference_mode()
def greedy_generate(model, tokenizer, device, user_text: str) -> str:
    if DRY_RUN or model is None:
        return "pick-up a\nstack a b"
    prompt = wrap_chat(tokenizer, user_text)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[0, input_ids.shape[1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    del out, input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text


print("[o5] teacher-forced primitives ready")
'''),
    md("""## Score 360 rows × 2 models (resume-safe)

For each row: Probe-1 BW prompt → teacher-forced `mean_logprob` of `correct_answer`,
then greedy generation + `verify_answer(..., family="blocksworld")`."""),
    code(r'''
MODEL_SHORT = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen1.5B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen3B",
}


def _done_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file():
        return set()
    df = pd.read_csv(path, dtype=str).fillna("")
    return set(zip(df["model"], df["pair_id"], df["arm"]))


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in new.columns:
            new[c] = ""
    new = new[COLUMNS]
    if path.is_file():
        old = pd.read_csv(path, dtype=str).fillna("")
        for c in COLUMNS:
            if c not in old.columns:
                old[c] = ""
        old = old[COLUMNS]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(path, index=False)


done = _done_keys(O14B_CSV) if RESUME else set()
print(f"[resume] {len(done)} rows already scored; RESUME={RESUME}")

for model_id, quant in MODELS:
    short = MODEL_SHORT[model_id]
    pending = [
        r
        for r in bank_df.to_dict(orient="records")
        if (model_id, str(r["pair_id"]), str(r["arm"])) not in done
    ]
    print(f"\n=== {model_id}  pending={len(pending)}/{len(bank_df)} ===")
    if not pending and not DRY_RUN:
        continue
    tok, mdl, device = load_model(model_id, quant)
    buf: list[dict[str, Any]] = []
    for r in tqdm(pending, desc=short):
        user = build_prompt(str(r["problem_text"]), "BW")
        gold = str(r["correct_answer"])
        tf = teacher_forced_metrics(mdl, tok, device, user, gold)
        greedy = greedy_generate(mdl, tok, device, user)
        try:
            ok = bool(
                verify_answer(
                    str(r["problem_id"]),
                    greedy,
                    gold,
                    "blocksworld",
                    problem_text=str(r["problem_text"]),
                )
            )
        except Exception as exc:  # noqa: BLE001
            ok = False
            greedy = f"VERIFY_ERROR: {exc}\n{greedy}"
        buf.append(
            {
                "pair_id": r["pair_id"],
                "arm": r["arm"],
                "problem_id": r["problem_id"],
                "model": model_id,
                "model_short": short,
                "num_blocks": r["num_blocks"],
                "plan_length": r["plan_length"],
                "n_gold_tokens": tf["n_gold_tokens"],
                "sum_logprob": tf["sum_logprob"],
                "mean_logprob": tf["mean_logprob"],
                "gold_first_token_rank": tf["gold_first_token_rank"],
                "gold_first_token_logprob": tf["gold_first_token_logprob"],
                "prompt_n_tokens": tf["prompt_n_tokens"],
                "sep_note": tf["sep_note"],
                "greedy_response": greedy.replace("\n", "\\n"),
                "greedy_correct": ok,
                "binary_flag": "",  # filled in analysis
            }
        )
        if len(buf) >= 20:
            append_rows(O14B_CSV, buf)
            buf = []
    append_rows(O14B_CSV, buf)
    unload(mdl)
    done = _done_keys(O14B_CSV)

print(f"\n[wrote] {O14B_CSV}")
print(pd.read_csv(O14B_CSV).groupby(["model_short", "arm"]).size().unstack(fill_value=0).to_string())
'''),
    md("""## Analysis (pre-registered)

1. Paired within-pair **A vs B**, **A vs C**, **B vs C** on `mean_logprob`.
2. Cluster bootstrap on **`pair_id`**, B=5000, seed=42 → Δ ± 95% CI.
3. OLS: within-pair Δ ~ num_blocks + plan_length; also Spearman(Δ, moderator).
4. Greedy accuracy → `BINARY_DEGENERATE` if mean ≤ 0.05."""),
    code(r'''
from probes.common.cluster_inference import bootstrap_p_two_sided, cluster_bootstrap_assoc

raw = pd.read_csv(O14B_CSV, dtype=str).fillna("")
raw["mean_logprob"] = pd.to_numeric(raw["mean_logprob"], errors="coerce")
raw["num_blocks"] = pd.to_numeric(raw["num_blocks"], errors="coerce")
raw["plan_length"] = pd.to_numeric(raw["plan_length"], errors="coerce")
raw["greedy_correct"] = (
    raw["greedy_correct"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
)
raw = raw.drop_duplicates(["model", "pair_id", "arm"], keep="last")

ARM_A, ARM_B, ARM_C = "A_sequential", "B_scattered", "C_indexed"
CONTRASTS = [
    ("A_minus_B", ARM_A, ARM_B),
    ("A_minus_C", ARM_A, ARM_C),
    ("B_minus_C", ARM_B, ARM_C),
]


def cluster_mean_ci(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, float]:
    df = pd.DataFrame({"v": values, "c": cluster_ids})
    per = df.groupby("c", sort=False)["v"].mean()
    estimate = float(per.mean()) if len(per) else float("nan")
    vals = per.to_numpy(dtype=float)
    if len(vals) == 0:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_clustered": float("nan"),
            "n_clusters": 0,
        }
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = rng.choice(len(vals), size=len(vals), replace=True)
        boots[i] = float(np.mean(vals[draw]))
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "p_clustered": float(bootstrap_p_two_sided(boots)),
        "n_clusters": int(len(vals)),
    }


def ols_two_moderators(y: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> dict[str, float]:
    """y ~ 1 + x1 + x2 (num_blocks, plan_length)."""
    X = np.column_stack([np.ones(len(y)), x1, x2])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "intercept": float(coef[0]),
        "coef_num_blocks": float(coef[1]),
        "coef_plan_length": float(coef[2]),
        "r2": r2,
    }


analysis_rows: list[dict[str, Any]] = []

for model_id, gmod in raw.groupby("model"):
    short = MODEL_SHORT.get(str(model_id), str(model_id))
    acc = float(gmod["greedy_correct"].mean()) if len(gmod) else float("nan")
    binary_flag = "BINARY_DEGENERATE" if (acc == acc and acc <= FLOOR_ACC) else "BINARY_OK"
    analysis_rows.append(
        {
            "analysis": "greedy_accuracy",
            "contrast": "all_arms",
            "model": model_id,
            "model_short": short,
            "n_pairs": int(gmod["pair_id"].nunique()),
            "mean_arm_hi": "",
            "mean_arm_lo": "",
            "delta_mean": "",
            "ci_low": "",
            "ci_high": "",
            "p_clustered": "",
            "n_clusters": "",
            "greedy_accuracy": round(acc, 6),
            "binary_flag": binary_flag,
            "moderator": "",
            "spearman_r": "",
            "spearman_ci_low": "",
            "spearman_ci_high": "",
            "spearman_p": "",
            "ols_intercept": "",
            "ols_coef_num_blocks": "",
            "ols_coef_plan_length": "",
            "ols_r2": "",
            "interpretation_gate": binary_flag,
            "scope_caveat": PREREG["scope_caveat"],
        }
    )
    print(f"[{short}] greedy_acc={acc:.4f} → {binary_flag}")

    wide = gmod.pivot_table(
        index="pair_id", columns="arm", values="mean_logprob", aggfunc="last"
    )
    meta = (
        gmod.drop_duplicates("pair_id")
        .set_index("pair_id")[["num_blocks", "plan_length"]]
    )

    for contrast, arm_hi, arm_lo in CONTRASTS:
        if arm_hi not in wide.columns or arm_lo not in wide.columns:
            continue
        sub = wide[[arm_hi, arm_lo]].dropna().join(meta, how="inner")
        if sub.empty:
            continue
        delta = (sub[arm_hi] - sub[arm_lo]).to_numpy(dtype=float)
        clusters = sub.index.to_numpy()
        seed_c = (hash(f"o14b|{model_id}|{contrast}") % (2**31 - 1)) or SEED
        ci = cluster_mean_ci(delta, clusters, seed=seed_c)
        gate = (
            "naming_moves_likelihood"
            if (
                ci["ci_low"] == ci["ci_low"]
                and (ci["ci_low"] > 0 or ci["ci_high"] < 0)
            )
            else "naming_null_compatible"
        )
        analysis_rows.append(
            {
                "analysis": "paired_mean_logprob_delta",
                "contrast": contrast,
                "model": model_id,
                "model_short": short,
                "n_pairs": int(len(sub)),
                "mean_arm_hi": round(float(sub[arm_hi].mean()), 6),
                "mean_arm_lo": round(float(sub[arm_lo].mean()), 6),
                "delta_mean": round(ci["estimate"], 6),
                "ci_low": round(ci["ci_low"], 6),
                "ci_high": round(ci["ci_high"], 6),
                "p_clustered": round(ci["p_clustered"], 6),
                "n_clusters": ci["n_clusters"],
                "greedy_accuracy": round(acc, 6),
                "binary_flag": binary_flag,
                "moderator": "",
                "spearman_r": "",
                "spearman_ci_low": "",
                "spearman_ci_high": "",
                "spearman_p": "",
                "ols_intercept": "",
                "ols_coef_num_blocks": "",
                "ols_coef_plan_length": "",
                "ols_r2": "",
                "interpretation_gate": gate,
                "scope_caveat": PREREG["scope_caveat"],
            }
        )
        print(
            f"  {contrast}: Δ={ci['estimate']:.4f} "
            f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] p={ci['p_clustered']:.4f} → {gate}"
        )

        # Scaling: OLS + Spearman per moderator
        x1 = sub["num_blocks"].to_numpy(dtype=float)
        x2 = sub["plan_length"].to_numpy(dtype=float)
        ols = ols_two_moderators(delta, x1, x2)
        analysis_rows.append(
            {
                "analysis": "delta_ols_num_blocks_plan_length",
                "contrast": contrast,
                "model": model_id,
                "model_short": short,
                "n_pairs": int(len(sub)),
                "mean_arm_hi": "",
                "mean_arm_lo": "",
                "delta_mean": round(float(np.mean(delta)), 6),
                "ci_low": "",
                "ci_high": "",
                "p_clustered": "",
                "n_clusters": int(len(sub)),
                "greedy_accuracy": round(acc, 6),
                "binary_flag": binary_flag,
                "moderator": "num_blocks+plan_length",
                "spearman_r": "",
                "spearman_ci_low": "",
                "spearman_ci_high": "",
                "spearman_p": "",
                "ols_intercept": round(ols["intercept"], 6),
                "ols_coef_num_blocks": round(ols["coef_num_blocks"], 6),
                "ols_coef_plan_length": round(ols["coef_plan_length"], 6),
                "ols_r2": round(ols["r2"], 6) if ols["r2"] == ols["r2"] else "",
                "interpretation_gate": "",
                "scope_caveat": PREREG["scope_caveat"],
            }
        )
        for moderator, x in (("num_blocks", x1), ("plan_length", x2)):
            sp = cluster_bootstrap_assoc(
                x,
                delta,
                clusters.tolist(),
                kind="spearman",
                n_boot=N_BOOT,
                seed=(hash(f"o14b|{model_id}|{contrast}|{moderator}") % (2**31 - 1))
                or SEED,
            )
            analysis_rows.append(
                {
                    "analysis": "delta_scales_with",
                    "contrast": contrast,
                    "model": model_id,
                    "model_short": short,
                    "n_pairs": int(len(sub)),
                    "mean_arm_hi": "",
                    "mean_arm_lo": "",
                    "delta_mean": round(float(np.mean(delta)), 6),
                    "ci_low": "",
                    "ci_high": "",
                    "p_clustered": "",
                    "n_clusters": sp["n_clusters"],
                    "greedy_accuracy": round(acc, 6),
                    "binary_flag": binary_flag,
                    "moderator": moderator,
                    "spearman_r": round(sp["estimate"], 6)
                    if sp["estimate"] == sp["estimate"]
                    else "",
                    "spearman_ci_low": round(sp["ci_low"], 6)
                    if sp["ci_low"] == sp["ci_low"]
                    else "",
                    "spearman_ci_high": round(sp["ci_high"], 6)
                    if sp["ci_high"] == sp["ci_high"]
                    else "",
                    "spearman_p": round(sp["p_clustered"], 6)
                    if sp["p_clustered"] == sp["p_clustered"]
                    else "",
                    "ols_intercept": "",
                    "ols_coef_num_blocks": "",
                    "ols_coef_plan_length": "",
                    "ols_r2": "",
                    "interpretation_gate": "",
                    "scope_caveat": PREREG["scope_caveat"],
                }
            )

# Headline interpretation (primary contrast A−B)
paired = [
    r
    for r in analysis_rows
    if r["analysis"] == "paired_mean_logprob_delta" and r["contrast"] == "A_minus_B"
]
moves = sum(1 for r in paired if r["interpretation_gate"] == "naming_moves_likelihood")
analysis_rows.append(
    {
        "analysis": "headline",
        "contrast": "A_minus_B",
        "model": "ALL",
        "model_short": "ALL",
        "n_pairs": "",
        "mean_arm_hi": "",
        "mean_arm_lo": "",
        "delta_mean": "",
        "ci_low": "",
        "ci_high": "",
        "p_clustered": "",
        "n_clusters": "",
        "greedy_accuracy": "",
        "binary_flag": "",
        "moderator": "",
        "spearman_r": "",
        "spearman_ci_low": "",
        "spearman_ci_high": "",
        "spearman_p": "",
        "ols_intercept": "",
        "ols_coef_num_blocks": "",
        "ols_coef_plan_length": "",
        "ols_r2": "",
        "interpretation_gate": (
            f"models_with_substantial_naming_effect={moves}/{len(paired)}; "
            + (
                "PRIMARY_FINDING_naming_is_lexical"
                if moves > 0
                else "NULL_COMPATIBLE_L1_survives_on_open_models"
            )
        ),
        "scope_caveat": PREREG["scope_caveat"],
    }
)

out_a = pd.DataFrame(analysis_rows)
out_a.to_csv(O14B_ANALYSIS, index=False)
print(f"\n[wrote] {O14B_ANALYSIS}")
print(out_a[out_a["analysis"].isin(["greedy_accuracy", "paired_mean_logprob_delta", "headline"])][
    ["analysis", "contrast", "model_short", "delta_mean", "ci_low", "ci_high", "p_clustered", "greedy_accuracy", "binary_flag", "interpretation_gate"]
].to_string(index=False))
print("\nSCOPE CAVEAT:", PREREG["scope_caveat"])
'''),
    md("""## Copy into the repo

After the Colab run:

```text
colab_out/O14b_naming_likelihood.csv  →  results/raw/O14b_naming_likelihood.csv
colab_out/O14b_naming_analysis.csv    →  results/derived/O14b_naming_analysis.csv
```

Do **not** overwrite `O14_naming_bank.jsonl` or any API O14 shard paths.
"""),
]


def build() -> Path:
    path = OUT / "o14b_naming_likelihood.ipynb"
    path.write_text(json.dumps(nb(NB_O14B, "o14b_naming_likelihood.ipynb"), indent=1) + "\n")
    print(f"wrote {path} ({len(NB_O14B)} cells)")
    return path


if __name__ == "__main__":
    build()
