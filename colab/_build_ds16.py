#!/usr/bin/env python3
"""Build colab/ds16_recognition_recall.ipynb (Colab T4).

DS-16 / BX-03: recognition vs recall dissociation on Probe-1 canonical + W3.
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


NB_DS16 = [
    md("""# DS-16 — Recognition vs recall (Memory-Signature Suite arm)

**Colab T4 · Phase-0-adjacent · BX-03 / DS-16**

Memory science: **recall** and **recognition** dissociate. A *retrieved* answer
should remain recognizable even when it cannot be produced. A *computed* answer
should show a much smaller recognition–recall gap.

### Per item (canonical + W3)
| Channel | Definition |
|---------|------------|
| **RECALL** | Teacher-forced `mean_logprob` of the gold (O5 path) **and** greedy verify → `recall_correct` |
| **RECOGNITION** | Gold + **k=4** scripted distractors; score each option by `mean_logprob`; gold ranks first? + margin |

Distractors: deterministic module `scripts/consolidate/ds16_distractors.py`
(BW: goal-failing plan mutations; GSM: arithmetic slips; ALGO: wrong-related algorithm outputs). **Not hand-written.**

### Metric
`recognition_recall_gap = recognition_accuracy − recall_accuracy`  
aggregated per `(family, model, variant)`, and item-level for correlations.

### Convergent test (the one that matters)
Canonical→W3 **drop** in the gap vs:
1. Infini-gram `contamination_score` (item-level)
2. C1 intrusion fingerprint (item-level intrusion prevalence on W3 among paper-model C1 errors)

Three independent retrieval signatures agreeing would be the first real
**convergent validity** in this program (notable given prior convergence failures).

### Models
`Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct` — fp16, `sdpa`.

### Outputs
- `colab_out/DS16_recognition_recall.csv` → `results/derived/DS16_recognition_recall.csv`
- `colab_out/DS16_gap_correlations.csv` → `results/derived/DS16_gap_correlations.csv`

### Pre-registered interpretation (freeze before run)
- **Gap drop tracks contamination and/or C1 intrusion** (CI excludes 0, predicted sign: higher contamination / intrusion → larger can−W3 gap drop, i.e. recognition advantage collapses under rename for retrieval-like items) → convergent retrieval signature; report as primary DS-16 result.
- **Null-compatible** → recognition–recall does not add convergent validity with C1/Infini-gram on these open models; still report gaps descriptively.
"""),
    code(SETUP_PIP),
    code(
        SETUP_REPO
        + r'''

# DS-16 knobs
LIMIT_PER_FAMILY = None   # e.g. 4 for smoke (IDs per family × can+W3)
K_DISTRACTORS = 4
MAX_NEW_TOKENS = {"GSM": 256, "ALGO": 192, "BW": 512}
N_BOOT = 5000
SEED = 42
VARIANTS = ("canonical", "W3")
'''
    ),
    md("""## Pre-registration gate"""),
    code(r'''
PREREG = {
    "k_distractors": K_DISTRACTORS,
    "distractor_module": "scripts/consolidate/ds16_distractors.py",
    "recall": "O5 teacher-forced mean_logprob(gold) + greedy verify_answer → recall_correct",
    "recognition": "argmax mean_logprob over gold+k distractors; margin = lp_gold - max(lp_d)",
    "gap": "recognition_correct - recall_correct (item); mean gap per (family,model,variant)",
    "drop": "gap_canonical - gap_W3 (item-level, matched problem_id)",
    "primary_tests": [
        "Spearman(drop, contamination_score) cluster-bootstrap problem_id/clone_family",
        "Spearman(drop, c1_item_intrusion_rate) same inference",
    ],
    "predicted_sign": (
        "positive: high contamination/intrusion → larger drop (recognition advantage "
        "shrinks more under W3 for retrieval-like items)"
    ),
}
for k, v in PREREG.items():
    print(f"{k}: {v}")
assert K_DISTRACTORS == 4 and N_BOOT == 5000
print("[prereg] frozen")
'''),
    md("""## Item queue (canonical + W3) + distractor documentation check"""),
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

from probes.common.clones import algo_cluster_map
from probes.contamination.verify import verify_answer, verify_gsm_answer
from probes.contamination.verify_algo import verify_algo
from scripts.consolidate.ds16_distractors import (
    K_DEFAULT,
    make_distractors,
    recognition_options,
)

assert K_DEFAULT == K_DISTRACTORS

# Document generator in-run (also in module docstring)
print(make_distractors.__doc__ or "(see scripts/consolidate/ds16_distractors.py)")
print("module file:", (REPO_ROOT / "scripts/consolidate/ds16_distractors.py").resolve())

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
    ("Qwen/Qwen2.5-1.5B-Instruct", "fp16"),
    ("Qwen/Qwen2.5-3B-Instruct", "fp16"),
]
MODEL_SHORT = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen1.5B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen3B",
}

OUT_MAIN = OUT_DIR / "DS16_recognition_recall.csv"
OUT_CORR = OUT_DIR / "DS16_gap_correlations.csv"


def _norm_vt(v: str) -> str:
    v = str(v).strip()
    return "canonical" if v.lower() == "canonical" else v.upper()


def _strip_csv_quotes(text: str) -> str:
    s = str(text)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s


def build_prompt(problem_text: str, family: str) -> str:
    return PROBE1_TEMPLATE.format(
        problem=problem_text.strip(),
        family_specific_output_format=FAMILY_FORMAT[family],
    )


def load_items(limit_per_family: int | None) -> list[dict[str, Any]]:
    specs = [
        ("GSM", REPO_ROOT / "data/problems/question_bank_gsm.csv"),
        ("ALGO", REPO_ROOT / "data/problems/question_bank_algo.csv"),
        ("BW", REPO_ROOT / "data/problems/question_bank_bw.csv"),
    ]
    cmap = algo_cluster_map()
    items: list[dict[str, Any]] = []
    for family, path in specs:
        df = pd.read_csv(path, dtype=str).fillna("")
        df["problem_id"] = df["problem_id"].astype(str).str.strip()
        df["variant_type"] = df["variant_type"].map(_norm_vt)
        df["problem_text"] = df["problem_text"].map(_strip_csv_quotes)
        df["correct_answer"] = df["correct_answer"].map(_strip_csv_quotes)
        # family-native IDs only (BW bank can mix)
        if family == "BW":
            df = df[df["problem_id"].str.startswith(("BW_", "MBW_"))]
        elif family == "ALGO":
            df = df[df["problem_id"].str.startswith(("CC_", "SP_", "WIS_"))]
        elif family == "GSM":
            df = df[df["problem_id"].str.startswith("GSM_")]
        sub = df[df["variant_type"].isin(VARIANTS)].copy()
        ids = sorted(sub["problem_id"].unique())
        if limit_per_family is not None:
            ids = ids[: int(limit_per_family)]
        sub = sub[sub["problem_id"].isin(ids)]
        for _, row in sub.iterrows():
            pid = str(row["problem_id"])
            items.append(
                {
                    "family": family,
                    "problem_id": pid,
                    "variant": str(row["variant_type"]),
                    "problem_text": str(row["problem_text"]),
                    "gold": str(row["correct_answer"]),
                    "problem_subtype": str(row.get("problem_subtype", "") or ""),
                    "difficulty_params": str(row.get("difficulty_params", "") or ""),
                    "clone_family": (
                        cmap.get(pid, f"SINGLETON_{pid}")
                        if family == "ALGO"
                        else f"SINGLETON_{pid}"
                    ),
                }
            )
    return items


ITEMS = load_items(LIMIT_PER_FAMILY)
print(f"[queue] {len(ITEMS)} items (LIMIT_PER_FAMILY={LIMIT_PER_FAMILY})")
print(pd.DataFrame(ITEMS).groupby(["family", "variant"]).size().unstack(fill_value=0).to_string())

# Smoke: distractors for one item per family
for fam in ("GSM", "ALGO", "BW"):
    it = next(x for x in ITEMS if x["family"] == fam and x["variant"] == "canonical")
    opts = recognition_options(
        fam, it["problem_id"], it["variant"], it["gold"], it["problem_text"], k=K_DISTRACTORS
    )
    assert sum(o["is_gold"] for o in opts) == 1 and len(opts) == 1 + K_DISTRACTORS
    print(f"[distractors {fam}/{it['problem_id']}] {[o['option_id'] for o in opts]}")
'''),
    md("""## O5 teacher-forcing + greedy (verbatim O5 path)"""),
    code(r'''
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
            "mean_logprob": float(hash(gold_text) % 1000) / -1000.0,
            "gold_first_token_rank": 1,
            "gold_first_token_logprob": 0.0,
            "prompt_n_tokens": n_prompt,
            "sep_note": "DRY_RUN",
        }

    input_ids = torch.tensor([prompt_ids + gold_ids], dtype=torch.long, device=device)
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0]
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
    assert torch.cuda.is_available() or DRY_RUN, "GPU required unless DRY_RUN"
    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN or True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if DRY_RUN:
        print(f"[model] DRY_RUN {model_id}")
        return tok, None, torch.device("cpu")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        token=HF_TOKEN or True,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
    )
    mdl.eval()
    device = next(mdl.parameters()).device
    print(f"[model] {model_id} fp16+sdpa device={device}")
    return tok, mdl, device


def unload(mdl):
    if mdl is None:
        return
    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def greedy_generate(model, tokenizer, device, user_text: str, family: str) -> str:
    if DRY_RUN or model is None:
        return "DRY_RUN_ANSWER"
    prompt = wrap_chat(tokenizer, user_text)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS[family],
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)
    del out, input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text


def verify_correct(item: dict[str, Any], response: str) -> bool:
    family = item["family"]
    problem_id = item["problem_id"]
    gold = item["gold"]
    problem_text = item["problem_text"]
    try:
        if family == "GSM":
            return bool(verify_gsm_answer(response, gold))
        if family == "ALGO":
            ok, _reason, _meta = verify_algo(
                problem_id,
                response,
                gold,
                item.get("problem_subtype", ""),
                item["variant"],
                item.get("difficulty_params", ""),
                problem_text=problem_text,
            )
            return bool(ok)
        vf = (
            "mystery_blocksworld"
            if item.get("problem_subtype") == "mystery_blocksworld"
            or str(problem_id).startswith("MBW_")
            else "blocksworld"
        )
        return bool(
            verify_answer(
                problem_id,
                response,
                gold,
                vf,
                problem_text=problem_text,
            )
        )
    except Exception:
        return False


print("[o5] primitives ready")
'''),
    md("""## Score recognition + recall (log every per-option score)

Resume key: `(model, family, problem_id, variant)`."""),
    code(r'''
ROW_COLUMNS = [
    "family", "problem_id", "variant", "model", "model_short", "clone_family",
    "recall_mean_logprob", "recall_sum_logprob", "recall_n_gold_tokens",
    "recall_correct", "greedy_response",
    "recognition_correct", "recognition_rank_of_gold", "recognition_margin",
    "n_options", "option_id_order",
    # per-option scores (JSON list aligned with option_id_order)
    "option_mean_logprobs", "option_is_gold_flags", "option_texts",
]


def _done_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.is_file():
        return set()
    df = pd.read_csv(path, dtype=str).fillna("")
    return set(zip(df["model"], df["family"], df["problem_id"], df["variant"]))


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    for c in ROW_COLUMNS:
        if c not in new.columns:
            new[c] = ""
    new = new[ROW_COLUMNS]
    if path.is_file():
        old = pd.read_csv(path, dtype=str).fillna("")
        for c in ROW_COLUMNS:
            if c not in old.columns:
                old[c] = ""
        out = pd.concat([old[ROW_COLUMNS], new], ignore_index=True)
    else:
        out = new
    out.to_csv(path, index=False)


done = _done_keys(OUT_MAIN) if RESUME else set()
print(f"[resume] done={len(done)}")

for model_id, quant in MODELS:
    short = MODEL_SHORT[model_id]
    pending = [
        it for it in ITEMS
        if (model_id, it["family"], it["problem_id"], it["variant"]) not in done
    ]
    print(f"\n=== {model_id} pending={len(pending)}/{len(ITEMS)} ===")
    if not pending:
        continue
    tok, mdl, device = load_model(model_id, quant)
    buf: list[dict[str, Any]] = []
    for it in tqdm(pending, desc=short):
        fam = it["family"]
        user = build_prompt(it["problem_text"], fam)
        gold = it["gold"]
        # Recall TF
        tf = teacher_forced_metrics(mdl, tok, device, user, gold)
        greedy = greedy_generate(mdl, tok, device, user, fam)
        recall_ok = verify_correct(it, greedy)
        # Recognition options
        opts = recognition_options(
            fam, it["problem_id"], it["variant"], gold, it["problem_text"], k=K_DISTRACTORS
        )
        scores: list[float] = []
        for o in opts:
            # DRY_RUN: perturb so gold is not always rank-1
            if DRY_RUN:
                base = float(hash(o["text"]) % 1000) / -1000.0
                if o["is_gold"]:
                    base += 0.02
                scores.append(base)
            else:
                m = teacher_forced_metrics(mdl, tok, device, user, o["text"])
                scores.append(float(m["mean_logprob"]))
        gold_idx = next(i for i, o in enumerate(opts) if o["is_gold"])
        best = int(np.nanargmax(np.asarray(scores, dtype=float)))
        recog_ok = best == gold_idx
        gold_lp = scores[gold_idx]
        other = [s for i, s in enumerate(scores) if i != gold_idx]
        margin = float(gold_lp - max(other)) if other else float("nan")
        rank = int(1 + sum(1 for s in scores if s > gold_lp + 1e-15))
        buf.append(
            {
                "family": fam,
                "problem_id": it["problem_id"],
                "variant": it["variant"],
                "model": model_id,
                "model_short": short,
                "clone_family": it["clone_family"],
                "recall_mean_logprob": tf["mean_logprob"],
                "recall_sum_logprob": tf["sum_logprob"],
                "recall_n_gold_tokens": tf["n_gold_tokens"],
                "recall_correct": recall_ok,
                "greedy_response": greedy.replace("\n", "\\n")[:2000],
                "recognition_correct": recog_ok,
                "recognition_rank_of_gold": rank,
                "recognition_margin": round(margin, 6),
                "n_options": len(opts),
                "option_id_order": json.dumps([o["option_id"] for o in opts]),
                "option_mean_logprobs": json.dumps([round(s, 6) for s in scores]),
                "option_is_gold_flags": json.dumps([bool(o["is_gold"]) for o in opts]),
                "option_texts": json.dumps([o["text"] for o in opts]),
            }
        )
        if len(buf) >= 10:
            append_rows(OUT_MAIN, buf)
            buf = []
    append_rows(OUT_MAIN, buf)
    unload(mdl)
    done = _done_keys(OUT_MAIN)

print(f"[wrote] {OUT_MAIN}")
summary = pd.read_csv(OUT_MAIN)
summary["recall_correct"] = summary["recall_correct"].astype(str).str.lower().isin({"true", "1"})
summary["recognition_correct"] = summary["recognition_correct"].astype(str).str.lower().isin({"true", "1"})
summary["gap"] = summary["recognition_correct"].astype(float) - summary["recall_correct"].astype(float)
print(summary.groupby(["model_short", "family", "variant"])[["recall_correct", "recognition_correct", "gap"]].mean().round(3).to_string())
'''),
    md("""## Gaps + convergent correlations

Item-level `drop = gap_can − gap_W3`. Correlate with Infini-gram contamination and
C1 item intrusion rate (fraction of C1 W3 errors on that item that are INTRUSION)."""),
    code(r'''
from probes.common.cluster_inference import cluster_bootstrap_assoc

raw = pd.read_csv(OUT_MAIN, dtype=str).fillna("")
raw["recall_correct"] = raw["recall_correct"].astype(str).str.lower().isin({"true", "1", "yes"})
raw["recognition_correct"] = raw["recognition_correct"].astype(str).str.lower().isin({"true", "1", "yes"})
raw["gap"] = raw["recognition_correct"].astype(float) - raw["recall_correct"].astype(float)
raw["recognition_margin"] = pd.to_numeric(raw["recognition_margin"], errors="coerce")
raw["recall_mean_logprob"] = pd.to_numeric(raw["recall_mean_logprob"], errors="coerce")

# --- aggregate gaps ---
agg_rows: list[dict[str, Any]] = []
for (model, fam, variant), g in raw.groupby(["model", "family", "variant"]):
    agg_rows.append(
        {
            "analysis": "gap_by_cell",
            "family": fam,
            "model": model,
            "model_short": MODEL_SHORT.get(model, model),
            "variant": variant,
            "n": int(len(g)),
            "recall_accuracy": round(float(g["recall_correct"].mean()), 6),
            "recognition_accuracy": round(float(g["recognition_correct"].mean()), 6),
            "recognition_recall_gap": round(float(g["gap"].mean()), 6),
            "mean_recognition_margin": round(float(g["recognition_margin"].mean()), 6),
            "spearman_rho": "",
            "ci_low": "",
            "ci_high": "",
            "p_clustered": "",
            "n_clusters": "",
            "verdict": "",
            "note": "gap = recognition_acc - recall_acc",
        }
    )

# --- item-level drop ---
corr_rows: list[dict[str, Any]] = list(agg_rows)

# Infini-gram
cont_parts = []
for fam, name in (
    ("GSM", "GSM_P3_contamination.csv"),
    ("ALGO", "ALGO_P3_contamination.csv"),
    ("BW", "BW_P3_contamination.csv"),
):
    path = REPO_ROOT / "results/raw" / name
    if not path.is_file():
        print("[warn] missing", path)
        continue
    c = pd.read_csv(path)
    c["family"] = fam
    c["problem_id"] = c["problem_id"].astype(str).str.strip()
    c["contamination_score"] = pd.to_numeric(c["contamination_score"], errors="coerce")
    cont_parts.append(
        c[["family", "problem_id", "contamination_score"]].drop_duplicates(
            ["family", "problem_id"]
        )
    )
cont = pd.concat(cont_parts, ignore_index=True) if cont_parts else pd.DataFrame()

# C1 item intrusion rate on W3
c1_path = REPO_ROOT / "results/derived/C1_intrusion_errors.csv"
if c1_path.is_file():
    c1 = pd.read_csv(c1_path, dtype=str).fillna("")
    c1 = c1[c1["variant"].astype(str).str.upper().eq("W3")].copy()
    c1["is_intrusion"] = c1["error_class"].astype(str).str.upper().eq("INTRUSION")
    c1_rate = (
        c1.groupby(["family", "problem_id"], as_index=False)
        .agg(
            c1_n_errors=("is_intrusion", "size"),
            c1_n_intrusion=("is_intrusion", "sum"),
        )
    )
    c1_rate["c1_item_intrusion_rate"] = c1_rate["c1_n_intrusion"] / c1_rate["c1_n_errors"].clip(lower=1)
else:
    c1_rate = pd.DataFrame(columns=["family", "problem_id", "c1_item_intrusion_rate"])
    print("[warn] missing C1_intrusion_errors.csv")

for model_id, gmod in raw.groupby("model"):
    short = MODEL_SHORT.get(model_id, model_id)
    can = gmod[gmod["variant"] == "canonical"][
        ["family", "problem_id", "clone_family", "gap"]
    ].rename(columns={"gap": "gap_can"})
    w3 = gmod[gmod["variant"] == "W3"][
        ["family", "problem_id", "gap"]
    ].rename(columns={"gap": "gap_w3"})
    merged = can.merge(w3, on=["family", "problem_id"], how="inner")
    merged["drop"] = merged["gap_can"] - merged["gap_w3"]
    if len(cont):
        merged = merged.merge(cont, on=["family", "problem_id"], how="left")
    else:
        merged["contamination_score"] = np.nan
    if len(c1_rate):
        merged = merged.merge(c1_rate[["family", "problem_id", "c1_item_intrusion_rate"]], on=["family", "problem_id"], how="left")
    else:
        merged["c1_item_intrusion_rate"] = np.nan

    for fam, gf in merged.groupby("family"):
        clusters = gf["clone_family"].tolist()
        for xname, predicted in (
            ("contamination_score", "positive"),
            ("c1_item_intrusion_rate", "positive"),
        ):
            sub = gf.dropna(subset=["drop", xname])
            if len(sub) < 8:
                corr_rows.append(
                    {
                        "analysis": "drop_vs_" + xname,
                        "family": fam,
                        "model": model_id,
                        "model_short": short,
                        "variant": "can_minus_W3",
                        "n": int(len(sub)),
                        "recall_accuracy": "",
                        "recognition_accuracy": "",
                        "recognition_recall_gap": "",
                        "mean_recognition_margin": "",
                        "spearman_rho": "",
                        "ci_low": "",
                        "ci_high": "",
                        "p_clustered": "",
                        "n_clusters": "",
                        "verdict": "insufficient_n",
                        "note": f"need ≥8 items; predicted_sign={predicted}",
                    }
                )
                continue
            res = cluster_bootstrap_assoc(
                sub[xname].to_numpy(dtype=float),
                sub["drop"].to_numpy(dtype=float),
                sub["clone_family"].tolist(),
                kind="spearman",
                n_boot=N_BOOT,
                seed=SEED,
            )
            rho = res["estimate"]
            lo, hi = res["ci_low"], res["ci_high"]
            # predicted positive: CI entirely > 0
            if lo == lo and lo > 0:
                verdict = "convergent_positive"
            elif hi == hi and hi < 0:
                verdict = "opposite_sign"
            else:
                verdict = "null_compatible"
            corr_rows.append(
                {
                    "analysis": "drop_vs_" + xname,
                    "family": fam,
                    "model": model_id,
                    "model_short": short,
                    "variant": "can_minus_W3",
                    "n": int(len(sub)),
                    "recall_accuracy": "",
                    "recognition_accuracy": "",
                    "recognition_recall_gap": round(float(sub["drop"].mean()), 6),
                    "mean_recognition_margin": "",
                    "spearman_rho": round(rho, 6) if rho == rho else "",
                    "ci_low": round(lo, 6) if lo == lo else "",
                    "ci_high": round(hi, 6) if hi == hi else "",
                    "p_clustered": round(res["p_clustered"], 6)
                    if res["p_clustered"] == res["p_clustered"]
                    else "",
                    "n_clusters": res["n_clusters"],
                    "verdict": verdict,
                    "note": (
                        f"drop=gap_can-gap_W3; cluster=clone_family; "
                        f"predicted_sign={predicted}; {PREREG['predicted_sign']}"
                    ),
                }
            )
            print(
                f"[{short}/{fam}] drop vs {xname}: ρ={rho:.3f} "
                f"[{lo:.3f},{hi:.3f}] → {verdict} (n={len(sub)})"
            )

# Headline: any convergent_positive?
n_pos = sum(1 for r in corr_rows if r.get("verdict") == "convergent_positive")
n_tests = sum(1 for r in corr_rows if r.get("analysis", "").startswith("drop_vs_"))
corr_rows.append(
    {
        "analysis": "headline",
        "family": "ALL",
        "model": "ALL",
        "model_short": "ALL",
        "variant": "can_minus_W3",
        "n": n_tests,
        "recall_accuracy": "",
        "recognition_accuracy": "",
        "recognition_recall_gap": "",
        "mean_recognition_margin": "",
        "spearman_rho": "",
        "ci_low": "",
        "ci_high": "",
        "p_clustered": "",
        "n_clusters": "",
        "verdict": (
            "CONVERGENT_VALIDITY_SIGNAL"
            if n_pos > 0
            else "NULL_COMPATIBLE_no_convergent_signal"
        ),
        "note": f"n_convergent_positive_cells={n_pos}/{n_tests}",
    }
)

out = pd.DataFrame(corr_rows)
out.to_csv(OUT_CORR, index=False)
# Also copy main into derived-friendly name already OUT_MAIN
print(f"[wrote] {OUT_CORR}")
print(out[out["analysis"] == "headline"][["verdict", "note"]].to_string(index=False))
print(
    out[out["analysis"] == "gap_by_cell"][
        ["model_short", "family", "variant", "recall_accuracy", "recognition_accuracy", "recognition_recall_gap"]
    ].to_string(index=False)
)
'''),
    md("""## Copy into the repo

```text
colab_out/DS16_recognition_recall.csv   →  results/derived/DS16_recognition_recall.csv
colab_out/DS16_gap_correlations.csv     →  results/derived/DS16_gap_correlations.csv
```

Distractor generator (auditable): `scripts/consolidate/ds16_distractors.py`.
"""),
]


def build() -> Path:
    path = OUT / "ds16_recognition_recall.ipynb"
    path.write_text(json.dumps(nb(NB_DS16, "ds16_recognition_recall.ipynb"), indent=1) + "\n")
    print(f"wrote {path} ({len(NB_DS16)} cells)")
    return path


if __name__ == "__main__":
    build()
