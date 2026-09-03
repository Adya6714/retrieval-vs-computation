#!/usr/bin/env python3
"""Build the two standalone Colab notebooks. Run from repo root or colab/.

After a Colab run, copy ``colab_out/`` CSVs into ``results/`` using the names
in ``colab/README.md`` (not space-named browser downloads).
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
    lines = src.splitlines(keepends=True)
    return lines


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


# ---------------------------------------------------------------------------
# Shared setup cells
# ---------------------------------------------------------------------------

SETUP_PIP = r'''
# Colab T4: bitsandbytes for quantized loads. Restart the runtime if
# bitsandbytes was just installed and the kernel has not picked it up.
import sys
import subprocess
subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "transformers>=4.44",
        "accelerate>=0.33",
        "bitsandbytes>=0.43",
        "pandas",
        "scipy",
        "networkx",
        "tqdm",
        "huggingface_hub",
    ]
)
'''

SETUP_REPO = r'''
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ── knobs ────────────────────────────────────────────────────────────────
# Set LIMIT to an int for a smoke test (e.g. 2 items per family). None = full run.
LIMIT = None
DRY_RUN = False          # True: skip GPU, write placeholder rows (pipeline check)
RESUME = True

# Private GitHub clone (Colab secret GITHUB_TOKEN, or env). Public clone works
# without a token. If this notebook is already inside the repo, clone is skipped.
REPO_URL = os.environ.get(
    "RVC_REPO_URL",
    "https://github.com/Adya6714/retrieval-vs-computation.git",
)
REPO_COMMIT = os.environ.get("RVC_REPO_COMMIT", "")  # empty = default branch HEAD

def _secret(name: str) -> str:
    v = os.environ.get(name, "")
    if v:
        return v
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(name) or ""
    except Exception:
        return ""

HF_TOKEN = _secret("HF_TOKEN") or _secret("HUGGING_FACE_HUB_TOKEN")
GH_TOKEN = _secret("GITHUB_TOKEN")

# Llama-3.1-8B-Instruct is gated: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception as _hf_exc:
        print("[setup] huggingface login skipped:", _hf_exc)

def _looks_like_repo(p: Path) -> bool:
    return (p / "probes" / "contamination" / "verify.py").is_file() and (
        p / "data" / "problems" / "question_bank_gsm.csv"
    ).is_file()

def _find_repo() -> Path:
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        if _looks_like_repo(cand):
            return cand
    colab = Path("/content/retrieval-vs-computation")
    if _looks_like_repo(colab):
        return colab
    return colab

REPO_ROOT = _find_repo()
if not _looks_like_repo(REPO_ROOT):
    REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)
    url = REPO_URL
    if GH_TOKEN and "github.com" in url and url.startswith("https://"):
        url = url.replace("https://", f"https://{GH_TOKEN}@")
    print(f"[setup] cloning {REPO_URL} → {REPO_ROOT}")
    cmd = ["git", "clone", "--depth", "1", url, str(REPO_ROOT)]
    subprocess.check_call(cmd)
    if REPO_COMMIT:
        subprocess.check_call(["git", "-C", str(REPO_ROOT), "fetch", "--depth", "1", "origin", REPO_COMMIT])
        subprocess.check_call(["git", "-C", str(REPO_ROOT), "checkout", REPO_COMMIT])

assert _looks_like_repo(REPO_ROOT), (
    f"Could not find probes/ + question banks under {REPO_ROOT}. "
    "Clone the retrieval-vs-computation repo, or set RVC_REPO_URL / GITHUB_TOKEN."
)
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/content/colab_out") if Path("/content").exists() else (REPO_ROOT / "colab_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[setup] REPO_ROOT={REPO_ROOT}")
print(f"[setup] OUT_DIR={OUT_DIR}")
print(f"[setup] LIMIT={LIMIT} DRY_RUN={DRY_RUN} RESUME={RESUME}")
'''

# ===========================================================================
# NOTEBOOK 1
# ===========================================================================

NB1 = [
    md("""# Llama-3.1-8B-Instruct — Probe-1 greedy behavioural (Colab T4)

**Model:** `meta-llama/Llama-3.1-8B-Instruct` · **8-bit (int8 bitsandbytes)** · **greedy** (`do_sample=False`).

Runs Probe-1 **canonical and W3** for GSM, ALGO, and BW using:

1. The **Appendix N** Probe-1 template (paper `\\label{app:prompts}`).
2. The **released verifiers** in `probes/` — imported, not reimplemented.
3. The **OpenRouter answer-extraction rule**: GSM prefers `#### <num>`, else the **last** numeric token (`verify_gsm_answer`). Appendix O / the local-harness bug is routing GSM through `verify_answer(..., family="gsm")`, which uses **first-number** extraction (`_verify_numeric`) and scores 0/44.

Output: `colab_out/llama_greedy_p1.csv` plus a Table-7 comparison.

Greedy is deterministic, so 3 seeds are N/A. A second pass draws **3 samples at T=1.0** on a stratified subset to quantify decoding variance.

**Secrets (Colab → 🔑):** `HF_TOKEN` (gated Llama), optional `GITHUB_TOKEN` if the repo is private."""),
    code(SETUP_PIP),
    code(SETUP_REPO),
    md("""## Appendix O item 7 — extraction rule (must pass before any scoring)

OpenRouter P1 GSM uses `verify_gsm_answer` (via `family="arithmetic_reasoning"`).

Passing `family="gsm"` hits `_verify_numeric`, which takes the **first** number in the string. That is the documented local-harness bug."""),
    code(r'''
import re
from probes.contamination.verify import verify_answer, verify_gsm_answer, _verify_numeric
from probes.contamination.verify_algo import verify_algo

# --- Appendix O item 7: first-number vs last-number / #### -----------------
BUG_RESPONSE = (
    "The hotel charges 0.9 first. Working: 20*0.9=18. Then #### 51"
)
LAST_ONLY = "step1=12 step2=18 the cost is 51"
FIRST_WRONG = "12 is an intermediate; the answer is 51"

assert verify_gsm_answer(BUG_RESPONSE, "51") is True, "#### tag must win"
assert verify_gsm_answer(LAST_ONLY, "51") is True, "last number must win when no ####"
assert verify_gsm_answer(FIRST_WRONG, "51") is True

# family='gsm' is the bug path (first number: 0.9 or 12)
assert verify_answer("GSM_001", BUG_RESPONSE, "51", "gsm") is False, (
    "family='gsm' must NOT be used: it is first-number extraction"
)
assert verify_answer("GSM_001", LAST_ONLY, "51", "gsm") is False
assert _verify_numeric(LAST_ONLY, "51") is False, "first number of LAST_ONLY is 12, not 51"

# OpenRouter / bank routing: arithmetic_reasoning → verify_gsm_answer
assert verify_answer("GSM_001", BUG_RESPONSE, "51", "arithmetic_reasoning") is True
assert verify_answer("GSM_001", LAST_ONLY, "51", "arithmetic_reasoning") is True

print("PASS: GSM scoring uses #### / last-number (OpenRouter). First-number path is asserted-against.")
print("  verify_gsm_answer(BUG_RESPONSE, 51) =", verify_gsm_answer(BUG_RESPONSE, "51"))
print("  verify_answer(..., family='gsm')     =", verify_answer("GSM_001", BUG_RESPONSE, "51", "gsm"), "  ← bug")
print("  verify_answer(..., arithmetic_reasoning) =", verify_answer("GSM_001", BUG_RESPONSE, "51", "arithmetic_reasoning"))
'''),
    md("""## Prompt (Appendix N) and item queue

Paper Appendix (Prompts):

> Solve the following problem exactly and provide only the final answer in the required output format. Problem: `{problem}`. Format instruction: `{family_specific_output_format}`.

The same template is used for canonical and W3; only `{problem}` changes. Instruct models still need the chat template around that user string."""),
    code(r'''
import csv
import json
import random
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

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

TABLE7_LLAMA = {
    # Table 7 Probe-1 Llama cells (OpenRouter). GSM n=20 (GSM_001–020). BW n=65.
    ("GSM", "Can."): 0.800,
    ("GSM", "W3"): 0.150,
    ("BW", "Can."): 0.015,
    ("BW", "W3"): 0.108,
}


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


def build_prompt(problem_text: str, family: str) -> str:
    return PROBE1_TEMPLATE.format(
        problem=problem_text.strip(),
        family_specific_output_format=FAMILY_FORMAT[family],
    )


def load_items(limit: int | None) -> list[dict[str, Any]]:
    """Canonical + W3 for all three families (paired IDs only)."""
    specs = [
        ("GSM", REPO_ROOT / "data/problems/question_bank_gsm.csv"),
        ("ALGO", REPO_ROOT / "data/problems/question_bank_algo.csv"),
        ("BW", REPO_ROOT / "data/problems/question_bank_bw.csv"),
    ]
    items: list[dict[str, Any]] = []
    for family, path in specs:
        df = _load_bank(path)
        can_ids = set(df.loc[df.variant_type == "canonical", "problem_id"])
        w3_ids = set(df.loc[df.variant_type == "W3", "problem_id"])
        paired = sorted(can_ids & w3_ids)
        if family == "GSM":
            # 44 bank IDs (same universe as GSM_P1_behavioral_claude.csv)
            pass
        n_take = paired if limit is None else paired[:limit]
        for pid in n_take:
            for vt in ("canonical", "W3"):
                row = df[(df.problem_id == pid) & (df.variant_type == vt)].iloc[0]
                items.append(
                    {
                        "problem_id": pid,
                        "family": family,
                        "variant": vt,
                        "problem_text": str(row["problem_text"]),
                        "correct_answer": str(row["correct_answer"]),
                        "problem_subtype": str(row.get("problem_subtype", "")).strip().lower(),
                        "difficulty_params": str(row.get("difficulty_params", "{}") or "{}"),
                    }
                )
    return items


def score_item(item: dict, model_answer: str) -> bool:
    """Import-only scoring. GSM must never go through family='gsm'."""
    fam = item["family"]
    if fam == "GSM":
        return bool(verify_gsm_answer(model_answer, item["correct_answer"]))
    if fam == "ALGO":
        ok, _reason, _meta = verify_algo(
            item["problem_id"],
            model_answer,
            item["correct_answer"],
            item["problem_subtype"],
            item["variant"],
            item["difficulty_params"],
        )
        return bool(ok)
    # BW / Mystery BW
    vf = (
        "mystery_blocksworld"
        if item["problem_subtype"] == "mystery_blocksworld"
        or str(item["problem_id"]).startswith("MBW_")
        else "blocksworld"
    )
    return bool(
        verify_answer(
            item["problem_id"],
            model_answer,
            item["correct_answer"],
            vf,
            problem_text=item["problem_text"],
        )
    )


ITEMS = load_items(LIMIT)
print(f"[queue] {len(ITEMS)} prompts")
print(pd.DataFrame(ITEMS).groupby(["family", "variant"]).size().to_string())
'''),
    md("""## Hugging Face login (gated Llama)

Run this **before** the model-load cell. Uses `HF_TOKEN` from Colab secrets if set; otherwise prompts."""),
    code(r'''
from huggingface_hub import login, get_token

_tok = HF_TOKEN or get_token()
if _tok:
    login(token=_tok, add_to_git_credential=False)
    print("[hf] authenticated")
else:
    login()
'''),
    md("""## Google Drive: restore `llama_greedy_p1.csv` if this runtime has no copy

Copies only. Does not use the GPU or rerun inference. Restores from
`MyDrive/llama_outputs/llama_greedy_p1.csv` so `RESUME=True` can skip completed rows."""),
    code(r'''
import shutil
from pathlib import Path

GREEDY_CSV = Path("/content/colab_out/llama_greedy_p1.csv") if Path("/content").exists() else (OUT_DIR / "llama_greedy_p1.csv")
DRIVE_CSV = Path("/content/drive/MyDrive/llama_outputs/llama_greedy_p1.csv")

if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
    except Exception as exc:
        print("[drive] mount skipped:", exc)

OUT_DIR.mkdir(parents=True, exist_ok=True)
if (not GREEDY_CSV.exists() or GREEDY_CSV.stat().st_size == 0) and DRIVE_CSV.exists():
    shutil.copy2(DRIVE_CSV, GREEDY_CSV)
    print(f"[restore] {DRIVE_CSV} -> {GREEDY_CSV}  ({GREEDY_CSV.stat().st_size} bytes)")
elif GREEDY_CSV.exists():
    print(f"[restore] local CSV already present: {GREEDY_CSV}  ({GREEDY_CSV.stat().st_size} bytes)")
else:
    print(f"[restore] no local or Drive CSV yet; greedy sweep will write {GREEDY_CSV}")
'''),
    md("""## Load Llama-3.1-8B-Instruct in 8-bit (T4 16GB)

Skip this cell if `llama_greedy_p1.csv` is already recovered — the sweep resumes completed rows and the summary does not need the GPU."""),
    code(r'''
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
COMPUTE_DTYPE = torch.float16  # T4 has no native bfloat16

# GSM was 128; 23/37 wrong canonical answers in llama_greedy_p1.csv end mid-token
# (correct median 74 chars, incorrect median 444, max 552). G4 reruns at 768.
MAX_NEW = {"GSM": 768, "ALGO": 192, "BW": 512}

tokenizer = None
model = None
DEVICE = "cpu"

if not DRY_RUN:
    assert torch.cuda.is_available(), "This notebook expects a GPU (Colab T4)."
    print("[gpu]", torch.cuda.get_device_name(0), "mem_GB", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN or True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=COMPUTE_DTYPE,
        token=HF_TOKEN or True,
    )
    model.eval()
    DEVICE = next(model.parameters()).device
    print("[model] loaded 8-bit", MODEL_ID, "device", DEVICE)
else:
    print("[dry-run] skipping model load")


def wrap_chat(user_text: str) -> str:
    assert tokenizer is not None
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        tokenize=False,
    )


@torch.inference_mode()
def generate(user_text: str, family: str, *, do_sample: bool, temperature: float | None = None) -> str:
    if DRY_RUN:
        return "#### 0" if family == "GSM" else "DRY_RUN"
    prompt = wrap_chat(user_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    gen_kwargs: dict = dict(
        max_new_tokens=MAX_NEW[family],
        pad_token_id=tokenizer.pad_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature if temperature is not None else 1.0)
        gen_kwargs["top_p"] = 1.0
    out = model.generate(**inputs, **gen_kwargs)
    new = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new, skip_special_tokens=True).strip()
'''),
    md("""## Greedy sweep → `colab_out/llama_greedy_p1.csv`

`RESUME=True` skips completed `(problem_id, family, variant)` rows in this file.
That **does not** re-run truncated GSM canonical answers (25/37 wrong answers
are exactly 128 tokens). The 768-token GSM canonical rerun is a **later cell**
and writes a **new** CSV. Never overwrite `llama_greedy_p1.csv` with
the 768 run."""),
    code(r'''
GREEDY_CSV = OUT_DIR / "llama_greedy_p1.csv"
GREEDY_COLS = ["problem_id", "family", "variant", "model_answer", "correct"]

def _done_keys(path: Path) -> set[tuple[str, str, str]]:
    if not (RESUME and path.exists() and path.stat().st_size > 0):
        return set()
    prev = pd.read_csv(path, dtype=str)
    return {
        (str(r.problem_id), str(r.family), str(r.variant))
        for _, r in prev.iterrows()
    }

done = _done_keys(GREEDY_CSV)
write_header = not GREEDY_CSV.exists() or GREEDY_CSV.stat().st_size == 0
if not RESUME and GREEDY_CSV.exists():
    GREEDY_CSV.unlink()
    write_header = True
    done = set()

n_ok = n_done = 0
if RESUME and GREEDY_CSV.exists() and GREEDY_CSV.stat().st_size > 0:
    prev = pd.read_csv(GREEDY_CSV, dtype=str)
    n_done = len(prev)
    n_ok = int(prev["correct"].astype(str).str.lower().isin(["true", "1"]).sum())

with GREEDY_CSV.open("a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=GREEDY_COLS)
    if write_header:
        w.writeheader()
    for item in tqdm(ITEMS, desc="greedy"):
        key = (item["problem_id"], item["family"], item["variant"])
        if key in done:
            continue
        user = build_prompt(item["problem_text"], item["family"])
        ans = generate(user, item["family"], do_sample=False)
        correct = bool(score_item(item, ans))
        w.writerow(
            {
                "problem_id": item["problem_id"],
                "family": item["family"],
                "variant": item["variant"],
                "model_answer": ans,
                "correct": correct,
            }
        )
        f.flush()
        n_done += 1
        n_ok += int(correct)
        done.add(key)

print(f"wrote {GREEDY_CSV}  running_acc={n_ok}/{n_done}")
greedy_df = pd.read_csv(GREEDY_CSV, dtype=str)
greedy_df["correct_bool"] = greedy_df["correct"].astype(str).str.lower().isin(["true", "1"])
print(greedy_df.groupby(["family", "variant"])["correct_bool"].agg(["mean", "sum", "count"]))
'''),
    md("""## Google Drive backup (copy only — no GPU, no inference)"""),
    code(r'''
from pathlib import Path
import shutil

src = Path("/content/colab_out/llama_greedy_p1.csv")
drive_dir = Path("/content/drive/MyDrive/llama_outputs")

if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
    except Exception as exc:
        print("[drive] mount skipped:", exc)

drive_dir.mkdir(parents=True, exist_ok=True)

dst = drive_dir / "llama_greedy_p1.csv"

if src.exists():
    shutil.copy2(src, dst)
    print(f"[backup] copied {src} -> {dst}  ({dst.stat().st_size} bytes)")
else:
    print(f"[backup] source not found: {src}")

# Also pull the files onto your laptop (Colab → browser download).
try:
    from google.colab import files as _colab_files  # type: ignore
    for p in (src, Path("/content/colab_out/llama_greedy_p1_manifest.json")):
        if p.exists():
            _colab_files.download(str(p))
            print(f"[download] {p.name}")
except Exception as exc:
    print("[download] skipped (not Colab or download blocked):", exc)
'''),
    md("""## H6 — GSM canonical greedy rerun at 768 tokens (new file)

The original `llama_greedy_p1.csv` GSM canonical cell is 7/44 = 0.159 because
generation was capped at 128 tokens. This cell reruns **GSM canonical only**
at `max_new_tokens=768` and writes `llama_greedy_p1_gsm_canonical_768.csv`.
It does **not** read or write `llama_greedy_p1.csv`, so `RESUME=True` on the
truncated file cannot skip these rows.

Requires GPU. Copy the new CSV into `results/raw/llama_greedy_p1_gsm_canonical_768.csv`
after the run. Until it exists, there is no local-vs-OpenRouter GSM greedy comparison."""),
    code(r'''
from pathlib import Path
import csv

GSM_768_CSV = OUT_DIR / "llama_greedy_p1_gsm_canonical_768.csv"
if GSM_768_CSV.resolve() == (OUT_DIR / "llama_greedy_p1.csv").resolve():
    raise SystemExit("Refusing to overwrite llama_greedy_p1.csv")

gsm_can_items = [
    it for it in ITEMS
    if it["family"] == "GSM" and str(it["variant"]).lower() == "canonical"
]
print(f"[H6] GSM canonical queue n={len(gsm_can_items)}  out={GSM_768_CSV}")

def _done_768(path: Path) -> set[str]:
    if not (path.exists() and path.stat().st_size > 0):
        return set()
    prev = pd.read_csv(path, dtype=str)
    return set(prev["problem_id"].astype(str))

done768 = _done_768(GSM_768_CSV)
write_header768 = not GSM_768_CSV.exists() or GSM_768_CSV.stat().st_size == 0
n_ok768 = n_done768 = 0
if GSM_768_CSV.exists() and GSM_768_CSV.stat().st_size > 0:
    prev = pd.read_csv(GSM_768_CSV, dtype=str)
    n_done768 = len(prev)
    n_ok768 = int(prev["correct"].astype(str).str.lower().isin(["true", "1"]).sum())

COLS768 = ["problem_id", "family", "variant", "model_answer", "correct", "max_new_tokens"]
with GSM_768_CSV.open("a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS768)
    if write_header768:
        w.writeheader()
    for item in tqdm(gsm_can_items, desc="gsm_canonical_768"):
        pid = str(item["problem_id"])
        if pid in done768:
            continue
        user = build_prompt(item["problem_text"], "GSM")
        ans = generate(user, "GSM", do_sample=False)
        correct = bool(score_item(item, ans))
        w.writerow(
            {
                "problem_id": pid,
                "family": "GSM",
                "variant": "canonical",
                "model_answer": ans,
                "correct": correct,
                "max_new_tokens": 768,
            }
        )
        f.flush()
        n_done768 += 1
        n_ok768 += int(correct)
        done768.add(pid)

print(f"wrote {GSM_768_CSV}  running_acc={n_ok768}/{n_done768}")
if GSM_768_CSV.exists() and GSM_768_CSV.stat().st_size > 0:
    g768 = pd.read_csv(GSM_768_CSV, dtype=str)
    g768["correct_bool"] = g768["correct"].astype(str).str.lower().isin(["true", "1"])
    print(g768["correct_bool"].agg(["mean", "sum", "count"]))
'''),
    md("""## Google Drive backup of the 768 GSM CSV (copy only)"""),
    code(r'''
from pathlib import Path
import shutil

src768 = OUT_DIR / "llama_greedy_p1_gsm_canonical_768.csv"
drive_dir = Path("/content/drive/MyDrive/llama_outputs")
if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
    except Exception as exc:
        print("[drive] mount skipped:", exc)
drive_dir.mkdir(parents=True, exist_ok=True)
if src768.exists():
    dst768 = drive_dir / "llama_greedy_p1_gsm_canonical_768.csv"
    shutil.copy2(src768, dst768)
    print(f"[backup] copied {src768} -> {dst768}  ({dst768.stat().st_size} bytes)")
    try:
        from google.colab import files as _colab_files  # type: ignore
        _colab_files.download(str(src768))
        print(f"[download] {src768.name}")
    except Exception as exc:
        print("[download] skipped:", exc)
else:
    print(f"[backup] 768 GSM CSV not found: {src768} (run the H6 cell on a GPU runtime)")
'''),
    md("""## Summary vs Table 7 (Llama OpenRouter)

Table 7 GSM Llama is **n=20** (GSM_001–020). This notebook scores the full **n=44** bank; both slices are printed. ALGO Table 7 is per subtype×instance-type — we print pooled can/W3 plus a subtype breakdown when `difficulty_params` is present."""),
    code(r'''
def acc_table(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["family", "variant"], as_index=False)["correct_bool"]
        .agg(n="count", n_correct="sum", acc="mean")
    )
    return g.sort_values(["family", "variant"])

summary = acc_table(greedy_df)
print("=== Local greedy (this notebook) ===")
print(summary.to_string(index=False))

print("\n=== vs Table 7 Llama OpenRouter (Can / W3) ===")
rows = []
for fam in ("GSM", "ALGO", "BW"):
    sub = greedy_df[greedy_df.family == fam]
    for vt, t7k in (("canonical", "Can."), ("W3", "W3")):
        s = sub[sub.variant == vt]
        local = float(s.correct_bool.mean()) if len(s) else float("nan")
        t7 = TABLE7_LLAMA.get((fam, t7k), float("nan"))
        rows.append(
            {
                "family": fam,
                "variant": vt,
                "n_local": len(s),
                "acc_local_greedy": None if pd.isna(local) else round(local, 3),
                "acc_table7_llama_openrouter": None if pd.isna(t7) else t7,
            }
        )
print(pd.DataFrame(rows).to_string(index=False))

# GSM bank-valid n=20 slice (matches Table 7 denominator)
gsm20 = greedy_df[
    (greedy_df.family == "GSM")
    & (greedy_df.problem_id.str.extract(r"GSM_(\d+)", expand=False).astype(float) <= 20)
]
if len(gsm20):
    print("\n=== GSM_001–020 only (Table 7 n=20) ===")
    print(acc_table(gsm20).to_string(index=False))

# ALGO subtype breakdown (Table 7 slices)
algo_items = { (it["problem_id"], it["variant"]): it for it in ITEMS if it["family"] == "ALGO" }
if algo_items:
    parts = []
    for _, r in greedy_df[greedy_df.family == "ALGO"].iterrows():
        it = algo_items.get((r.problem_id, r.variant))
        if not it:
            continue
        try:
            dp = json.loads(it["difficulty_params"] or "{}")
        except json.JSONDecodeError:
            dp = {}
        parts.append(
            {
                "subtype": it["problem_subtype"],
                "instance_type": dp.get("instance_type", ""),
                "variant": r.variant,
                "correct_bool": r.correct_bool,
            }
        )
    if parts:
        adf = pd.DataFrame(parts)
        print("\n=== ALGO by subtype × instance_type ===")
        print(
            adf.groupby(["subtype", "instance_type", "variant"])["correct_bool"]
            .agg(["mean", "sum", "count"])
            .to_string()
        )
'''),
    md("""## Decoding variance — 3 samples at `temperature=1.0`

Greedy has no seed axis. This cell samples the same Appendix-N prompt three times at T=1.0 on a **stratified subset** (2 problem IDs × canonical+W3 × 3 families = 12 items × 3 = 36 generations). Set `FULL_VARIANCE=True` to sample every greedy item (slow on a free T4)."""),
    code(r'''
FULL_VARIANCE = False
VARIANCE_IDS_PER_FAMILY = 2
N_SAMPLES = 3
TEMP = 1.0
VAR_CSV = OUT_DIR / "llama_temp1_variance.csv"
VAR_COLS = [
    "problem_id", "family", "variant", "sample_id", "model_answer", "correct",
]

rng = random.Random(0)
var_items: list[dict] = []
if FULL_VARIANCE:
    var_items = list(ITEMS)
else:
    by_fam: dict[str, list[str]] = {}
    for it in ITEMS:
        if it["variant"] == "canonical":
            by_fam.setdefault(it["family"], []).append(it["problem_id"])
    chosen = []
    for fam, pids in by_fam.items():
        uniq = sorted(set(pids))
        k = min(VARIANCE_IDS_PER_FAMILY, len(uniq))
        sampled = rng.sample(uniq, k) if k else []
        chosen.extend((fam, pid) for pid in sampled)
    want = {(fam, pid) for fam, pid in chosen}
    var_items = [it for it in ITEMS if (it["family"], it["problem_id"]) in want]

print(f"[variance] {len(var_items)} items × {N_SAMPLES} samples at T={TEMP}")

var_header = not VAR_CSV.exists() or VAR_CSV.stat().st_size == 0
var_done = set()
if RESUME and VAR_CSV.exists() and VAR_CSV.stat().st_size > 0:
    vprev = pd.read_csv(VAR_CSV, dtype=str)
    var_done = {
        (str(r.problem_id), str(r.family), str(r.variant), str(r.sample_id))
        for _, r in vprev.iterrows()
    }

with VAR_CSV.open("a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=VAR_COLS)
    if var_header:
        w.writeheader()
    for item in tqdm(var_items, desc="T=1.0"):
        user = build_prompt(item["problem_text"], item["family"])
        for sid in range(N_SAMPLES):
            key = (item["problem_id"], item["family"], item["variant"], str(sid))
            if key in var_done:
                continue
            ans = generate(user, item["family"], do_sample=True, temperature=TEMP)
            w.writerow(
                {
                    "problem_id": item["problem_id"],
                    "family": item["family"],
                    "variant": item["variant"],
                    "sample_id": sid,
                    "model_answer": ans,
                    "correct": bool(score_item(item, ans)),
                }
            )
            f.flush()

vdf = pd.read_csv(VAR_CSV, dtype=str)
vdf["correct_bool"] = vdf["correct"].astype(str).str.lower().isin(["true", "1"])
# pairwise exact-match of raw strings across the 3 samples
agree_rows = []
for (pid, fam, vt), g in vdf.groupby(["problem_id", "family", "variant"]):
    texts = g.sort_values("sample_id")["model_answer"].astype(str).tolist()
    accs = g.sort_values("sample_id")["correct_bool"].tolist()
    n_unique = len(set(texts))
    agree_rows.append(
        {
            "problem_id": pid,
            "family": fam,
            "variant": vt,
            "n_unique_strings": n_unique,
            "all_three_identical": n_unique == 1,
            "acc_mean": float(sum(accs) / max(len(accs), 1)),
            "acc_min": float(min(accs) if accs else 0),
            "acc_max": float(max(accs) if accs else 0),
        }
    )
adf = pd.DataFrame(agree_rows)
print("\n=== T=1.0 variance ===")
print(f"items={len(adf)}  fraction all-3-identical={adf.all_three_identical.mean():.3f}")
print(adf.groupby("family")[["all_three_identical", "acc_mean", "n_unique_strings"]].mean())
print(f"wrote {VAR_CSV}")
'''),
    md("""## Manifest"""),
    code(r'''
import subprocess

def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:
        return f"unavailable ({exc})"

quant = "int8 bitsandbytes" if not DRY_RUN else "DRY_RUN"
dtype = str(COMPUTE_DTYPE) if not DRY_RUN else "n/a"
n_items = int(len(greedy_df.drop_duplicates(["problem_id", "family", "variant"]))) if "greedy_df" in dir() else len(ITEMS)

manifest = {
    "notebook": "llama_greedy_behavioural.ipynb",
    "model_string": MODEL_ID,
    "dtype": dtype,
    "quantization": quant,
    "decoding_config": {
        "greedy": {"do_sample": False, "temperature": None},
        "variance_pass": {
            "do_sample": True,
            "temperature": 1.0,
            "n_samples": 3,
            "full_variance": FULL_VARIANCE,
        },
    },
    "n_items": n_items,
    "n_greedy_rows": int(len(greedy_df)) if "greedy_df" in dir() else None,
    "families": ["GSM", "ALGO", "BW"],
    "variants": ["canonical", "W3"],
    "prompt": "Appendix N Probe-1 + Llama-3.1 Instruct chat template",
    "verifier": "probes.contamination.verify.verify_gsm_answer / verify_answer(blocksworld) / verify_algo",
    "extraction_rule": "GSM: #### tag else last number (OpenRouter). family='gsm' first-number path asserted-against.",
    "git_commit_hash": git_hash(),
    "output_csv": str(GREEDY_CSV),
}
print("=== MANIFEST ===")
print(json.dumps(manifest, indent=2))
(OUT_DIR / "llama_greedy_p1_manifest.json").write_text(json.dumps(manifest, indent=2))

_out_files = [
    OUT_DIR / "llama_greedy_p1.csv",
    OUT_DIR / "llama_greedy_p1_manifest.json",
]
_drive_dir = Path("/content/drive/MyDrive/llama_outputs")
if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
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

# ===========================================================================
# NOTEBOOK 2
# ===========================================================================

NB2 = [
    md("""# Frequency-controlled mechanistic readout — Run 2: ALGO + BW only

Llama-3.1-8B and Qwen2.5-7B: **4-bit NF4**. Qwen2.5-1.5B: **unquantized** (bf16 if supported, else fp16 on T4).

**This run does not queue GSM.** Existing GSM results live in `colab_out/mech_freq_controlled.csv` and must not be overwritten. This notebook writes:

`colab_out/mech_freq_controlled_algo_bw.csv`

**Items** (canonical + W3 only):

- **ALGO:** frozen 61-problem adversarial pool (34 SP + 10 CC + 17 WIS). Not the full 110-ID bank.
- **BW:** 65 PlanBench bank IDs.

Gold is answer **content**, not format scaffolding (Appendix H): ALGO cost/count/total; BW first action word. Families where more than half the golds are the same token are flagged **degenerate** and Wilcoxon is not reported.

`RESUME=True` skips completed `(model, family, problem_id, variant)` rows in the ALGO+BW CSV.

**Secrets:** `HF_TOKEN` (gated Llama)."""),
    code(SETUP_PIP),
    code(SETUP_REPO),
    md("""## IDs, Appendix-N prompt, content-gold (not format keywords)"""),
    code(r'''
from __future__ import annotations

import csv
import gc
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROBE1_TEMPLATE = (
    "Solve the following problem exactly and provide only the final answer "
    "in the required output format. Problem: {problem}. Format instruction: "
    "{family_specific_output_format}."
)
GSM_FORMAT = (
    "Write the final numerical answer on its own line as #### <number>. "
    "No other text after that tag."
)
ALGO_FORMAT = (
    "Follow the problem's required output format exactly "
    "(Path: / Count: / Selected: or Total: / Scoops:). No explanation."
)
BW_FORMAT = (
    "A numbered list of actions only. Each action must be one of the "
    "permitted operators with their arguments. No explanation."
)
FAMILY_FORMAT = {"GSM": GSM_FORMAT, "ALGO": ALGO_FORMAT, "BW": BW_FORMAT}

# Frozen ALGO adversarial pool (rebuild/FROZEN_FILTERS.md). Not bank instance_type.
ALGO_ADV = {
    "CC": [f"CC_{i:02d}" for i in range(1, 11)],
    "SP": [
        "SP_003", "SP_004", "SP_005", "SP_019", "SP_020", "SP_021", "SP_023",
        "SP_024", "SP_026", "SP_027", "SP_028", "SP_029", "SP_030", "SP_037",
        "SP_038", "SP_039", "SP_040", "SP_042", "SP_044", "SP_045", "SP_046",
        "SP_047", "SP_048", "SP_062", "SP_063", "SP_064", "SP_065", "SP_066",
        "SP_068", "SP_069", "SP_070", "SP_071", "SP_072", "SP_073",
    ],
    "WIS": [
        "WIS_003", "WIS_004", "WIS_013", "WIS_014", "WIS_015", "WIS_016",
        "WIS_017", "WIS_018", "WIS_019", "WIS_020", "WIS_023", "WIS_024",
        "WIS_025", "WIS_026", "WIS_027", "WIS_028", "WIS_029",
    ],
}
ALGO_ADV_IDS = ALGO_ADV["CC"] + ALGO_ADV["SP"] + ALGO_ADV["WIS"]
assert len(ALGO_ADV_IDS) == 61, len(ALGO_ADV_IDS)

# Tokens that are format scaffolding, not answer content (Appendix H confound).
FORMAT_KEYWORDS = {
    "path", "count", "selected", "coins", "scoops", "total", "answer",
    "final", "####", "#", ":", "[", "]", "{", "}", ",",
    "path:", "count:", "selected:", "coins:", "scoops:",
}

CLAUDE_P1 = REPO_ROOT / "results/raw/GSM_P1_behavioral_claude.csv"  # GSM already computed; not queued
GSM_BANK = REPO_ROOT / "data/problems/question_bank_gsm.csv"
ALGO_BANK = REPO_ROOT / "data/problems/question_bank_algo.csv"
BW_BANK = REPO_ROOT / "data/problems/question_bank_bw.csv"

GSM_CSV_DO_NOT_TOUCH = (
    Path("/content/colab_out/mech_freq_controlled.csv")
    if Path("/content").exists()
    else (OUT_DIR / "mech_freq_controlled.csv")
)
print(f"[gsm] leave untouched: {GSM_CSV_DO_NOT_TOUCH}  exists={GSM_CSV_DO_NOT_TOUCH.exists()}")


def _norm_bank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["problem_id"] = df["problem_id"].astype(str).str.strip()
    vt_col = "variant_type" if "variant_type" in df.columns else "variant"
    df["variant_type"] = df[vt_col].astype(str).str.strip()
    df.loc[df.variant_type.str.lower() == "canonical", "variant_type"] = "canonical"
    return df


gsm_df = _norm_bank(GSM_BANK)
algo_df = _norm_bank(ALGO_BANK)
bw_df = _norm_bank(BW_BANK)
# aliases used by the unigram counter
bank, algo_bank, bw_bank = gsm_df, algo_df, bw_df


def _paired_ids(df: pd.DataFrame) -> list[str]:
    can = set(df.loc[df.variant_type == "canonical", "problem_id"])
    w3 = set(df.loc[df.variant_type == "W3", "problem_id"])
    return sorted(can & w3)


# Frozen adversarial 61 (not all 110 unique ALGO bank IDs). GSM is not queued.
_bank_algo_unique = sorted(algo_df["problem_id"].unique())
ALGO_IDS = [pid for pid in ALGO_ADV_IDS if pid in set(_paired_ids(algo_df))]
BW_IDS = _paired_ids(bw_df)
assert len(ALGO_IDS) == 61, len(ALGO_IDS)
assert len(BW_IDS) == 65, f"expected 65 BW bank IDs, got {len(BW_IDS)}"
print(f"[banks] ALGO={len(ALGO_IDS)} (frozen adv; bank unique={len(_bank_algo_unique)}) BW={len(BW_IDS)}")
print("[banks] GSM_IDS not used for this run")


def bank_row_from(df: pd.DataFrame, pid: str, vt: str, label: str) -> pd.Series:
    sub = df[(df.problem_id == pid) & (df.variant_type == vt)]
    if sub.empty:
        raise KeyError(f"{pid}/{vt} missing from {label} bank")
    return sub.iloc[0]


def gsm_gold_content(correct_answer: str) -> str:
    """Numeric answer-content span. Never #### / Path / Count scaffolding."""
    s = str(correct_answer).strip()
    s = re.sub(r"^####\s*", "", s)
    s = s.replace(",", "")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        m = re.findall(r"-?\d+(?:\.\d+)?", s)
        if not m:
            raise ValueError(f"no numeric gold in {correct_answer!r}")
        return m[-1]


def algo_gold_content(problem_id: str, correct_answer: str) -> str:
    """Answer-content numeric token: SP cost, CC count, WIS total.

    Not Path / Coins / Selected (Appendix H format-keyword confound).
    W3 coin-change uses ``Total:`` instead of ``Count:``.
    """
    s = str(correct_answer)
    pid = str(problem_id).strip().upper()
    if pid.startswith("SP"):
        m = re.search(r"Cost\s*:\s*(-?\d+)", s, flags=re.I)
        if not m:
            raise ValueError(f"{problem_id}: no Cost: gold in {s!r}")
        return m.group(1)
    if pid.startswith("CC"):
        m = re.search(r"(?:Count|Total)\s*:\s*(-?\d+)", s, flags=re.I)
        if not m:
            raise ValueError(f"{problem_id}: no Count:/Total: gold in {s!r}")
        return m.group(1)
    if pid.startswith("WIS"):
        m = re.search(r"Total\s*:\s*(-?\d+)", s, flags=re.I)
        if not m:
            raise ValueError(f"{problem_id}: no Total: gold in {s!r}")
        return m.group(1)
    raise ValueError(f"{problem_id}: unknown ALGO subtype")


def bw_gold_content(correct_answer: str) -> str:
    """First action word of the gold plan (pick-up / unstack / attack / …)."""
    lines = [ln.strip() for ln in str(correct_answer).splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty BW gold plan")
    line = re.sub(r"^\d+[\.)]\s*", "", lines[0])
    m = re.match(r"([A-Za-z][A-Za-z0-9_-]*)", line)
    if not m:
        raise ValueError(f"no action word in {lines[0]!r}")
    return m.group(1)


def build_user(problem_text: str, family: str = "GSM") -> str:
    return PROBE1_TEMPLATE.format(
        problem=str(problem_text).strip(),
        family_specific_output_format=FAMILY_FORMAT[family],
    )


# Run 2 queue: ALGO + BW only. GSM is not queued and will not be recalculated.
ITEMS: list[dict] = []

for pid in ALGO_IDS:
    for vt in ("canonical", "W3"):
        r = bank_row_from(algo_df, pid, vt, "ALGO")
        ITEMS.append(
            {
                "family": "ALGO",
                "problem_id": pid,
                "variant": vt,
                "problem_text": str(r["problem_text"]),
                "correct_answer": str(r["correct_answer"]),
                "gold_content": algo_gold_content(pid, r["correct_answer"]),
            }
        )

for pid in BW_IDS:
    for vt in ("canonical", "W3"):
        r = bank_row_from(bw_df, pid, vt, "BW")
        ITEMS.append(
            {
                "family": "BW",
                "problem_id": pid,
                "variant": vt,
                "problem_text": str(r["problem_text"]),
                "correct_answer": str(r["correct_answer"]),
                "gold_content": bw_gold_content(r["correct_answer"]),
            }
        )

if LIMIT is not None:
    keep = {("ALGO", pid) for pid in ALGO_IDS[:LIMIT]} | {("BW", pid) for pid in BW_IDS[:LIMIT]}
    ITEMS = [x for x in ITEMS if (x["family"], x["problem_id"]) in keep]

print(
    f"[queue] {len(ITEMS)} items "
    f"({len({x['problem_id'] for x in ITEMS if x['family']=='ALGO'})} ALGO + "
    f"{len({x['problem_id'] for x in ITEMS if x['family']=='BW'})} BW × 2 variants)"
)
assert not any(x["family"] == "GSM" for x in ITEMS), "GSM must not be in the Run 2 queue"
for fam in ("ALGO", "BW"):
    sub = [x for x in ITEMS if x["family"] == fam]
    vc = Counter(x["gold_content"] for x in sub)
    can_vc = Counter(x["gold_content"] for x in sub if x["variant"] == "canonical")
    if vc:
        modal, n_m = vc.most_common(1)[0]
        frac = n_m / len(sub)
        can_modal, can_n = can_vc.most_common(1)[0]
        can_n_ids = sum(can_vc.values())
        can_frac = can_n / max(can_n_ids, 1)
        flag = " DEGENERATE" if (frac > 0.5 or can_frac > 0.5) else ""
        print(
            f"  {fam}: modal={modal!r} {n_m}/{len(sub)}={frac:.1%}; "
            f"canonical {can_modal!r} {can_n}/{can_n_ids}={can_frac:.1%}{flag}"
        )
'''),
    md("""## Token targeting + unigram frequency proxy

Target id is the **first BPE of the gold content that would follow the chat prompt** (joint encode), not an isolated `'5'` vs `' 5'` mismatch.

Frequency proxy (no Pile dump on Colab):

1. Tokenize every GSM + ALGO + BW bank `problem_text` + `correct_answer` with **this model's tokenizer** → unigram counts.
2. Also record `token_id` (BPE id rank) as a secondary proxy.

Terciles are computed **within (model, family)** over unique canonical gold-token unigram counts.

A family is **degenerate** if more than half of its golds (all items, or canonical-only) are the same token. Wilcoxon is not reported for a degenerate family."""),
    code(r'''
def wrap_chat(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        tokenize=False,
    )


def resolve_target_token(tokenizer, prompt: str, answer: str) -> tuple[int, str, list[int], str]:
    """Prompt-aware first token of `answer` after `prompt` (same as mechanistic scripts)."""
    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    prompt_ids = enc(prompt)
    answer_ids_bare = enc(answer)
    candidates: list[tuple[str, int, int, list[int]]] = []
    for sep in ("", " "):
        joint = enc(prompt + sep + answer)
        if len(joint) <= len(prompt_ids):
            continue
        if joint[: len(prompt_ids)] != prompt_ids:
            continue
        rest = joint[len(prompt_ids) :]
        candidates.append((sep, int(rest[0]), len(joint), rest))
    if not candidates:
        if not answer_ids_bare:
            return -1, "", [], "EMPTY"
        tid = int(answer_ids_bare[0])
        return tid, tokenizer.decode([tid]), answer_ids_bare, "FALLBACK"
    candidates.sort(key=lambda c: c[2])
    sep, tid, _, rest = candidates[0]
    return tid, tokenizer.decode([tid]), rest, repr(sep)


def assert_content_gold(decoded: str, family: str) -> None:
    d = decoded.strip().lower()
    compact = d.replace(" ", "")
    if compact in FORMAT_KEYWORDS or d in FORMAT_KEYWORDS:
        raise AssertionError(f"gold token is a format keyword: {decoded!r}")
    if family in {"GSM", "ALGO"}:
        if not re.search(r"\d", decoded):
            raise AssertionError(
                f"{family} content-gold token must contain a digit, got {decoded!r}"
            )


def bank_unigram_counter(tokenizer) -> Counter:
    """Training-proxy: question-bank token unigrams (this tokenizer)."""
    c: Counter = Counter()
    for df in (bank, algo_bank, bw_bank):
        for _, r in df.iterrows():
            text = f"{r['problem_text']}\n{r['correct_answer']}"
            ids = tokenizer.encode(str(text), add_special_tokens=False)
            c.update(ids)
    return c


UNQUANTIZED = {
    "Qwen/Qwen2.5-1.5B-Instruct",  # fits unquantized on T4; 7B/8B stay nf4
}


def load_model(model_id: str):
    """4-bit NF4 for 7B/8B; unquantized bf16 (fp16 fallback) for the 1.5B."""
    assert torch.cuda.is_available(), "GPU required (Colab T4)."
    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN or True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if model_id in UNQUANTIZED:
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            dtype_label = "bfloat16"
        else:
            dtype = torch.float16
            dtype_label = "float16"
            print(
                f"[model] {model_id}: GPU lacks native bf16 "
                f"(T4 is sm_75); loading unquantized {dtype_label}"
            )
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype,
            token=HF_TOKEN or True,
        )
        quant_label = f"unquantized {dtype_label}"
    else:
        dtype = torch.float16
        dtype_label = "float16"
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=dtype,
            token=HF_TOKEN or True,
        )
        quant_label = "nf4 4-bit bitsandbytes (double quant)"

    mdl.eval()
    device = next(mdl.parameters()).device
    n_layers = int(mdl.config.num_hidden_layers)
    print(f"[model] {model_id}  {quant_label}  layers={n_layers} device={device}")
    return tok, mdl, device, n_layers, dtype, quant_label


def unload(mdl):
    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def readout_layers(model, tokenizer, device, user_text: str, gold_content: str, family: str) -> dict:
    prompt = wrap_chat(tokenizer, user_text)
    tid, decoded, gold_ids, sep_note = resolve_target_token(tokenizer, prompt, gold_content)
    assert_content_gold(decoded, family)
    if DRY_RUN or model is None:
        n = 32
        if model is not None:
            n = int(getattr(model.config, "num_hidden_layers", 32))
        return {
            "gold_token_id": tid,
            "gold_token_decoded": decoded,
            "gold_token_ids": [int(x) for x in gold_ids],
            "sep_note": "DRY_RUN",
            "n_layers": n,
            "ranks": [1] * n,
            "logprobs": [0.0] * n,
            "cosines": [0.0] * n,
        }
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = out.hidden_states[1:]  # skip embedding; layer 0 = first block
    W_U = model.lm_head.weight.detach().float()  # [vocab, d]
    u = W_U[tid]
    u_norm = F.normalize(u.unsqueeze(0), dim=-1)
    ranks, logprobs, cosines = [], [], []
    for layer_h in hidden_states:
        h = layer_h[0, -1, :].float()
        logits = h @ W_U.T
        target_logit = logits[tid]
        rank = int((logits > target_logit).sum().item()) + 1
        lp = float(F.log_softmax(logits, dim=-1)[tid].item())
        cos = float(F.cosine_similarity(h.unsqueeze(0), u_norm, dim=-1).item())
        ranks.append(rank)
        logprobs.append(round(lp, 6))
        cosines.append(round(cos, 6))
    n_layers = len(ranks)
    del out, hidden_states, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "gold_token_id": int(tid),
        "gold_token_decoded": decoded,
        "gold_token_ids": [int(x) for x in gold_ids],
        "sep_note": sep_note,
        "n_layers": n_layers,
        "ranks": ranks,
        "logprobs": logprobs,
        "cosines": cosines,
    }
'''),
    md("""## Hugging Face login (gated Llama)

Run this **before** loading Llama. Uses `HF_TOKEN` from Colab secrets if set; otherwise prompts."""),
    code(r'''
from huggingface_hub import login, get_token

_tok = HF_TOKEN or get_token()
if _tok:
    login(token=_tok, add_to_git_credential=False)
    print("[hf] authenticated")
else:
    login()
'''),
    md("""## Run models (one at a time — T4 16GB)"""),
    code(r'''
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

MECH_CSV = OUT_DIR / "mech_freq_controlled_algo_bw.csv"
MECH_COLS = [
    "family", "model", "problem_id", "variant", "layer",
    "rank", "logprob", "cosine_to_gold_unembed",
    "gold_content", "gold_token_id", "gold_token_decoded",
    "gold_unigram_count", "gold_token_id_rank_proxy",
    "n_layers", "sep_note",
]

# Never write to the GSM file.
assert MECH_CSV.name != "mech_freq_controlled.csv"
print(f"[out] ALGO+BW -> {MECH_CSV}")
print(f"[out] GSM file left as {GSM_CSV_DO_NOT_TOUCH}")

done: set[tuple[str, str, str, str]] = set()
write_header = True
if RESUME and MECH_CSV.exists() and MECH_CSV.stat().st_size > 0:
    prev = pd.read_csv(MECH_CSV, dtype=str)
    if "family" not in prev.columns:
        prev.insert(0, "family", prev["problem_id"].map(
            lambda p: "ALGO" if str(p).split("_")[0] in {"CC", "SP", "WIS"} else "BW"
        ))
        prev.to_csv(MECH_CSV, index=False)
    done = set(
        zip(
            prev["model"].astype(str),
            prev["family"].astype(str),
            prev["problem_id"].astype(str),
            prev["variant"].astype(str),
        )
    )
    write_header = False
    print(f"[resume] {len(done)} completed (model, family, problem_id, variant) keys")
elif MECH_CSV.exists() and not RESUME:
    MECH_CSV.unlink()

FREQ_BY_MODEL: dict[str, Counter] = {}
N_LAYERS_BY_MODEL: dict[str, int] = {}
QUANT_BY_MODEL: dict[str, str] = {}
DTYPE_BY_MODEL: dict[str, str] = {}
DTYPE_USED = "float16"
QUANT = "mixed: Llama-3.1-8B + Qwen2.5-7B = nf4 4-bit; Qwen2.5-1.5B = unquantized (see quantization_by_model)"

for model_id in MODELS:
    pending = [
        it for it in ITEMS
        if (model_id, it["family"], it["problem_id"], it["variant"]) not in done
    ]
    print(f"\n===== {model_id}  pending={len(pending)}/{len(ITEMS)} =====")
    if DRY_RUN:
        class _Dummy:
            config = type("c", (), {"num_hidden_layers": 32})
        tok = AutoTokenizer.from_pretrained("gpt2")  # tiny, dry-run only
        mdl, device, n_layers = None, "cpu", 32
        # still want a real tokenizer for gold-token / unigram if possible
        try:
            tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN or True)
        except Exception as exc:
            print("[dry-run] tokenizer fallback gpt2:", exc)
        FREQ_BY_MODEL[model_id] = bank_unigram_counter(tok)
        N_LAYERS_BY_MODEL[model_id] = n_layers
        QUANT_BY_MODEL[model_id] = "DRY_RUN"
        DTYPE_BY_MODEL[model_id] = "n/a"
    else:
        tok, mdl, device, n_layers, dtype, quant_label = load_model(model_id)
        FREQ_BY_MODEL[model_id] = bank_unigram_counter(tok)
        N_LAYERS_BY_MODEL[model_id] = n_layers
        QUANT_BY_MODEL[model_id] = quant_label
        DTYPE_BY_MODEL[model_id] = str(dtype).replace("torch.", "")

    freq = FREQ_BY_MODEL[model_id]
    with MECH_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MECH_COLS)
        if write_header:
            w.writeheader()
            write_header = False
        for it in tqdm(pending, desc=model_id.split("/")[-1]):
            user = build_user(it["problem_text"], it["family"])
            metrics = readout_layers(mdl, tok, device, user, it["gold_content"], it["family"])
            tid = metrics["gold_token_id"]
            uni = int(freq.get(tid, 0))
            for layer_i in range(metrics["n_layers"]):
                w.writerow(
                    {
                        "family": it["family"],
                        "model": model_id,
                        "problem_id": it["problem_id"],
                        "variant": it["variant"],
                        "layer": layer_i,
                        "rank": metrics["ranks"][layer_i],
                        "logprob": metrics["logprobs"][layer_i],
                        "cosine_to_gold_unembed": metrics["cosines"][layer_i],
                        "gold_content": it["gold_content"],
                        "gold_token_id": tid,
                        "gold_token_decoded": metrics["gold_token_decoded"],
                        "gold_unigram_count": uni,
                        "gold_token_id_rank_proxy": tid,  # lower id ≈ more frequent in many BPEs
                        "n_layers": metrics["n_layers"],
                        "sep_note": metrics["sep_note"],
                    }
                )
            f.flush()
            done.add((model_id, it["family"], it["problem_id"], it["variant"]))

    if not DRY_RUN:
        unload(mdl)
    del tok
    gc.collect()

print(f"wrote {MECH_CSV}")

# Drive copy as soon as the sweep finishes (survives a later crash).
_drive_dir = Path("/content/drive/MyDrive/rvc_colab_out")
if Path("/content").exists():
    if not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive")
        except Exception as exc:
            print("[drive] mount skipped:", exc)
    try:
        _drive_dir.mkdir(parents=True, exist_ok=True)
        if MECH_CSV.exists():
            import shutil as _shutil
            _dst = _drive_dir / MECH_CSV.name
            _shutil.copy2(MECH_CSV, _dst)
            print(f"[backup] {MECH_CSV} -> {_dst}  ({_dst.stat().st_size} bytes)")
    except Exception as exc:
        print("[backup] skipped:", exc)
'''),
    md("""## Summary: canonical vs W3 final-layer rank, Wilcoxon, frequency terciles"""),
    code(r'''
df = pd.read_csv(MECH_CSV)
if "family" not in df.columns:
    df["family"] = df["problem_id"].map(
        lambda p: "GSM" if str(p).startswith("GSM") else (
            "ALGO" if str(p).split("_")[0] in {"CC", "SP", "WIS"} else "BW"
        )
    )
df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
df["gold_unigram_count"] = pd.to_numeric(df["gold_unigram_count"], errors="coerce")
df["layer"] = pd.to_numeric(df["layer"], errors="coerce")

# final layer per (model, family, problem, variant)
final = (
    df.sort_values("layer")
    .groupby(["model", "family", "problem_id", "variant"], as_index=False)
    .tail(1)
)

print("=== median final-layer rank ===")
print(
    final.groupby(["model", "family", "variant"])["rank"]
    .median()
    .unstack("variant")
    .to_string()
)

DEGEN_FRAC = 0.5


def degeneracy(toks: pd.Series) -> tuple[bool, str, int, int]:
    s = toks.astype(str).str.strip()
    n = int(len(s))
    if n == 0:
        return True, "", 0, 0
    vc = s.value_counts()
    modal = str(vc.index[0])
    n_m = int(vc.iloc[0])
    return (n_m / n) > DEGEN_FRAC, modal, n_m, n


print("\n=== gold-token degeneracy (Appendix H) ===")
degen_keys: set[tuple[str, str]] = set()
audit_rows = []
for (model_id, fam), g in final.groupby(["model", "family"]):
    uniq = g.drop_duplicates(["problem_id", "variant"])
    can = uniq[uniq.variant == "canonical"]
    dec_col = "gold_token_decoded"
    all_d, all_m, all_n, all_N = degeneracy(uniq[dec_col])
    can_d, can_m, can_n, can_N = degeneracy(can[dec_col])
    flag = all_d or can_d
    if flag:
        degen_keys.add((model_id, fam))
    why = []
    if all_d:
        why.append(f"all-items {all_m!r}={all_n}/{all_N}")
    if can_d:
        why.append(f"canonical {can_m!r}={can_n}/{can_N}")
    audit_rows.append(
        {
            "model": model_id,
            "family": fam,
            "degenerate": flag,
            "modal_all": all_m,
            "share_all": f"{all_n}/{all_N}",
            "modal_canonical": can_m,
            "share_canonical": f"{can_n}/{can_N}",
            "note": ("DEGENERATE: " + "; ".join(why) + " — Wilcoxon not reported")
            if flag
            else "ok",
        }
    )
audit = pd.DataFrame(audit_rows)
print(audit.to_string(index=False))

# terciles of gold unigram frequency (canonical gold, within model × family)
can_freq = final[final.variant == "canonical"][
    ["model", "family", "problem_id", "gold_unigram_count"]
].drop_duplicates()

def tercile_labels(s: pd.Series) -> pd.Series:
    # qcut can collapse if many ties (e.g. lots of '0'); fall back to rank-based
    try:
        return pd.qcut(s, 3, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError:
        return pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"])

parts = []
for (model_id, fam), g in can_freq.groupby(["model", "family"]):
    t = g.copy()
    t["freq_tercile"] = tercile_labels(t["gold_unigram_count"])
    parts.append(t)
terc = pd.concat(parts, ignore_index=True) if parts else can_freq.assign(freq_tercile="mid")
paired = final.merge(
    terc[["model", "family", "problem_id", "freq_tercile"]],
    on=["model", "family", "problem_id"],
    how="left",
)

wide = paired.pivot_table(
    index=["model", "family", "problem_id", "freq_tercile"],
    columns="variant",
    values="rank",
    aggfunc="first",
).reset_index()
wide = wide.dropna(subset=["canonical", "W3"])


def wilcoxon_block(
    sub: pd.DataFrame,
    model_id: str,
    fam: str,
    tercile: str,
    *,
    degenerate: bool = False,
    degen_note: str = "",
) -> dict:
    label = f"{model_id} | {fam} | {tercile}"
    a = sub["canonical"].to_numpy(dtype=float)
    b = sub["W3"].to_numpy(dtype=float)
    base = {
        "slice": label,
        "model": model_id,
        "family": fam,
        "freq_tercile": tercile,
        "n_pairs": int(len(sub)),
    }
    if degenerate:
        return {
            **base,
            "median_rank_canonical": float("nan"),
            "median_rank_W3": float("nan"),
            "median_can_minus_W3": float("nan"),
            "W": float("nan"),
            "p_two_sided": float("nan"),
            "note": degen_note or "DEGENERATE — result not reported",
        }
    if len(a) < 3 or np.allclose(a, b):
        stat, p = float("nan"), float("nan")
        note = "n<3 or identical"
    else:
        try:
            stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            note = "ok"
        except ValueError as exc:
            stat, p, note = float("nan"), float("nan"), str(exc)
    return {
        **base,
        "median_rank_canonical": float(np.median(a)),
        "median_rank_W3": float(np.median(b)),
        "median_can_minus_W3": float(np.median(a - b)),
        "W": stat,
        "p_two_sided": p,
        "note": note,
    }


rows = []
for (model_id, fam), g in wide.groupby(["model", "family"]):
    flag = (model_id, fam) in degen_keys
    degen_note = ""
    if flag:
        hit = audit[(audit.model == model_id) & (audit.family == fam)]
        degen_note = str(hit.iloc[0]["note"]) if len(hit) else "DEGENERATE — result not reported"
        print(f"\n[skip] {model_id} / {fam}: {degen_note}")
    rows.append(
        wilcoxon_block(g, model_id, fam, "all", degenerate=flag, degen_note=degen_note)
    )
    for tname, sg in g.groupby("freq_tercile", observed=False):
        rows.append(
            wilcoxon_block(
                sg, model_id, fam, f"tercile={tname}",
                degenerate=flag, degen_note=degen_note,
            )
        )

wtab = pd.DataFrame(rows)
print("\n=== paired Wilcoxon (final-layer rank, canonical vs W3) ===")
print(wtab.to_string(index=False))
wtab.to_csv(OUT_DIR / "mech_freq_controlled_algo_bw_summary.csv", index=False)

print("\n=== gold token audit (must not be format keywords) ===")
dec = df.drop_duplicates(["model", "family", "problem_id", "variant"])[
    ["model", "family", "gold_token_decoded", "gold_content", "gold_unigram_count"]
]
for fam, sg in dec.groupby("family"):
    print(f"\n-- {fam} --")
    print(sg["gold_token_decoded"].value_counts().head(10).to_string())
bad = dec[dec["gold_token_decoded"].str.strip().str.lower().isin(FORMAT_KEYWORDS)]
print(f"\nformat-keyword golds: {len(bad)} (expect 0)")
assert len(bad) == 0, bad.head()
'''),
    md("""## Manifest"""),
    code(r'''
import subprocess

def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:
        return f"unavailable ({exc})"

n_items = int(df.drop_duplicates(["model", "family", "problem_id", "variant"]).shape[0] / max(len(MODELS), 1)) if "df" in dir() else len(ITEMS)

manifest = {
    "notebook": "mechanistic_frequency_controlled.ipynb",
    "model_string": MODELS,
    "dtype": DTYPE_BY_MODEL if "DTYPE_BY_MODEL" in dir() else DTYPE_USED,
    "quantization": QUANT_BY_MODEL if "QUANT_BY_MODEL" in dir() else (QUANT if not DRY_RUN else "DRY_RUN"),
    "quantization_confound": (
        "Qwen2.5-1.5B-Instruct is unquantized (bf16 if supported, else fp16 on T4). "
        "Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct are nf4 4-bit. "
        "Do not treat rank differences as a pure size effect."
    ),
    "decoding_config": {
        "readout": "forward pass only (no generation); last prompt position",
        "chat_template": True,
        "prompt": "Appendix N Probe-1 family-specific format instruction",
        "gold": {
            "ALGO": "SP Cost / CC Count|Total / WIS Total (not Path/Coins/Selected)",
            "BW": "first action word of gold plan",
        },
    },
    "n_items": n_items,
    "n_ids": {
        "ALGO": len({x["problem_id"] for x in ITEMS if x["family"] == "ALGO"}),
        "BW": len({x["problem_id"] for x in ITEMS if x["family"] == "BW"}),
    },
    "n_models": len(MODELS),
    "id_source": {
        "ALGO": "frozen adversarial pool 34 SP + 10 CC + 17 WIS = 61",
        "BW": "data/problems/question_bank_bw.csv canonical IDs (n=65)",
        "GSM": "NOT QUEUED — existing mech_freq_controlled.csv left untouched",
    },
    "frequency_proxy": "GSM+ALGO+BW bank unigram counts under each model's tokenizer; token_id as secondary rank proxy",
    "degeneracy_rule": "family flagged if modal gold token is >50% of all items or of canonical items; Wilcoxon not reported",
    "git_commit_hash": git_hash(),
    "output_csv": str(MECH_CSV),
    "n_layers_by_model": {k: int(v) for k, v in N_LAYERS_BY_MODEL.items()} if "N_LAYERS_BY_MODEL" in dir() else {},
}
print("=== MANIFEST ===")
print(json.dumps(manifest, indent=2))
(OUT_DIR / "mech_freq_controlled_algo_bw_manifest.json").write_text(json.dumps(manifest, indent=2))

# Drive + laptop download for the three ALGO/BW artifacts. Does not touch GSM CSV.
_out_files = [
    OUT_DIR / "mech_freq_controlled_algo_bw.csv",
    OUT_DIR / "mech_freq_controlled_algo_bw_summary.csv",
    OUT_DIR / "mech_freq_controlled_algo_bw_manifest.json",
]
_drive_dir = Path("/content/drive/MyDrive/rvc_colab_out")
if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
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


# ===========================================================================
# NOTEBOOK 3 — O5 teacher-forced gold-sequence likelihood
# ===========================================================================

NB3 = [
    md("""# O5 — Teacher-forced gold-sequence likelihood (Probe-1 grid)

**Floor-free robustness measure.** Binary retention is undefined at floor and ceiling; the 0.30 accuracy floor suppresses most retention cells; N3 collapsed when W3 accuracy was 1/60 and 0/61. Log-likelihood of the gold answer is continuous, defined everywhere, and deterministic (no sampling noise).

**Models (HuggingFace, Colab T4):**
| Model | Load | Role |
|-------|------|------|
| `Qwen/Qwen2.5-1.5B-Instruct` | fp16, `attn_implementation="sdpa"` | primary |
| `Qwen/Qwen2.5-3B-Instruct` | fp16, `attn_implementation="sdpa"` | primary |
| `meta-llama/Llama-3.1-8B-Instruct` | 4-bit NF4, `compute_dtype=float16`, sdpa | robustness check only |

**T4 hard constraints:** fp16 only (no bf16), no FlashAttention-2, `attn_implementation="sdpa"`.

**Grid:** every `(family ∈ {GSM, ALGO, BW}, problem_id, variant ∈ {canonical, W1…W6}, model)` present in the question banks.

**Prompt:** identical Appendix-N Probe-1 template used by the other Colab Probe-1 notebooks (`PROBE1_TEMPLATE` + `FAMILY_FORMAT` + chat template). Do not rewrite.

**Output:** `colab_out/O5_teacher_forced_likelihood.csv` — **per-item rows only**. No aggregates here (analysis is O10).

**Secrets:** `HF_TOKEN` (gated Llama); optional `GITHUB_TOKEN` if the repo is private.

---

### CRITICAL CAVEAT — gold TARGET STRING can change under W3 / W6

W3 renames entities and W6 regenerates parameters, so the gold **target string** often differs from canonical. A raw Δ mean_logprob then confounds *"the model's belief moved"* with *"we are scoring a different string."*

This notebook handles that as follows:
1. **Primary metric is always `mean_logprob`** (length-normalized). Persist `sum_logprob` but do not treat it as the comparison unit.
2. **`target_identical`:** True iff normalized variant gold == normalized canonical gold (cleanest Δ cells; report separately in O10). Checked for all variants; especially informative for W1/W2/W4/W5.
3. **`target_comparable` + `control_*`:** when the variant gold is well-formed under the **canonical** prompt, also teacher-force that gold under the canonical prompt and persist `control_mean_logprob` (etc.). When impossible, `target_comparable=False` and control fields are empty.
4. Well-formed rule (documented, deterministic): identical golds → comparable; else GSM numeric golds remain format-valid under any GSM prompt → comparable; ALGO/BW with a **different** gold → not comparable (entity/operator/instance mismatch)."""),
    code(SETUP_PIP),
    code(SETUP_REPO),
    md("""## Item queue — full Probe-1 bank grid + clone families

Loads `data/problems/question_bank_{gsm,algo,bw}.csv`. Every bank row is one cell. Clone IDs come from `probes.common.clones` / `bank_clone_audit.csv` (ALGO); GSM/BW use `SINGLETON_{problem_id}`."""),
    code(r'''
from __future__ import annotations

import csv
import gc
import re
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from probes.common.clones import algo_cluster_map

# Same Appendix-N template as llama_greedy_behavioural / mechanistic notebooks.
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

VARIANTS = ("canonical", "W1", "W2", "W3", "W4", "W5", "W6")

MODELS: list[tuple[str, str]] = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "fp16"),
    ("Qwen/Qwen2.5-3B-Instruct", "fp16"),
    ("meta-llama/Llama-3.1-8B-Instruct", "nf4"),  # robustness check only
]

O5_CSV = OUT_DIR / "O5_teacher_forced_likelihood.csv"

OUT_COLUMNS = [
    "family",
    "problem_id",
    "variant",
    "model",
    "n_gold_tokens",
    "sum_logprob",
    "mean_logprob",
    "gold_first_token_rank",
    "gold_first_token_logprob",
    "prompt_n_tokens",
    "clone_family",
    "target_identical",
    "target_comparable",
    "control_n_gold_tokens",
    "control_sum_logprob",
    "control_mean_logprob",
    "control_gold_first_token_rank",
    "control_gold_first_token_logprob",
]


def _norm_vt(v: str) -> str:
    v = str(v).strip()
    return "canonical" if v.lower() == "canonical" else v.upper()


def _strip_csv_quotes(text: str) -> str:
    s = str(text)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s


def norm_gold(text: str) -> str:
    """Whitespace-normalized gold for identity checks (not for tokenization)."""
    lines = [ln.strip() for ln in str(text).replace("\r\n", "\n").split("\n")]
    return "\n".join(ln for ln in lines if ln)


def looks_numeric_gold(text: str) -> bool:
    s = norm_gold(text)
    s = re.sub(r"^####\s*", "", s).replace(",", "").strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", s))


def build_prompt(problem_text: str, family: str) -> str:
    """Identical Probe-1 user string construction as the behavioural Colab notebooks."""
    return PROBE1_TEMPLATE.format(
        problem=problem_text.strip(),
        family_specific_output_format=FAMILY_FORMAT[family],
    )


def _load_bank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["problem_id"] = df["problem_id"].astype(str).str.strip()
    df["variant_type"] = df["variant_type"].map(_norm_vt)
    df["problem_text"] = df["problem_text"].map(_strip_csv_quotes)
    df["correct_answer"] = df["correct_answer"].map(_strip_csv_quotes)
    return df


def clone_family_for(family: str, problem_id: str, cmap: dict[str, str]) -> str:
    if family == "ALGO":
        return cmap.get(problem_id, f"SINGLETON_{problem_id}")
    return f"SINGLETON_{problem_id}"


def target_flags(
    family: str,
    variant: str,
    can_gold: str,
    var_gold: str,
) -> tuple[bool, bool]:
    """Return (target_identical, target_comparable).

    Comparable ⇒ we may teacher-force *variant gold* under the *canonical* prompt.
    """
    identical = norm_gold(can_gold) == norm_gold(var_gold)
    if identical:
        return True, True
    # Different string: only GSM keeps a format-valid numeric answer under can-prompt.
    if family == "GSM" and looks_numeric_gold(var_gold):
        return False, True
    # W3/W6 (and non-identical ALGO/BW W*) change entities/operators/instance.
    _ = variant  # retained for callers / future tightening
    return False, False


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
        can = {
            str(r.problem_id): str(r.correct_answer)
            for r in df.loc[df.variant_type == "canonical"].itertuples(index=False)
        }
        can_text = {
            str(r.problem_id): str(r.problem_text)
            for r in df.loc[df.variant_type == "canonical"].itertuples(index=False)
        }
        for _, row in df.iterrows():
            pid = str(row["problem_id"])
            vt = str(row["variant_type"])
            if vt not in VARIANTS:
                continue
            if pid not in can:
                continue
            var_gold = str(row["correct_answer"])
            c_gold = can[pid]
            identical, comparable = target_flags(family, vt, c_gold, var_gold)
            items.append(
                {
                    "family": family,
                    "problem_id": pid,
                    "variant": vt,
                    "problem_text": str(row["problem_text"]),
                    "gold": var_gold,
                    "canonical_problem_text": can_text[pid],
                    "canonical_gold": c_gold,
                    "target_identical": identical,
                    "target_comparable": comparable and vt != "canonical",
                    "clone_family": clone_family_for(family, pid, cmap),
                }
            )
    if limit is not None:
        # Keep a balanced smoke slice: first `limit` IDs per family × all their variants.
        keep: set[tuple[str, str]] = set()
        for fam in ("GSM", "ALGO", "BW"):
            ids = sorted({x["problem_id"] for x in items if x["family"] == fam})[:limit]
            keep |= {(fam, pid) for pid in ids}
        items = [x for x in items if (x["family"], x["problem_id"]) in keep]
    return items


ITEMS = load_items(LIMIT)
print(f"[queue] {len(ITEMS)} cells (LIMIT={LIMIT})")
print(pd.DataFrame(ITEMS).groupby(["family", "variant"]).size().unstack(fill_value=0).to_string())
print(
    "[caveat] target_identical rates by variant:\n",
    pd.DataFrame(ITEMS)
    .groupby("variant")["target_identical"]
    .mean()
    .reindex(list(VARIANTS))
    .round(3)
    .to_string(),
)
print(
    "[caveat] target_comparable rates by variant:\n",
    pd.DataFrame(ITEMS)
    .groupby("variant")["target_comparable"]
    .mean()
    .reindex(list(VARIANTS))
    .round(3)
    .to_string(),
)
'''),
    md("""## Teacher-forced likelihood

For each cell: chat-wrap the Probe-1 user prompt, append the bank gold string (verbatim `correct_answer`), run one forward pass, and sum token log-probs of the gold continuation.

Primary fields: `sum_logprob`, `mean_logprob = sum / n_gold_tokens`, `gold_first_token_rank`, `gold_first_token_logprob`.

When `target_comparable`, also score **variant gold under the canonical prompt** → `control_*` columns."""),
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
    elif quant == "nf4":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            **common,
        )
        label = "nf4 4-bit bitsandbytes (compute_dtype=float16) + sdpa"
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


def empty_control() -> dict[str, Any]:
    return {
        "control_n_gold_tokens": "",
        "control_sum_logprob": "",
        "control_mean_logprob": "",
        "control_gold_first_token_rank": "",
        "control_gold_first_token_logprob": "",
    }


def score_item(model, tokenizer, device, item: dict, model_id: str) -> dict[str, Any]:
    user = build_prompt(item["problem_text"], item["family"])
    primary = teacher_forced_metrics(model, tokenizer, device, user, item["gold"])
    row: dict[str, Any] = {
        "family": item["family"],
        "problem_id": item["problem_id"],
        "variant": item["variant"],
        "model": model_id,
        "n_gold_tokens": primary["n_gold_tokens"],
        "sum_logprob": primary["sum_logprob"],
        "mean_logprob": primary["mean_logprob"],
        "gold_first_token_rank": primary["gold_first_token_rank"],
        "gold_first_token_logprob": primary["gold_first_token_logprob"],
        "prompt_n_tokens": primary["prompt_n_tokens"],
        "clone_family": item["clone_family"],
        "target_identical": bool(item["target_identical"]),
        "target_comparable": bool(item["target_comparable"]),
    }
    row.update(empty_control())
    if item["target_comparable"]:
        can_user = build_prompt(item["canonical_problem_text"], item["family"])
        # Control: W-variant gold under the canonical prompt (same gold string, different surface).
        ctrl = teacher_forced_metrics(model, tokenizer, device, can_user, item["gold"])
        row["control_n_gold_tokens"] = ctrl["n_gold_tokens"]
        row["control_sum_logprob"] = ctrl["sum_logprob"]
        row["control_mean_logprob"] = ctrl["mean_logprob"]
        row["control_gold_first_token_rank"] = ctrl["gold_first_token_rank"]
        row["control_gold_first_token_logprob"] = ctrl["gold_first_token_logprob"]
    return row
'''),
    md("""## Run — write every per-item row (resume-safe)

No aggregate statistics. Resume key: `(model, family, problem_id, variant)`."""),
    code(r'''
def _done_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    need = {"model", "family", "problem_id", "variant"}
    if not need.issubset(df.columns):
        return set()
    return {
        (str(r.model), str(r.family), str(r.problem_id), str(r.variant))
        for r in df.itertuples(index=False)
    }


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in OUT_COLUMNS})


done = _done_keys(O5_CSV) if RESUME else set()
print(f"[resume] {len(done)} rows already in {O5_CSV}")

for model_id, quant in MODELS:
    pending = [
        it
        for it in ITEMS
        if (model_id, it["family"], it["problem_id"], it["variant"]) not in done
    ]
    print(f"\n=== {model_id} ({quant})  pending={len(pending)}/{len(ITEMS)} ===")
    if not pending:
        continue
    tok, mdl, device = load_model(model_id, quant)
    buf: list[dict[str, Any]] = []
    try:
        for it in tqdm(pending, desc=model_id.split("/")[-1]):
            row = score_item(mdl, tok, device, it, model_id)
            buf.append(row)
            if len(buf) >= 25:
                append_rows(O5_CSV, buf)
                buf.clear()
        append_rows(O5_CSV, buf)
    finally:
        unload(mdl)

print(f"\n[done] wrote {O5_CSV}")
if O5_CSV.exists():
    out_df = pd.read_csv(O5_CSV)
    print(f"[done] n_rows={len(out_df)}  models={sorted(out_df['model'].unique())}")
    print(out_df.groupby(["model", "family"]).size().unstack(fill_value=0).to_string())
    # Intentionally no mean/retention aggregates — O10 owns analysis.
'''),
    md("""## Download / Drive backup

Copy `O5_teacher_forced_likelihood.csv` into the repo as `results/raw/O5_teacher_forced_likelihood.csv` after the Colab run."""),
    code(r'''
_out_files = [O5_CSV]
_drive_dir = Path("/content/drive/MyDrive/rvc_colab_out")
if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
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


# ===========================================================================
# NOTEBOOK 4 — O6 quantization sensitivity bound
# ===========================================================================

NB4 = [
    md("""# O6 — Quantization sensitivity bound (O5 measurement)

Quantization perturbs logits. O5's `gold_first_token_rank` is a rank over the full vocabulary, so **4-bit ranks may not be comparable to fp16**. This notebook bounds the error on Qwen2.5-1.5B-Instruct.

**Design**
1. Draw a **fixed-seed stratified subsample of 60** Probe-1 cells (20 GSM + 20 ALGO + 20 BW) from the same O5 item universe.
2. Run the **identical O5 teacher-forced gold-sequence measurement** at three precisions on `Qwen/Qwen2.5-1.5B-Instruct`:
   - **fp16** (reference)
   - **8-bit** (bitsandbytes)
   - **4-bit NF4** (`compute_dtype=float16`)
3. For each precision pair, report: mean |Δ mean_logprob|, Spearman(mean_logprob), median |Δ gold_first_token_rank|, 95th percentile |Δ rank|.

**T4:** fp16 only for unquantized loads; `attn_implementation="sdpa"`; no FlashAttention-2; no bf16.

**Decision rule (methods):** if the **median absolute rank shift** for fp16↔4-bit exceeds roughly **50** positions, drop all 4-bit rank measurements from the paper and keep only fp16 (state this explicitly).

**Outputs**
- `colab_out/O6_quantization_sensitivity.csv` — pairwise summary (required)
- `colab_out/O6_quantization_sensitivity_items.csv` — every per-item score (audit; do not lose intermediates)
- `colab_out/O6_quantization_sensitivity_summary.txt` — one-paragraph usability verdict

`LIMIT` (setup cell) shrinks the per-family draw for smoke tests (e.g. `LIMIT=2` → 6 items)."""),
    code(SETUP_PIP),
    code(SETUP_REPO),
    md("""## Stratified 60-item subsample + O5 measurement primitives

Same Appendix-N prompt and teacher-forced gold continuation as O5. Subsample seed is fixed (`SUBSAMPLE_SEED=42`)."""),
    code(r'''
from __future__ import annotations

import csv
import gc
import json
import random
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PRECISIONS = ("fp16", "int8", "nf4")
SUBSAMPLE_SEED = 42
N_PER_FAMILY = 20  # 20 × 3 = 60; overridden downward when LIMIT is set
RANK_DROP_THRESHOLD = 50  # median |Δ rank| fp16↔nf4; methods kill criterion

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
VARIANTS = ("canonical", "W1", "W2", "W3", "W4", "W5", "W6")

O6_ITEMS_CSV = OUT_DIR / "O6_quantization_sensitivity_items.csv"
O6_CSV = OUT_DIR / "O6_quantization_sensitivity.csv"
O6_SUMMARY_TXT = OUT_DIR / "O6_quantization_sensitivity_summary.txt"

ITEM_COLUMNS = [
    "family", "problem_id", "variant", "precision", "model",
    "n_gold_tokens", "sum_logprob", "mean_logprob",
    "gold_first_token_rank", "gold_first_token_logprob", "prompt_n_tokens",
]
PAIR_COLUMNS = [
    "precision_a", "precision_b", "n_items",
    "mean_abs_diff_mean_logprob", "spearman_mean_logprob", "spearman_pvalue",
    "median_abs_rank_shift", "p95_abs_rank_shift",
    "max_abs_rank_shift",
]


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


def _load_bank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["problem_id"] = df["problem_id"].astype(str).str.strip()
    df["variant_type"] = df["variant_type"].map(_norm_vt)
    df["problem_text"] = df["problem_text"].map(_strip_csv_quotes)
    df["correct_answer"] = df["correct_answer"].map(_strip_csv_quotes)
    return df


def load_o5_universe() -> list[dict[str, Any]]:
    """Full O5 cell universe (family × problem × variant present in banks)."""
    specs = [
        ("GSM", REPO_ROOT / "data/problems/question_bank_gsm.csv"),
        ("ALGO", REPO_ROOT / "data/problems/question_bank_algo.csv"),
        ("BW", REPO_ROOT / "data/problems/question_bank_bw.csv"),
    ]
    items: list[dict[str, Any]] = []
    for family, path in specs:
        df = _load_bank(path)
        can_ids = set(df.loc[df.variant_type == "canonical", "problem_id"])
        for _, row in df.iterrows():
            pid = str(row["problem_id"])
            vt = str(row["variant_type"])
            if vt not in VARIANTS or pid not in can_ids:
                continue
            items.append(
                {
                    "family": family,
                    "problem_id": pid,
                    "variant": vt,
                    "problem_text": str(row["problem_text"]),
                    "gold": str(row["correct_answer"]),
                }
            )
    return items


def stratified_subsample(
    universe: list[dict[str, Any]],
    n_per_family: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for fam in ("GSM", "ALGO", "BW"):
        pool = [x for x in universe if x["family"] == fam]
        if not pool:
            raise RuntimeError(f"empty pool for {fam}")
        k = min(n_per_family, len(pool))
        out.extend(rng.sample(pool, k=k))
    out.sort(key=lambda x: (x["family"], x["problem_id"], x["variant"]))
    return out


_n_per = N_PER_FAMILY if LIMIT is None else min(int(LIMIT), N_PER_FAMILY)
UNIVERSE = load_o5_universe()
ITEMS = stratified_subsample(UNIVERSE, _n_per, SUBSAMPLE_SEED)
print(
    f"[sample] n={len(ITEMS)}  seed={SUBSAMPLE_SEED}  "
    f"n_per_family={_n_per}  universe={len(UNIVERSE)}"
)
print(pd.DataFrame(ITEMS).groupby(["family", "variant"]).size().unstack(fill_value=0).to_string())
(OUT_DIR / "O6_subsample_manifest.json").write_text(
    json.dumps(
        {
            "model": MODEL_ID,
            "seed": SUBSAMPLE_SEED,
            "n_per_family": _n_per,
            "n_items": len(ITEMS),
            "items": [
                {"family": x["family"], "problem_id": x["problem_id"], "variant": x["variant"]}
                for x in ITEMS
            ],
        },
        indent=2,
    )
)
print(f"[sample] wrote {OUT_DIR / 'O6_subsample_manifest.json'}")
'''),
    md("""## Teacher-forced metrics (identical to O5) + precision loaders"""),
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
        return prompt_ids, enc(answer), "FALLBACK"
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
        # Deterministic fake ranks keyed by precision label length — only for pipeline checks.
        return {
            "n_gold_tokens": n_gold,
            "sum_logprob": -0.1 * n_gold,
            "mean_logprob": -0.1,
            "gold_first_token_rank": 1,
            "gold_first_token_logprob": -0.1,
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


def load_model(precision: str):
    assert torch.cuda.is_available() or DRY_RUN, "GPU required (Colab T4) unless DRY_RUN."
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN or True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if DRY_RUN:
        print(f"[model] DRY_RUN skip load: {MODEL_ID} ({precision})")
        return tok, None, torch.device("cpu")

    common = dict(
        device_map="auto",
        token=HF_TOKEN or True,
        attn_implementation="sdpa",
    )
    if precision == "fp16":
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            **common,
        )
        label = "fp16 unquantized + sdpa"
    elif precision == "int8":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb,
            torch_dtype=torch.float16,
            **common,
        )
        label = "int8 bitsandbytes + sdpa"
    elif precision == "nf4":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb,
            torch_dtype=torch.float16,
            **common,
        )
        label = "nf4 4-bit (compute_dtype=float16) + sdpa"
    else:
        raise ValueError(precision)
    mdl.eval()
    device = next(mdl.parameters()).device
    print(f"[model] {MODEL_ID}  {label}  device={device}")
    return tok, mdl, device


def unload(mdl):
    if mdl is None:
        return
    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_primary(model, tokenizer, device, item: dict, precision: str) -> dict[str, Any]:
    user = build_prompt(item["problem_text"], item["family"])
    m = teacher_forced_metrics(model, tokenizer, device, user, item["gold"])
    # DRY_RUN: inject precision-dependent rank noise so pairwise shifts are nonzero in pipeline checks
    if DRY_RUN or model is None:
        bump = {"fp16": 0, "int8": 3, "nf4": 80}[precision]
        m = dict(m)
        m["gold_first_token_rank"] = 1 + bump
        m["mean_logprob"] = round(-0.1 - 0.01 * bump, 6)
        m["sum_logprob"] = round(m["mean_logprob"] * m["n_gold_tokens"], 6)
    return {
        "family": item["family"],
        "problem_id": item["problem_id"],
        "variant": item["variant"],
        "precision": precision,
        "model": MODEL_ID,
        "n_gold_tokens": m["n_gold_tokens"],
        "sum_logprob": m["sum_logprob"],
        "mean_logprob": m["mean_logprob"],
        "gold_first_token_rank": m["gold_first_token_rank"],
        "gold_first_token_logprob": m["gold_first_token_logprob"],
        "prompt_n_tokens": m["prompt_n_tokens"],
    }
'''),
    md("""## Run all three precisions → pairwise sensitivity + verdict

Decision: if median |Δ rank| for **fp16 vs nf4** > ~50, 4-bit ranks are **not usable** in the paper."""),
    code(r'''
def _done_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    need = {"precision", "family", "problem_id", "variant"}
    if not need.issubset(df.columns):
        return set()
    return {
        (str(r.precision), str(r.family), str(r.problem_id), str(r.variant))
        for r in df.itertuples(index=False)
    }


def append_item_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ITEM_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ITEM_COLUMNS})


done = _done_keys(O6_ITEMS_CSV) if RESUME else set()
print(f"[resume] {len(done)} item-rows in {O6_ITEMS_CSV}")

for precision in PRECISIONS:
    pending = [
        it
        for it in ITEMS
        if (precision, it["family"], it["problem_id"], it["variant"]) not in done
    ]
    print(f"\n=== {MODEL_ID} @ {precision}  pending={len(pending)}/{len(ITEMS)} ===")
    if not pending:
        continue
    tok, mdl, device = load_model(precision)
    buf: list[dict[str, Any]] = []
    try:
        for it in tqdm(pending, desc=f"{precision}"):
            buf.append(score_primary(mdl, tok, device, it, precision))
            if len(buf) >= 20:
                append_item_rows(O6_ITEMS_CSV, buf)
                buf.clear()
        append_item_rows(O6_ITEMS_CSV, buf)
    finally:
        unload(mdl)

items_df = pd.read_csv(O6_ITEMS_CSV)
print(f"[items] n={len(items_df)}  precisions={sorted(items_df.precision.unique())}")


def pairwise_row(df: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
    wide = (
        df[df.precision.isin([a, b])]
        .pivot_table(
            index=["family", "problem_id", "variant"],
            columns="precision",
            values=["mean_logprob", "gold_first_token_rank"],
            aggfunc="first",
        )
        .dropna()
    )
    # Flatten MultiIndex columns
    lp_a = wide[("mean_logprob", a)].astype(float)
    lp_b = wide[("mean_logprob", b)].astype(float)
    rk_a = wide[("gold_first_token_rank", a)].astype(float)
    rk_b = wide[("gold_first_token_rank", b)].astype(float)
    abs_lp = (lp_a - lp_b).abs()
    abs_rk = (rk_a - rk_b).abs()
    if len(lp_a) >= 2 and lp_a.nunique() > 1 and lp_b.nunique() > 1:
        spearman_r, spearman_p = stats.spearmanr(lp_a, lp_b)
    elif len(lp_a) >= 2:
        spearman_r, spearman_p = float("nan"), float("nan")
    else:
        spearman_r, spearman_p = float("nan"), float("nan")
    return {
        "precision_a": a,
        "precision_b": b,
        "n_items": int(len(wide)),
        "mean_abs_diff_mean_logprob": round(float(abs_lp.mean()), 6),
        "spearman_mean_logprob": (
            round(float(spearman_r), 6) if spearman_r == spearman_r else ""
        ),
        "spearman_pvalue": (
            round(float(spearman_p), 6) if spearman_p == spearman_p else ""
        ),
        "median_abs_rank_shift": round(float(abs_rk.median()), 3),
        "p95_abs_rank_shift": round(float(np.percentile(abs_rk, 95)), 3),
        "max_abs_rank_shift": round(float(abs_rk.max()), 3),
    }


pairs = [("fp16", "int8"), ("fp16", "nf4"), ("int8", "nf4")]
pair_rows = [pairwise_row(items_df, a, b) for a, b in pairs]
pair_df = pd.DataFrame(pair_rows, columns=PAIR_COLUMNS)
pair_df.to_csv(O6_CSV, index=False)
print("\n=== O6_quantization_sensitivity.csv ===")
print(pair_df.to_string(index=False))

fp16_nf4 = next(r for r in pair_rows if r["precision_a"] == "fp16" and r["precision_b"] == "nf4")
med = float(fp16_nf4["median_abs_rank_shift"])
p95 = float(fp16_nf4["p95_abs_rank_shift"])
mad_lp = float(fp16_nf4["mean_abs_diff_mean_logprob"])
sp = fp16_nf4["spearman_mean_logprob"]
n = int(fp16_nf4["n_items"])
drop_4bit_ranks = med > RANK_DROP_THRESHOLD

if drop_4bit_ranks:
    verdict = (
        f"On a stratified subsample of {n} Probe-1 cells (seed={SUBSAMPLE_SEED}, "
        f"Qwen2.5-1.5B-Instruct), fp16 vs 4-bit NF4 showed mean |Δ mean_logprob|={mad_lp}, "
        f"Spearman(mean_logprob)={sp}, median |Δ gold_first_token_rank|={med}, "
        f"and 95th-percentile rank shift={p95}. Because the median absolute rank shift "
        f"exceeds ~{RANK_DROP_THRESHOLD} vocabulary positions, 4-bit rank measurements are "
        f"NOT usable for paper claims: drop all 4-bit gold_first_token_rank results and "
        f"retain fp16 (and, where needed, 8-bit) ranks only; state this explicitly in Methods. "
        f"Pairwise table: O6_quantization_sensitivity.csv."
    )
else:
    verdict = (
        f"On a stratified subsample of {n} Probe-1 cells (seed={SUBSAMPLE_SEED}, "
        f"Qwen2.5-1.5B-Instruct), fp16 vs 4-bit NF4 showed mean |Δ mean_logprob|={mad_lp}, "
        f"Spearman(mean_logprob)={sp}, median |Δ gold_first_token_rank|={med}, "
        f"and 95th-percentile rank shift={p95}. The median absolute rank shift is ≤ "
        f"~{RANK_DROP_THRESHOLD} positions, so 4-bit rank measurements are usable as a "
        f"robustness check alongside fp16 primary results, with the tabulated sensitivity "
        f"bounds reported in Methods. Pairwise table: O6_quantization_sensitivity.csv."
    )

O6_SUMMARY_TXT.write_text(verdict + "\n")
print("\n=== SUMMARY (also written to O6_quantization_sensitivity_summary.txt) ===")
print(verdict)
print(f"\n[decision] drop_4bit_ranks={drop_4bit_ranks}  threshold={RANK_DROP_THRESHOLD}")
'''),
    md("""## Download / Drive backup

Copy into the repo after Colab:
- `O6_quantization_sensitivity.csv` → `results/derived/O6_quantization_sensitivity.csv`
- `O6_quantization_sensitivity_items.csv` → `results/raw/O6_quantization_sensitivity_items.csv`
- `O6_quantization_sensitivity_summary.txt` → `results/derived/O6_quantization_sensitivity_summary.txt`"""),
    code(r'''
_out_files = [
    O6_CSV,
    O6_ITEMS_CSV,
    O6_SUMMARY_TXT,
    OUT_DIR / "O6_subsample_manifest.json",
]
_drive_dir = Path("/content/drive/MyDrive/rvc_colab_out")
if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
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


# ===========================================================================
# NOTEBOOK 5 — O7 GSM gold-token degeneracy check
# ===========================================================================

NB5 = [
    md("""# O7 — GSM gold-token degeneracy screen (Probe-3 mechanistic gate)

Probe-3 mechanistic was **excluded for Blocksworld** because the gold token was degenerate (only ~3 distinct gold tokens across the bank). **ALGO passed** (modal gold token 21.3%, under the 50% cutoff). **GSM was never screened.**

This notebook runs the **same degeneracy check** on GSM:

1. Extract the gold final-answer **content** for every GSM bank row (canonical + W1–W6) via the same `gsm_gold_content` helper as the mechanistic Colab (not `####` scaffolding).
2. Tokenize with each model tokenizer: `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`.
3. Resolve the **prompt-aware first BPE** of the gold content after the Appendix-N Probe-1 chat prompt (identical `resolve_target_token` logic).
4. Report: `# distinct` gold first tokens, modal token + share, Shannon entropy (bits).
5. **Cutoff (same as ALGO/BW):** FAIL if modal share **> 50%** on all items **or** on canonical-only.

**PASS** → GSM becomes a second mechanistic family; O8 runs on ALGO **and** GSM.  
**FAIL** → report in the measurement-failure table alongside BW.

**Note:** tokenizer-only — GPU not required (T4 ok; no model weights loaded).

**Outputs:** `O7_gsm_degeneracy_check.csv`, `O7_gsm_degeneracy_items.csv` (per-item audit), printed PASS/FAIL."""),
    code(SETUP_PIP),
    code(SETUP_REPO),
    md("""## Load GSM bank + gold content (Appendix H: content, not format keywords)"""),
    code(r'''
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]
DEGEN_FRAC = 0.5  # same cutoff as ALGO / BW mechanistic screen
VARIANTS = ("canonical", "W1", "W2", "W3", "W4", "W5", "W6")

PROBE1_TEMPLATE = (
    "Solve the following problem exactly and provide only the final answer "
    "in the required output format. Problem: {problem}. Format instruction: "
    "{family_specific_output_format}."
)
GSM_FORMAT = (
    "Write the final numerical answer on its own line as #### <number>. "
    "No other text after that tag."
)
FORMAT_KEYWORDS = {
    "path", "count", "selected", "coins", "scoops", "total", "answer",
    "final", "####", "#", ":", "[", "]", "{", "}", ",",
    "path:", "count:", "selected:", "coins:", "scoops:",
}

GSM_BANK = REPO_ROOT / "data/problems/question_bank_gsm.csv"
O7_CSV = OUT_DIR / "O7_gsm_degeneracy_check.csv"
O7_ITEMS_CSV = OUT_DIR / "O7_gsm_degeneracy_items.csv"
O7_VERDICT_TXT = OUT_DIR / "O7_gsm_degeneracy_verdict.txt"

SUMMARY_COLUMNS = [
    "family",
    "model",
    "scope",  # all | canonical
    "n_items",
    "n_distinct_gold_first_tokens",
    "modal_gold_first_token",
    "modal_count",
    "modal_share",
    "entropy_bits",
    "degen_frac_cutoff",
    "degenerate",
    "verdict",  # PASS | FAIL
]


def _norm_vt(v: str) -> str:
    v = str(v).strip()
    return "canonical" if v.lower() == "canonical" else v.upper()


def _strip_csv_quotes(text: str) -> str:
    s = str(text)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s


def gsm_gold_content(correct_answer: str) -> str:
    """Numeric answer-content span. Never #### scaffolding (same as mech notebook)."""
    s = str(correct_answer).strip()
    s = re.sub(r"^####\s*", "", s)
    s = s.replace(",", "")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        m = re.findall(r"-?\d+(?:\.\d+)?", s)
        if not m:
            raise ValueError(f"no numeric gold in {correct_answer!r}")
        return m[-1]


def build_user(problem_text: str) -> str:
    return PROBE1_TEMPLATE.format(
        problem=str(problem_text).strip(),
        family_specific_output_format=GSM_FORMAT,
    )


def load_gsm_items(limit: int | None) -> list[dict[str, Any]]:
    df = pd.read_csv(GSM_BANK, dtype=str).fillna("")
    df["problem_id"] = df["problem_id"].astype(str).str.strip()
    df["variant_type"] = df["variant_type"].map(_norm_vt)
    df["problem_text"] = df["problem_text"].map(_strip_csv_quotes)
    df["correct_answer"] = df["correct_answer"].map(_strip_csv_quotes)
    can_ids = set(df.loc[df.variant_type == "canonical", "problem_id"])
    items: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        pid = str(row["problem_id"])
        vt = str(row["variant_type"])
        if vt not in VARIANTS or pid not in can_ids:
            continue
        gold = gsm_gold_content(str(row["correct_answer"]))
        items.append(
            {
                "family": "GSM",
                "problem_id": pid,
                "variant": vt,
                "problem_text": str(row["problem_text"]),
                "correct_answer": str(row["correct_answer"]),
                "gold_content": gold,
            }
        )
    if limit is not None:
        keep_ids = sorted(can_ids)[:limit]
        keep = set(keep_ids)
        items = [x for x in items if x["problem_id"] in keep]
    return items


ITEMS = load_gsm_items(LIMIT)
print(f"[gsm] n_items={len(ITEMS)}  LIMIT={LIMIT}")
print(pd.DataFrame(ITEMS).groupby("variant").size().reindex(list(VARIANTS)).to_string())
_content_vc = Counter(x["gold_content"] for x in ITEMS)
_modal, _n = _content_vc.most_common(1)[0]
print(
    f"[gsm] gold_content (pre-tokenize) modal={_modal!r} "
    f"{_n}/{len(ITEMS)}={_n/len(ITEMS):.1%}  distinct={len(_content_vc)}"
)
'''),
    md("""## Prompt-aware first gold token + degeneracy / entropy

Same `resolve_target_token` as the mechanistic frequency-controlled notebook. Degenerate if modal decoded first-token share > 50% on **all items** or on **canonical-only**."""),
    code(r'''
def wrap_chat(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        tokenize=False,
    )


def resolve_target_token(tokenizer, prompt: str, answer: str) -> tuple[int, str, list[int], str]:
    """Prompt-aware first token of `answer` after `prompt` (mechanistic scripts)."""

    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    prompt_ids = enc(prompt)
    answer_ids_bare = enc(answer)
    candidates: list[tuple[str, int, int, list[int]]] = []
    for sep in ("", " "):
        joint = enc(prompt + sep + answer)
        if len(joint) <= len(prompt_ids):
            continue
        if joint[: len(prompt_ids)] != prompt_ids:
            continue
        rest = joint[len(prompt_ids) :]
        candidates.append((sep, int(rest[0]), len(joint), rest))
    if not candidates:
        if not answer_ids_bare:
            return -1, "", [], "EMPTY"
        tid = int(answer_ids_bare[0])
        return tid, tokenizer.decode([tid]), answer_ids_bare, "FALLBACK"
    candidates.sort(key=lambda c: c[2])
    sep, tid, _, rest = candidates[0]
    return tid, tokenizer.decode([tid]), rest, repr(sep)


def assert_content_gold(decoded: str) -> None:
    d = decoded.strip().lower()
    compact = d.replace(" ", "")
    if compact in FORMAT_KEYWORDS or d in FORMAT_KEYWORDS:
        raise AssertionError(f"gold token is a format keyword: {decoded!r}")
    if not re.search(r"\d", decoded):
        raise AssertionError(f"GSM content-gold token must contain a digit, got {decoded!r}")


def shannon_entropy_bits(tokens: list[str]) -> float:
    n = len(tokens)
    if n == 0:
        return float("nan")
    vc = Counter(tokens)
    ent = 0.0
    for c in vc.values():
        p = c / n
        ent -= p * math.log2(p)
    return float(ent)


def degeneracy_stats(tokens: list[str]) -> dict[str, Any]:
    n = len(tokens)
    if n == 0:
        return {
            "n_items": 0,
            "n_distinct_gold_first_tokens": 0,
            "modal_gold_first_token": "",
            "modal_count": 0,
            "modal_share": float("nan"),
            "entropy_bits": float("nan"),
            "degenerate": True,
        }
    vc = Counter(tokens)
    modal, n_m = vc.most_common(1)[0]
    share = n_m / n
    return {
        "n_items": n,
        "n_distinct_gold_first_tokens": len(vc),
        "modal_gold_first_token": modal,
        "modal_count": int(n_m),
        "modal_share": round(share, 6),
        "entropy_bits": round(shannon_entropy_bits(tokens), 6),
        "degenerate": bool(share > DEGEN_FRAC),
    }


def load_tokenizer(model_id: str):
    if DRY_RUN:
        # Offline pipeline check: whitespace / digit-aware fake BPE.
        class FakeTok:
            def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
                return "USER:" + messages[0]["content"] + "\nASSISTANT:"

            def encode(self, text, add_special_tokens=False):
                # Keep leading digits of numbers as separate "tokens" for a realistic screen.
                ids: list[int] = []
                i = 0
                s = str(text)
                while i < len(s):
                    if s[i].isdigit():
                        j = i
                        while j < len(s) and s[j].isdigit():
                            j += 1
                        # first-digit token id stable across numbers sharing a leading digit
                        ids.append(1000 + int(s[i]))
                        if j > i + 1:
                            ids.append(2000 + int(s[i + 1 : j] or "0") % 997)
                        i = j
                    else:
                        ids.append(10 + (ord(s[i]) % 200))
                        i += 1
                return ids

            def decode(self, ids):
                if not ids:
                    return ""
                tid = int(ids[0]) if isinstance(ids, list) else int(ids)
                if 1000 <= tid <= 1009:
                    return str(tid - 1000)
                return f"t{tid}"

        print(f"[tok] DRY_RUN fake tokenizer for {model_id}")
        return FakeTok()
    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN or True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"[tok] loaded {model_id}")
    return tok


def score_items_for_model(model_id: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tok = load_tokenizer(model_id)
    rows: list[dict[str, Any]] = []
    for it in tqdm(items, desc=model_id.split("/")[-1]):
        user = build_user(it["problem_text"])
        prompt = wrap_chat(tok, user)
        tid, decoded, gold_ids, sep_note = resolve_target_token(tok, prompt, it["gold_content"])
        assert_content_gold(decoded)
        rows.append(
            {
                "family": "GSM",
                "model": model_id,
                "problem_id": it["problem_id"],
                "variant": it["variant"],
                "gold_content": it["gold_content"],
                "gold_first_token_id": int(tid),
                "gold_first_token_decoded": decoded,
                "n_gold_token_ids": len(gold_ids),
                "sep_note": sep_note,
            }
        )
    return rows


all_item_rows: list[dict[str, Any]] = []
summary_rows: list[dict[str, Any]] = []

for model_id in MODELS:
    rows = score_items_for_model(model_id, ITEMS)
    all_item_rows.extend(rows)
    dfm = pd.DataFrame(rows)
    scopes = {
        "all": dfm["gold_first_token_decoded"].astype(str).tolist(),
        "canonical": dfm.loc[
            dfm["variant"] == "canonical", "gold_first_token_decoded"
        ].astype(str).tolist(),
    }
    model_fail = False
    for scope, toks in scopes.items():
        st = degeneracy_stats(toks)
        verd = "FAIL" if st["degenerate"] else "PASS"
        if st["degenerate"]:
            model_fail = True
        summary_rows.append(
            {
                "family": "GSM",
                "model": model_id,
                "scope": scope,
                **{k: st[k] for k in (
                    "n_items",
                    "n_distinct_gold_first_tokens",
                    "modal_gold_first_token",
                    "modal_count",
                    "modal_share",
                    "entropy_bits",
                    "degenerate",
                )},
                "degen_frac_cutoff": DEGEN_FRAC,
                "verdict": verd,
            }
        )
    print(
        f"[{model_id}] model_verdict={'FAIL' if model_fail else 'PASS'} "
        f"(cutoff={DEGEN_FRAC})"
    )

items_df = pd.DataFrame(all_item_rows)
items_df.to_csv(O7_ITEMS_CSV, index=False)
summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
summary_df.to_csv(O7_CSV, index=False)

overall_fail = bool(summary_df["degenerate"].astype(bool).any())
overall = "FAIL" if overall_fail else "PASS"

if overall == "PASS":
    paragraph = (
        f"VERDICT: PASS. GSM gold-first-token degeneracy is below the {DEGEN_FRAC:.0%} "
        f"modal-share cutoff used for ALGO/BW on both Qwen2.5-1.5B and Qwen2.5-3B "
        f"(see O7_gsm_degeneracy_check.csv). GSM becomes a second mechanistic family; "
        f"O8 should run on ALGO and GSM."
    )
else:
    fail_bits = summary_df[summary_df["degenerate"].astype(bool)][
        ["model", "scope", "modal_gold_first_token", "modal_share"]
    ]
    paragraph = (
        f"VERDICT: FAIL. GSM fails the same {DEGEN_FRAC:.0%} modal-share gold-token "
        f"degeneracy screen used for ALGO/BW on at least one (model, scope) cell "
        f"({fail_bits.to_dict(orient='records')}). Report GSM in the measurement-failure "
        f"table alongside BW; do not escalate GSM into O8 mechanistic readout."
    )

O7_VERDICT_TXT.write_text(paragraph + "\n")
print("\n=== O7_gsm_degeneracy_check.csv ===")
print(summary_df.to_string(index=False))
print("\n=== VERDICT ===")
print(paragraph)
print(f"\n[wrote] {O7_CSV}")
print(f"[wrote] {O7_ITEMS_CSV}")
print(f"[wrote] {O7_VERDICT_TXT}")
'''),
    md("""## Download / Drive backup

Copy after Colab:
- `O7_gsm_degeneracy_check.csv` → `results/derived/O7_gsm_degeneracy_check.csv`
- `O7_gsm_degeneracy_items.csv` → `results/raw/O7_gsm_degeneracy_items.csv`
- `O7_gsm_degeneracy_verdict.txt` → `results/derived/O7_gsm_degeneracy_verdict.txt`"""),
    code(r'''
_out_files = [O7_CSV, O7_ITEMS_CSV, O7_VERDICT_TXT]
_drive_dir = Path("/content/drive/MyDrive/rvc_colab_out")
if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
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


# ===========================================================================
# NOTEBOOK 6 — O8 mech↔behavior link (replaces N3)
# ===========================================================================

NB6 = [
    md("""# O8 — Mechanistic↔behavioral instrument validation (replaces N3)

**N3 failed:** the outcome was constant (Llama W3 accuracy 1/60, Qwen 0/61) — you cannot correlate against a constant.

**This notebook** correlates **per-layer** canonical→W3 gold-token **rank shift** against the **continuous** O5 outcome (`delta_mean_logprob`), not binary correctness. Binary is still reported to show it is degenerate.

### Framing constraint (read before claiming anything)
This is **instrument validation only** ("does the behavioral measure track anything internal"), **never** a mechanism claim about where a split lives. arXiv 2602.04843 (Feb 2026) already did per-layer Mystery-Blocksworld mechanistic analysis.

### Models / families
| | |
|--|--|
| Models | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct` |
| Precision | **fp16 ONLY** + `attn_implementation="sdpa"` (no 4-bit/8-bit; see O6) |
| ALGO | frozen adversarial 61 (canonical + W3) |
| GSM | included **iff O7 PASS** (auto from verdict file; override with `INCLUDE_GSM`) |
| BW | **excluded** (gold-token degeneracy) |

### Outputs
- `O8_mech_behavior_link.csv` — per instance × layer
- `O8_layer_profile.csv` — Spearman(rank_shift, y) by layer with cluster-bootstrap CIs
- `O8_framing.txt` — validation-only disclaimer

Clone-family cluster bootstrap (`n_boot=5000`, seed 42), same stack as N3."""),
    code(SETUP_PIP),
    code(SETUP_REPO),
    md("""## Knobs, O7 gate, item queue (ALGO 61 + optional GSM)

`INCLUDE_GSM=None` reads O7 verdict (`PASS` → include). Set `True`/`False` to force."""),
    code(r'''
from __future__ import annotations

import csv
import gc
import re
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes.common.clones import algo_cluster_map
from probes.common.cluster_inference import cluster_bootstrap_assoc
from probes.contamination.verify import verify_gsm_answer
from probes.contamination.verify_algo import verify_algo

# ── knobs (override setup LIMIT/DRY_RUN/RESUME as needed) ─────────────────
INCLUDE_GSM = None  # None=auto from O7; True/False force
N_BOOT = 5000 if not DRY_RUN else 200
BOOT_SEED = 42
MAX_NEW_TOKENS_GREEDY = 128  # W3 binary correctness check only

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

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
}
FORMAT_KEYWORDS = {
    "path", "count", "selected", "coins", "scoops", "total", "answer",
    "final", "####", "#", ":", "[", "]", "{", "}", ",",
    "path:", "count:", "selected:", "coins:", "scoops:",
}

# Frozen ALGO adversarial pool (rebuild/FROZEN_FILTERS.md) — same as mech notebook.
ALGO_ADV = {
    "CC": [f"CC_{i:02d}" for i in range(1, 11)],
    "SP": [
        "SP_003", "SP_004", "SP_005", "SP_019", "SP_020", "SP_021", "SP_023",
        "SP_024", "SP_026", "SP_027", "SP_028", "SP_029", "SP_030", "SP_037",
        "SP_038", "SP_039", "SP_040", "SP_042", "SP_044", "SP_045", "SP_046",
        "SP_047", "SP_048", "SP_062", "SP_063", "SP_064", "SP_065", "SP_066",
        "SP_068", "SP_069", "SP_070", "SP_071", "SP_072", "SP_073",
    ],
    "WIS": [
        "WIS_003", "WIS_004", "WIS_013", "WIS_014", "WIS_015", "WIS_016",
        "WIS_017", "WIS_018", "WIS_019", "WIS_020", "WIS_023", "WIS_024",
        "WIS_025", "WIS_026", "WIS_027", "WIS_028", "WIS_029",
    ],
}
ALGO_ADV_IDS = ALGO_ADV["CC"] + ALGO_ADV["SP"] + ALGO_ADV["WIS"]
assert len(ALGO_ADV_IDS) == 61

O5_CANDIDATES = [
    OUT_DIR / "O5_teacher_forced_likelihood.csv",
    REPO_ROOT / "results/raw/O5_teacher_forced_likelihood.csv",
    Path("/content/drive/MyDrive/rvc_colab_out/O5_teacher_forced_likelihood.csv"),
]
O7_VERDICT_CANDIDATES = [
    OUT_DIR / "O7_gsm_degeneracy_verdict.txt",
    REPO_ROOT / "results/derived/O7_gsm_degeneracy_verdict.txt",
    OUT_DIR / "O7_gsm_degeneracy_check.csv",
    REPO_ROOT / "results/derived/O7_gsm_degeneracy_check.csv",
    Path("/content/drive/MyDrive/rvc_colab_out/O7_gsm_degeneracy_verdict.txt"),
]

O8_LINK = OUT_DIR / "O8_mech_behavior_link.csv"
O8_PROFILE = OUT_DIR / "O8_layer_profile.csv"
O8_FRAMING = OUT_DIR / "O8_framing.txt"
O8_BINARY = OUT_DIR / "O8_w3_binary_scores.csv"  # per-item greedy W3 correctness

LINK_COLUMNS = [
    "family", "model", "problem_id", "layer", "n_layers",
    "rank_canonical", "rank_w3", "rank_shift_canonical_minus_w3",
    "mean_logprob_canonical", "mean_logprob_w3", "delta_mean_logprob",
    "w3_correct", "binary_degenerate_cell", "clone_family",
    "gold_content_canonical", "gold_content_w3",
    "gold_token_id_canonical", "gold_token_id_w3",
    "gold_token_decoded_canonical", "gold_token_decoded_w3",
    "framing",
]
PROFILE_COLUMNS = [
    "family", "model", "layer", "y",
    "n", "n_clusters",
    "spearman_rho", "ci_low", "ci_high", "p_value",
    "p_value_method", "bootstrap", "n_boot", "seed",
    "y_nunique", "binary_outcome_degenerate", "note", "framing",
]
FRAMING = (
    "Instrument validation only — does the behavioral measure track anything "
    "internal. Not a mechanism claim. See arXiv 2602.04843."
)


def _norm_vt(v: str) -> str:
    v = str(v).strip()
    return "canonical" if v.lower() == "canonical" else v.upper()


def _strip_csv_quotes(text: str) -> str:
    s = str(text)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s


def _norm_bank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["problem_id"] = df["problem_id"].astype(str).str.strip()
    df["variant_type"] = df["variant_type"].map(_norm_vt)
    df["problem_text"] = df["problem_text"].map(_strip_csv_quotes)
    df["correct_answer"] = df["correct_answer"].map(_strip_csv_quotes)
    return df


def resolve_o7_include_gsm() -> bool:
    if INCLUDE_GSM is not None:
        print(f"[o7] INCLUDE_GSM forced={INCLUDE_GSM}")
        return bool(INCLUDE_GSM)
    for p in O7_VERDICT_CANDIDATES:
        if not p.exists():
            continue
        if p.suffix == ".txt":
            text = p.read_text().upper()
            if "VERDICT: PASS" in text or "\nPASS" in text or text.strip().startswith("PASS"):
                print(f"[o7] PASS from {p}")
                return True
            if "VERDICT: FAIL" in text or "FAIL" in text.split("VERDICT:", 1)[-1][:20]:
                print(f"[o7] FAIL from {p}")
                return False
        if p.suffix == ".csv":
            df = pd.read_csv(p, dtype=str)
            if "verdict" in df.columns and (df["verdict"].str.upper() == "FAIL").any():
                print(f"[o7] FAIL from {p}")
                return False
            if "verdict" in df.columns and (df["verdict"].str.upper() == "PASS").all():
                print(f"[o7] PASS from {p}")
                return True
    # Local preview / O7 not uploaded yet: default PASS (GSM content screen passed in O7 dry-run).
    print("[o7] verdict file not found — defaulting INCLUDE_GSM=True (override if O7 FAIL)")
    return True


def gsm_gold_content(correct_answer: str) -> str:
    s = str(correct_answer).strip()
    s = re.sub(r"^####\s*", "", s).replace(",", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        m = re.findall(r"-?\d+(?:\.\d+)?", s)
        if not m:
            raise ValueError(f"no numeric gold in {correct_answer!r}")
        return m[-1]


def algo_gold_content(problem_id: str, correct_answer: str) -> str:
    s = str(correct_answer)
    pid = str(problem_id).strip().upper()
    if pid.startswith("SP"):
        m = re.search(r"Cost\s*:\s*(-?\d+)", s, flags=re.I)
        if not m:
            raise ValueError(f"{problem_id}: no Cost: gold")
        return m.group(1)
    if pid.startswith("CC"):
        m = re.search(r"(?:Count|Total)\s*:\s*(-?\d+)", s, flags=re.I)
        if not m:
            raise ValueError(f"{problem_id}: no Count:/Total: gold")
        return m.group(1)
    if pid.startswith("WIS"):
        m = re.search(r"Total\s*:\s*(-?\d+)", s, flags=re.I)
        if not m:
            raise ValueError(f"{problem_id}: no Total: gold")
        return m.group(1)
    raise ValueError(f"{problem_id}: unknown ALGO subtype")


def build_user(problem_text: str, family: str) -> str:
    return PROBE1_TEMPLATE.format(
        problem=str(problem_text).strip(),
        family_specific_output_format=FAMILY_FORMAT[family],
    )


def bank_row(df: pd.DataFrame, pid: str, vt: str) -> pd.Series:
    sub = df[(df.problem_id == pid) & (df.variant_type == vt)]
    if sub.empty:
        raise KeyError(f"{pid}/{vt}")
    return sub.iloc[0]


algo_df = _norm_bank(REPO_ROOT / "data/problems/question_bank_algo.csv")
gsm_df = _norm_bank(REPO_ROOT / "data/problems/question_bank_gsm.csv")
paired_algo = sorted(
    set(algo_df.loc[algo_df.variant_type == "canonical", "problem_id"])
    & set(algo_df.loc[algo_df.variant_type == "W3", "problem_id"])
)
ALGO_IDS = [pid for pid in ALGO_ADV_IDS if pid in set(paired_algo)]
assert len(ALGO_IDS) == 61, len(ALGO_IDS)

include_gsm = resolve_o7_include_gsm()
GSM_IDS = sorted(
    set(gsm_df.loc[gsm_df.variant_type == "canonical", "problem_id"])
    & set(gsm_df.loc[gsm_df.variant_type == "W3", "problem_id"])
) if include_gsm else []

cmap = algo_cluster_map()


def clone_for(family: str, pid: str) -> str:
    if family == "ALGO":
        return cmap.get(pid, f"SINGLETON_{pid}")
    return f"SINGLETON_{pid}"


def make_item(family: str, pid: str, vt: str, df: pd.DataFrame) -> dict[str, Any]:
    r = bank_row(df, pid, vt)
    if family == "ALGO":
        gold = algo_gold_content(pid, r["correct_answer"])
    else:
        gold = gsm_gold_content(r["correct_answer"])
    return {
        "family": family,
        "problem_id": pid,
        "variant": vt,
        "problem_text": str(r["problem_text"]),
        "correct_answer": str(r["correct_answer"]),
        "gold_content": gold,
        "problem_subtype": str(r.get("problem_subtype", "")).strip().lower(),
        "difficulty_params": str(r.get("difficulty_params", "{}") or "{}"),
        "clone_family": clone_for(family, pid),
    }


ITEMS: list[dict[str, Any]] = []
for pid in ALGO_IDS:
    for vt in ("canonical", "W3"):
        ITEMS.append(make_item("ALGO", pid, vt, algo_df))
for pid in GSM_IDS:
    for vt in ("canonical", "W3"):
        ITEMS.append(make_item("GSM", pid, vt, gsm_df))

if LIMIT is not None:
    keep_algo = set(ALGO_IDS[:LIMIT])
    keep_gsm = set(GSM_IDS[:LIMIT])
    ITEMS = [
        x for x in ITEMS
        if (x["family"] == "ALGO" and x["problem_id"] in keep_algo)
        or (x["family"] == "GSM" and x["problem_id"] in keep_gsm)
    ]

n_algo = len({x["problem_id"] for x in ITEMS if x["family"] == "ALGO"})
n_gsm = len({x["problem_id"] for x in ITEMS if x["family"] == "GSM"})
print(f"[queue] {len(ITEMS)} prompts  ALGO_ids={n_algo}  GSM_ids={n_gsm}  include_gsm={include_gsm}")
print(pd.DataFrame(ITEMS).groupby(["family", "variant"]).size().unstack(fill_value=0).to_string())
assert not any(x["family"] == "BW" for x in ITEMS), "BW must stay excluded"
O8_FRAMING.write_text(FRAMING + "\n")
'''),
    md("""## Logit-lens readout + O5 mean_logprob + W3 greedy binary

Per-layer gold-token rank at the last prompt position (same targeting as the mechanistic notebook).  
`delta_mean_logprob = mean_logprob_canonical − mean_logprob_w3` (parallel to `rank_shift_canonical_minus_w3`).  
W3 binary correctness via greedy decode + released verifiers (to demonstrate degeneracy)."""),
    code(r'''
def wrap_chat(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        tokenize=False,
    )


def resolve_target_token(tokenizer, prompt: str, answer: str) -> tuple[int, str, list[int], str]:
    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    prompt_ids = enc(prompt)
    answer_ids_bare = enc(answer)
    candidates: list[tuple[str, int, int, list[int]]] = []
    for sep in ("", " "):
        joint = enc(prompt + sep + answer)
        if len(joint) <= len(prompt_ids):
            continue
        if joint[: len(prompt_ids)] != prompt_ids:
            continue
        rest = joint[len(prompt_ids) :]
        candidates.append((sep, int(rest[0]), len(joint), rest))
    if not candidates:
        if not answer_ids_bare:
            return -1, "", [], "EMPTY"
        tid = int(answer_ids_bare[0])
        return tid, tokenizer.decode([tid]), answer_ids_bare, "FALLBACK"
    candidates.sort(key=lambda c: c[2])
    sep, tid, _, rest = candidates[0]
    return tid, tokenizer.decode([tid]), rest, repr(sep)


def assert_content_gold(decoded: str, family: str) -> None:
    d = decoded.strip().lower()
    compact = d.replace(" ", "")
    if compact in FORMAT_KEYWORDS or d in FORMAT_KEYWORDS:
        raise AssertionError(f"format-keyword gold token: {decoded!r}")
    if family in {"GSM", "ALGO"} and not re.search(r"\d", decoded):
        raise AssertionError(f"{family} gold token must contain a digit: {decoded!r}")


def resolve_continuation(tokenizer, prompt: str, answer: str) -> tuple[list[int], list[int]]:
    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    prompt_ids = enc(prompt)
    for sep in ("", " "):
        joint = enc(prompt + sep + str(answer))
        if len(joint) > len(prompt_ids) and joint[: len(prompt_ids)] == prompt_ids:
            return prompt_ids, joint[len(prompt_ids) :]
    return prompt_ids, enc(str(answer))


@torch.inference_mode()
def readout_layers(model, tokenizer, device, user_text: str, gold_content: str, family: str) -> dict:
    prompt = wrap_chat(tokenizer, user_text)
    tid, decoded, gold_ids, sep_note = resolve_target_token(tokenizer, prompt, gold_content)
    assert_content_gold(decoded, family)
    if DRY_RUN or model is None:
        n = 8 if DRY_RUN else int(getattr(model.config, "num_hidden_layers", 8))
        # Inject item-hash variation so continuous correlations are defined in smoke tests.
        h = abs(hash(gold_content + user_text[:40])) % 97
        ranks = [1 + ((h + i) % 50) for i in range(n)]
        return {
            "gold_token_id": tid if tid >= 0 else 1,
            "gold_token_decoded": decoded or "1",
            "sep_note": "DRY_RUN",
            "n_layers": n,
            "ranks": ranks,
            "mean_logprob": round(-0.2 - 0.01 * (h % 10), 6),
        }

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = out.hidden_states[1:]  # layer 0 = first block
    W_U = model.lm_head.weight.detach().float()
    ranks = []
    for layer_h in hidden_states:
        h = layer_h[0, -1, :].float()
        logits = h @ W_U.T
        target_logit = logits[tid]
        rank = int((logits > target_logit).sum().item()) + 1
        ranks.append(rank)
    del out, hidden_states, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Teacher-forced mean_logprob of full gold content (O5 primary metric; length-normalized).
    prompt_ids, gold_toks = resolve_continuation(tokenizer, prompt, gold_content)
    n_prompt, n_gold = len(prompt_ids), len(gold_toks)
    if n_gold == 0:
        mean_lp = float("nan")
    else:
        inp = torch.tensor([prompt_ids + gold_toks], dtype=torch.long, device=device)
        out2 = model(input_ids=inp, use_cache=False)
        glo = out2.logits[0, n_prompt - 1 : n_prompt + n_gold - 1].float()
        log_probs = F.log_softmax(glo, dim=-1)
        gt = torch.tensor(gold_toks, device=device, dtype=torch.long)
        sum_lp = float(log_probs.gather(1, gt.unsqueeze(1)).squeeze(1).sum().item())
        mean_lp = sum_lp / n_gold
        del out2, inp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "gold_token_id": int(tid),
        "gold_token_decoded": decoded,
        "sep_note": sep_note,
        "n_layers": len(ranks),
        "ranks": ranks,
        "mean_logprob": round(float(mean_lp), 6),
    }


@torch.inference_mode()
def greedy_w3_correct(model, tokenizer, device, item: dict) -> bool:
    """Binary W3 correctness for degeneracy contrast (not the primary outcome)."""
    if item["variant"] != "W3":
        raise ValueError("greedy_w3_correct expects W3 items")
    user = build_user(item["problem_text"], item["family"])
    prompt = wrap_chat(tokenizer, user)
    if DRY_RUN or model is None:
        # Near-constant False → binary degeneracy in smoke (flip ~1/40).
        return (abs(hash(item["problem_id"])) % 40) == 0
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS_GREEDY,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = gen[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    del gen, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if item["family"] == "GSM":
        return bool(verify_gsm_answer(text, item["correct_answer"]))
    ok, _reason, _meta = verify_algo(
        item["problem_id"],
        text,
        item["correct_answer"],
        item["problem_subtype"],
        "W3",
        item["difficulty_params"],
    )
    return bool(ok)


def load_o5_lookup() -> dict[tuple[str, str, str, str], float]:
    """(model, family, problem_id, variant) → mean_logprob from O5 CSV if present."""
    path = next((p for p in O5_CANDIDATES if p.exists()), None)
    if path is None:
        print("[o5] no O5 CSV found — will compute mean_logprob in-session (identical teacher-force)")
        return {}
    df = pd.read_csv(path, dtype=str)
    need = {"model", "family", "problem_id", "variant", "mean_logprob"}
    if not need.issubset(df.columns):
        print(f"[o5] {path} missing columns — ignoring")
        return {}
    out: dict[tuple[str, str, str, str], float] = {}
    for r in df.itertuples(index=False):
        try:
            out[(str(r.model), str(r.family), str(r.problem_id), str(r.variant))] = float(r.mean_logprob)
        except Exception:
            continue
    print(f"[o5] loaded {len(out)} mean_logprob cells from {path}")
    return out


def load_model_fp16(model_id: str):
    assert torch.cuda.is_available() or DRY_RUN, "GPU required unless DRY_RUN"
    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN or True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if DRY_RUN:
        print(f"[model] DRY_RUN skip weights: {model_id} fp16")
        return tok, None, torch.device("cpu")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
        token=HF_TOKEN or True,
    )
    mdl.eval()
    device = next(mdl.parameters()).device
    print(f"[model] {model_id}  fp16 + sdpa  layers={mdl.config.num_hidden_layers}  device={device}")
    return tok, mdl, device


def unload(mdl):
    if mdl is None:
        return
    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
'''),
    md("""## Run readouts → instance×layer CSV → per-layer Spearman profile

Primary y = `delta_mean_logprob` (continuous, from O5 / in-session).  
Secondary y = `w3_correct` (binary) — expect `binary_outcome_degenerate=True` when y is constant."""),
    code(r'''
o5_lookup = load_o5_lookup()


def append_link_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_header = not O8_LINK.exists()
    with O8_LINK.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LINK_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LINK_COLUMNS})


def done_pairs(path: Path) -> set[tuple[str, str, str]]:
    """Completed (model, family, problem_id) pairs (both variants written)."""
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    if not {"model", "family", "problem_id", "layer"}.issubset(df.columns):
        return set()
    # A pair is done if layer 0 exists (implies full layer stack written together).
    sub = df[pd.to_numeric(df["layer"], errors="coerce") == 0]
    return set(zip(sub["model"], sub["family"], sub["problem_id"]))


done = done_pairs(O8_LINK) if RESUME else set()
print(f"[resume] {len(done)} instance keys in {O8_LINK}")

binary_rows: list[dict[str, Any]] = []

for model_id in MODELS:
    # Unique problem keys still pending for this model
    keys = sorted({(it["family"], it["problem_id"]) for it in ITEMS})
    pending_keys = [(f, p) for f, p in keys if (model_id, f, p) not in done]
    print(f"\n=== {model_id}  pending_instances={len(pending_keys)}/{len(keys)} ===")
    if not pending_keys:
        continue
    tok, mdl, device = load_model_fp16(model_id)
    by_key: dict[tuple[str, str], dict[str, dict]] = {}
    for it in ITEMS:
        by_key.setdefault((it["family"], it["problem_id"]), {})[it["variant"]] = it

    try:
        for fam, pid in tqdm(pending_keys, desc=model_id.split("/")[-1]):
            can_it = by_key[(fam, pid)]["canonical"]
            w3_it = by_key[(fam, pid)]["W3"]
            can_user = build_user(can_it["problem_text"], fam)
            w3_user = build_user(w3_it["problem_text"], fam)
            can_m = readout_layers(mdl, tok, device, can_user, can_it["gold_content"], fam)
            w3_m = readout_layers(mdl, tok, device, w3_user, w3_it["gold_content"], fam)

            # Prefer O5 CSV mean_logprob when present for this model cell.
            mlp_can = o5_lookup.get((model_id, fam, pid, "canonical"), can_m["mean_logprob"])
            mlp_w3 = o5_lookup.get((model_id, fam, pid, "W3"), w3_m["mean_logprob"])
            delta = float(mlp_can) - float(mlp_w3)

            w3_ok = greedy_w3_correct(mdl, tok, device, w3_it)
            binary_rows.append(
                {
                    "family": fam,
                    "model": model_id,
                    "problem_id": pid,
                    "w3_correct": bool(w3_ok),
                    "clone_family": can_it["clone_family"],
                }
            )

            n_layers = min(can_m["n_layers"], w3_m["n_layers"])
            buf = []
            for layer in range(n_layers):
                rc = int(can_m["ranks"][layer])
                rw = int(w3_m["ranks"][layer])
                buf.append(
                    {
                        "family": fam,
                        "model": model_id,
                        "problem_id": pid,
                        "layer": layer,
                        "n_layers": n_layers,
                        "rank_canonical": rc,
                        "rank_w3": rw,
                        "rank_shift_canonical_minus_w3": rc - rw,
                        "mean_logprob_canonical": mlp_can,
                        "mean_logprob_w3": mlp_w3,
                        "delta_mean_logprob": round(delta, 6),
                        "w3_correct": bool(w3_ok),
                        "binary_degenerate_cell": "",  # filled in profile pass
                        "clone_family": can_it["clone_family"],
                        "gold_content_canonical": can_it["gold_content"],
                        "gold_content_w3": w3_it["gold_content"],
                        "gold_token_id_canonical": can_m["gold_token_id"],
                        "gold_token_id_w3": w3_m["gold_token_id"],
                        "gold_token_decoded_canonical": can_m["gold_token_decoded"],
                        "gold_token_decoded_w3": w3_m["gold_token_decoded"],
                        "framing": FRAMING,
                    }
                )
            append_link_rows(buf)
    finally:
        unload(mdl)

if binary_rows:
    pd.DataFrame(binary_rows).drop_duplicates(
        ["family", "model", "problem_id"], keep="last"
    ).to_csv(O8_BINARY, index=False)

link = pd.read_csv(O8_LINK)
print(f"[link] rows={len(link)}  models={sorted(link.model.unique())}")


def profile_block(sub: pd.DataFrame, family: str, model_id: str, layer: int, y_col: str) -> dict:
    x = sub["rank_shift_canonical_minus_w3"]
    y = sub[y_col]
    if y_col == "w3_correct":
        y = y.astype(str).str.lower().isin({"true", "1", "yes"}).astype(int)
    else:
        y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    clusters = sub.loc[mask, "clone_family"].astype(str).tolist()
    y_nunique = int(pd.Series(y).nunique(dropna=True))
    binary_degen = y_col == "w3_correct" and y_nunique < 2
    note = FRAMING
    if binary_degen:
        note = (
            f"BINARY DEGENERATE: {y_col} is constant "
            f"(nunique={y_nunique}, n={len(y)}, mean={float(y.mean()) if len(y) else float('nan'):.4f}). "
            "Spearman undefined — this is why N3 collapsed; use delta_mean_logprob."
        )
        return {
            "family": family,
            "model": model_id,
            "layer": layer,
            "y": y_col,
            "n": int(len(y)),
            "n_clusters": len(set(clusters)),
            "spearman_rho": "",
            "ci_low": "",
            "ci_high": "",
            "p_value": "",
            "p_value_method": "cluster_bootstrap_two_sided",
            "bootstrap": "cluster_by_clone_family",
            "n_boot": N_BOOT,
            "seed": BOOT_SEED,
            "y_nunique": y_nunique,
            "binary_outcome_degenerate": True,
            "note": note,
            "framing": FRAMING,
        }
    if int(pd.Series(x).nunique(dropna=True)) < 2:
        return {
            "family": family,
            "model": model_id,
            "layer": layer,
            "y": y_col,
            "n": int(len(y)),
            "n_clusters": len(set(clusters)),
            "spearman_rho": "",
            "ci_low": "",
            "ci_high": "",
            "p_value": "",
            "p_value_method": "cluster_bootstrap_two_sided",
            "bootstrap": "cluster_by_clone_family",
            "n_boot": N_BOOT,
            "seed": BOOT_SEED,
            "y_nunique": y_nunique,
            "binary_outcome_degenerate": False,
            "note": "rank_shift constant — correlation undefined; " + FRAMING,
            "framing": FRAMING,
        }
    res = cluster_bootstrap_assoc(
        x, y, clusters, kind="spearman", n_boot=N_BOOT, seed=BOOT_SEED
    )

    def _r(v):
        return round(float(v), 4) if v == v else ""

    return {
        "family": family,
        "model": model_id,
        "layer": layer,
        "y": y_col,
        "n": res["n"],
        "n_clusters": res["n_clusters"],
        "spearman_rho": _r(res["estimate"]),
        "ci_low": _r(res["ci_low"]),
        "ci_high": _r(res["ci_high"]),
        "p_value": _r(res["p_clustered"]),
        "p_value_method": "cluster_bootstrap_two_sided",
        "bootstrap": "cluster_by_clone_family",
        "n_boot": N_BOOT,
        "seed": BOOT_SEED,
        "y_nunique": y_nunique,
        "binary_outcome_degenerate": False,
        "note": note,
        "framing": FRAMING,
    }


profile_rows: list[dict] = []
for (fam, model_id), g in link.groupby(["family", "model"]):
    # Mark binary degeneracy at the instance level for this model×family
    inst = g.drop_duplicates(["problem_id"])
    ybin = inst["w3_correct"].astype(str).str.lower().isin({"true", "1", "yes"})
    bin_degen = int(ybin.nunique()) < 2
    if bin_degen:
        print(
            f"[binary] DEGENERATE  {model_id} / {fam}: "
            f"w3_correct nunique={ybin.nunique()}  "
            f"acc={float(ybin.mean()):.4f}  n={len(ybin)}"
        )
    else:
        print(
            f"[binary] ok variation  {model_id} / {fam}: "
            f"acc={float(ybin.mean()):.4f}  n={len(ybin)}"
        )
    for layer, gl in g.groupby(pd.to_numeric(g["layer"], errors="coerce")):
        if pd.isna(layer):
            continue
        layer_i = int(layer)
        for y_col in ("delta_mean_logprob", "w3_correct"):
            profile_rows.append(profile_block(gl, fam, model_id, layer_i, y_col))

prof = pd.DataFrame(profile_rows, columns=PROFILE_COLUMNS)
prof.to_csv(O8_PROFILE, index=False)

print("\n=== O8_layer_profile.csv (final layer only, preview) ===")
final_layers = (
    link.groupby(["family", "model"])["n_layers"].first().astype(int) - 1
)
preview = []
for (fam, model_id), g in prof.groupby(["family", "model"]):
    fl = int(final_layers.get((fam, model_id), g["layer"].max()))
    preview.append(g[g["layer"] == fl])
if preview:
    print(pd.concat(preview).to_string(index=False))

print(f"\n[wrote] {O8_LINK} ({len(link)} rows)")
print(f"[wrote] {O8_PROFILE} ({len(prof)} rows)")
print(f"[wrote] {O8_FRAMING}")
print("\n" + FRAMING)
'''),
    md("""## Download / Drive backup

Copy after Colab:
- `O8_mech_behavior_link.csv` → `results/raw/O8_mech_behavior_link.csv`
- `O8_layer_profile.csv` → `results/derived/O8_layer_profile.csv`
- `O8_w3_binary_scores.csv` → `results/raw/O8_w3_binary_scores.csv`
- `O8_framing.txt` → `results/derived/O8_framing.txt`"""),
    code(r'''
_out_files = [O8_LINK, O8_PROFILE, O8_BINARY, O8_FRAMING]
_drive_dir = Path("/content/drive/MyDrive/rvc_colab_out")
if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
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


def write(name: str, cells: list[dict]) -> None:
    path = OUT / name
    payload = nb(cells, name)
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    write("llama_greedy_behavioural.ipynb", NB1)
    write("mechanistic_frequency_controlled.ipynb", NB2)
    write("o5_teacher_forced_likelihood.ipynb", NB3)
    write("o6_quantization_sensitivity.ipynb", NB4)
    write("o7_gsm_degeneracy_check.ipynb", NB5)
    write("o8_mech_behavior_link.ipynb", NB6)
    # O15 lives in a sibling builder to keep this file smaller.
    import importlib.util

    _o15_spec = importlib.util.spec_from_file_location(
        "_rvc_build_o15", Path(__file__).resolve().parent / "_build_o15.py"
    )
    assert _o15_spec is not None and _o15_spec.loader is not None
    _o15 = importlib.util.module_from_spec(_o15_spec)
    _o15_spec.loader.exec_module(_o15)
    _o15.build()

    _o16_spec = importlib.util.spec_from_file_location(
        "_rvc_build_o16", Path(__file__).resolve().parent / "_build_o16.py"
    )
    assert _o16_spec is not None and _o16_spec.loader is not None
    _o16 = importlib.util.module_from_spec(_o16_spec)
    _o16_spec.loader.exec_module(_o16)
    _o16.build()
