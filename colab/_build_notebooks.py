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


def write(name: str, cells: list[dict]) -> None:
    path = OUT / name
    payload = nb(cells, name)
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    write("llama_greedy_behavioural.ipynb", NB1)
    write("mechanistic_frequency_controlled.ipynb", NB2)
