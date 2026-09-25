#!/usr/bin/env python3
"""Build Track T Colab notebooks (T1 noise floor, T2 Qwen family, T3 precision).

Usage (from repo root):
  python colab/_build_trackT.py

Do not run the notebooks here — they are for Colab T4 with secrets.
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


CELL_ENV = r'''
# Cell 0 — environment fingerprint (run first; required by HP-16 validate header)
import platform
import subprocess
import sys

print("python:", sys.version.replace("\n", " "))
print("platform:", platform.platform())

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu_name:", torch.cuda.get_device_name(0))
        print("cuda:", torch.version.cuda)
    else:
        print("gpu_name:", None)
        print("cuda:", None)
except Exception as e:
    print("torch: UNAVAILABLE", e)

try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as e:
    print("transformers: UNAVAILABLE", e)

try:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True)
    print("nvidia-smi:", out.strip())
except Exception as e:
    print("nvidia-smi: UNAVAILABLE", e)
'''

CELL_CLONE = r'''
# Clone repo (public) or with Colab secret GITHUB_TOKEN for private fetch.
# Never hard-code tokens.
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = os.environ.get(
    "RVC_REPO_URL",
    "https://github.com/Adya6714/retrieval-vs-computation.git",
)
REPO_COMMIT = os.environ.get("RVC_REPO_COMMIT", "")  # empty = default branch HEAD
BRANCH = os.environ.get("RVC_BRANCH", "main")

def _secret(name: str) -> str:
    v = os.environ.get(name, "")
    if v:
        return v
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(name) or ""
    except Exception:
        return ""

GH_TOKEN = _secret("GITHUB_TOKEN") or _secret("RVC_PUSH_TOKEN")
HF_TOKEN = _secret("HF_TOKEN") or _secret("HUGGING_FACE_HUB_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception as exc:
        print("[setup] HF login skipped:", exc)

def _looks_like_repo(p: Path) -> bool:
    return (p / "data" / "problems" / "question_bank_gsm.csv").is_file() and (
        p / "scripts" / "trackT"
    ).is_dir()

def _find_repo() -> Path:
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        if _looks_like_repo(cand):
            return cand
    return Path("/content/retrieval-vs-computation")

REPO_ROOT = _find_repo()
if not _looks_like_repo(REPO_ROOT):
    REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)
    url = REPO_URL
    if GH_TOKEN and "github.com" in url and url.startswith("https://"):
        url = url.replace("https://", f"https://{GH_TOKEN}@")
    print(f"[setup] cloning {REPO_URL} → {REPO_ROOT}")
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", BRANCH, url, str(REPO_ROOT)])
    if REPO_COMMIT:
        subprocess.check_call(["git", "-C", str(REPO_ROOT), "fetch", "--depth", "1", "origin", REPO_COMMIT])
        subprocess.check_call(["git", "-C", str(REPO_ROOT), "checkout", REPO_COMMIT])

assert _looks_like_repo(REPO_ROOT), f"Repo not found at {REPO_ROOT}"
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
print(f"[setup] REPO_ROOT={REPO_ROOT}")
print(f"[setup] HEAD=", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip())
'''

CELL_PIP_BASE = r'''
# Install repo requirements + Track T extras (restart runtime if bitsandbytes was just added).
import subprocess
import sys
from pathlib import Path

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "pip"])
req = Path("requirements.txt")
assert req.is_file(), "run the clone cell first"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])
'''

CELL_PIP_T1 = CELL_PIP_BASE + r'''
extras = [
    "torch",
    "transformers>=4.44",
    "accelerate>=0.33",
    "huggingface_hub",
    "sentencepiece",
    "protobuf",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *extras])
print("[pip] T1 extras installed")
'''

CELL_PIP_T2 = CELL_PIP_BASE + r'''
extras = [
    "torch",
    "transformers>=4.44",
    "accelerate>=0.33",
    "huggingface_hub",
    "sentencepiece",
    "protobuf",
    "statsmodels>=0.14",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *extras])
print("[pip] T2 extras installed")
'''

CELL_PIP_T3 = CELL_PIP_BASE + r'''
extras = [
    "torch",
    "transformers>=4.44",
    "accelerate>=0.33",
    "huggingface_hub",
    "sentencepiece",
    "protobuf",
    "bitsandbytes>=0.43",
    "optimum>=1.21",
]
# GPTQ / AWQ loaders — optional; HP-18 skips arms when published checkpoints are missing.
for pkg in ["auto-gptq", "autoawq"]:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", pkg])
        print(f"[pip] {pkg} ok")
    except subprocess.CalledProcessError as e:
        print(f"[pip] {pkg} skipped ({e})")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *extras])
print("[pip] T3 extras installed")
'''

CELL_PUSH = r'''
# Push raw/derived Track T artefacts. Requires Colab secret RVC_PUSH_TOKEN
# (or GITHUB_TOKEN) with contents:write. Never hard-code a token.
from __future__ import annotations

import os
import subprocess
from pathlib import Path

def _secret(name: str) -> str:
    v = os.environ.get(name, "")
    if v:
        return v
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(name) or ""
    except Exception:
        return ""

TOKEN = _secret("RVC_PUSH_TOKEN") or _secret("GITHUB_TOKEN")
assert TOKEN, "Set Colab secret RVC_PUSH_TOKEN (preferred) or GITHUB_TOKEN before pushing."

root = Path.cwd()
assert (root / "results" / "raw").is_dir()

subprocess.check_call(["git", "config", "user.email", "colab-trackt@users.noreply.github.com"])
subprocess.check_call(["git", "config", "user.name", "RvC Track T Colab"])
subprocess.check_call(["git", "add", "results/raw", "results/derived", "results/figures", "docs/trackT"])
status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
if not status.strip():
    print("[push] nothing to commit")
else:
    print(status)
    msg = os.environ.get("RVC_COMMIT_MSG", "Track T Colab: append raw/derived outputs")
    subprocess.check_call(["git", "commit", "-m", msg])
    # rewrite origin to authenticated URL for this push only
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    if origin.startswith("https://") and "@" not in origin.split("://", 1)[1]:
        auth = origin.replace("https://", f"https://x-access-token:{TOKEN}@")
        subprocess.check_call(["git", "remote", "set-url", "origin", auth])
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    subprocess.check_call(["git", "push", "-u", "origin", "HEAD"])
    # scrub token from remote URL
    if "x-access-token:" in origin or origin.startswith("https://"):
        clean = origin
        if "x-access-token:" in clean:
            # already cleaned below
            pass
        clean = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
        if "x-access-token:" in clean:
            scrubbed = "https://github.com/" + clean.split("github.com/")[-1]
            subprocess.check_call(["git", "remote", "set-url", "origin", scrubbed])
    print("[push] done on", branch)
'''


def build_t1() -> list[dict]:
    return [
        md("""# T1 — Inference-stack noise floor (HP-16)

Colab **T4** runner for [[HP-16_Noise_Floor]] / [[D11_Inference_Stack_Noise_Floor]].

**Do not interpret results** — append raw CSVs only; derived tables are regenerated by script.

**Secrets (Colab → 🔑):** optional `HF_TOKEN`; `RVC_PUSH_TOKEN` (or `GITHUB_TOKEN`) for the push cell.

**Outputs**
- `results/raw/T1_noise_floor_{model_slug}.csv`
- `results/derived/T1_noise_floor_metrics.csv`
- `docs/trackT/T1_REPORT.md` (tables only)
"""),
        code(CELL_ENV),
        code(CELL_CLONE),
        code(CELL_PIP_T1),
        md("## Run HP-16 sweeps with `--resume`"),
        code(r'''
# Knobs
DRY_RUN = False   # True: schema-only / MockClient path (no GPU)
FAMILIES = ["GSM", "ALGO"]
MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

from pathlib import Path
import subprocess, sys

assert Path("scripts/trackT/T1_noise_floor.py").is_file()
for model in MODELS:
    for family in FAMILIES:
        cmd = [
            sys.executable, "scripts/trackT/T1_noise_floor.py",
            "--model", model,
            "--family", family,
            "--configs", "all",
            "--resume",
        ]
        if DRY_RUN:
            cmd.append("--dry-run")
        print(">>", " ".join(cmd))
        subprocess.check_call(cmd)

print(">> metrics")
subprocess.check_call([sys.executable, "scripts/trackT/T1_noise_floor_metrics.py"])
'''),
        md("## Push raw + derived (requires `RVC_PUSH_TOKEN`)"),
        code(CELL_PUSH),
    ]


def build_t2() -> list[dict]:
    return [
        md("""# T2 — Qwen base vs Coder vs Math (HP-17)

Colab **T4** runner for [[HP-17_Qwen_Family_Contrast]] / [[D13_Continued_Pretraining_Transfer]].

Uses the T1-stable inference config. Pre-register the Coder × W4 contrast in `PREREGISTRATION.md` before the first non-dry run.

**Secrets:** optional `HF_TOKEN`; `RVC_PUSH_TOKEN` for push.

**Outputs**
- `results/raw/T2_P1_{model_slug}.csv`
- existing P1 metric scripts against those raws
- `results/derived/T2_mixed_model.csv`
- `docs/trackT/T2_REPORT.md`
"""),
        code(CELL_ENV),
        code(CELL_CLONE),
        code(CELL_PIP_T2),
        md("## Run HP-17 local Probe 1 with `--resume`"),
        code(r'''
DRY_RUN = False
MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
]

from pathlib import Path
import subprocess, sys

assert Path("scripts/trackT/run_local_p1.py").is_file()
for model in MODELS:
    cmd = [
        sys.executable, "scripts/trackT/run_local_p1.py",
        "--model", model,
        "--families", "GSM,ALGO",
        "--subtypes", "coin_change,shortest_path",
        "--resume",
    ]
    if DRY_RUN:
        cmd.append("--dry-run")
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

# Existing metric scripts (raw-glob); do not fork metric logic.
for fam, script in [
    ("GSM", "scripts/GSM_P1_SCR_compute_metrics.py"),
    ("ALGO", "scripts/ALGO_P1_SCR_compute_metrics.py"),
]:
    if Path(script).is_file():
        cmd = [sys.executable, script, "--raw-glob", f"results/raw/T2_P1_*.csv"]
        print(">>", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[warn] {script} exited {e.returncode} — check --raw-glob support")

print(">> mixed model")
subprocess.check_call([sys.executable, "scripts/trackT/T2_mixed_model.py"])
'''),
        md("## Push raw + derived (requires `RVC_PUSH_TOKEN`)"),
        code(CELL_PUSH),
    ]


def build_t3() -> list[dict]:
    return [
        md("""# T3 — Precision sweep + per-band map (HP-18)

Colab **T4** runner for [[HP-18_Precision_Sweep]] / [[D12_Precision_Compression_Invariance]].

Extras: `bitsandbytes`, `optimum`; `auto-gptq` / `autoawq` when installable. Published GPTQ/AWQ checkpoints only — do not quantize yourself in this HP.

**Secrets:** optional `HF_TOKEN`; `RVC_PUSH_TOKEN` for push.

**Outputs**
- `results/raw/T3_P1_{model_slug}_{precision}.csv`
- `results/derived/T3_precision_metrics.csv`, `T3_band_map.csv`
- `results/figures/T3_band_map.pdf`
- `docs/trackT/T3_REPORT.md`
"""),
        code(CELL_ENV),
        code(CELL_CLONE),
        code(CELL_PIP_T3),
        md("## Run HP-18 precision + band arms with `--resume`"),
        code(r'''
DRY_RUN = False
MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
]
# fp16 reuses T2 when present; int8/nf4 via bitsandbytes; gptq/awq skip if unavailable.
PRECISIONS = ["fp16", "int8", "nf4", "gptq_int4", "awq_int4"]

from pathlib import Path
import subprocess, sys

assert Path("scripts/trackT/T3_precision_sweep.py").is_file()
assert Path("scripts/trackT/fake_quant.py").is_file()

for model in MODELS:
    for prec in PRECISIONS:
        cmd = [
            sys.executable, "scripts/trackT/T3_precision_sweep.py",
            "--model", model,
            "--precision", prec,
            "--resume",
        ]
        if DRY_RUN:
            cmd.append("--dry-run")
        print(">>", " ".join(cmd))
        subprocess.check_call(cmd)

    # Per-band fake INT4 (4 contiguous decoder bands)
    for band in range(4):
        cmd = [
            sys.executable, "scripts/trackT/fake_quant.py",
            "--model", model,
            "--band", str(band),
            "--n-bands", "4",
            "--resume",
        ]
        if DRY_RUN:
            cmd.append("--dry-run")
        print(">>", " ".join(cmd))
        subprocess.check_call(cmd)

print(">> metrics")
subprocess.check_call([sys.executable, "scripts/trackT/T3_precision_metrics.py"])
'''),
        md("## Push raw + derived (requires `RVC_PUSH_TOKEN`)"),
        code(CELL_PUSH),
    ]


def main() -> None:
    specs = [
        ("T1_noise_floor.ipynb", build_t1()),
        ("T2_qwen_family.ipynb", build_t2()),
        ("T3_precision.ipynb", build_t3()),
    ]
    for name, cells in specs:
        path = OUT / name
        path.write_text(json.dumps(nb(cells, name), indent=1) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
