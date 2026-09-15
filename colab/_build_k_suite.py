#!/usr/bin/env python3
"""Build colab/k_suite_colab.ipynb — K1–K7 in one Colab runtime.

Shared pip/repo setup once. Each arm keeps its own output filenames so downloads
stay separate. Toggle arms with RUN_K* flags; Run-all skips disabled arms.
"""

from __future__ import annotations

import copy
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

md = _bn.md
code = _bn.code
nb = _bn.nb
SETUP_PIP = _bn.SETUP_PIP
SETUP_REPO = _bn.SETUP_REPO
NB3 = _bn.NB3  # O5 = K1
NB4 = _bn.NB4  # O6 = K2
NB6 = _bn.NB6  # O8 = K4


def _load_sibling(name: str, attr: str):
    spec = importlib.util.spec_from_file_location(
        f"_rvc_{name}", OUT / f"_build_{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


NB_O15 = _load_sibling("o15", "NB_O15")
NB_O16 = _load_sibling("o16", "NB_O16")
NB_O14B = _load_sibling("o14b", "NB_O14B")
NB_DS16 = _load_sibling("ds16", "NB_DS16")


def _src(cell: dict) -> str:
    return "".join(cell.get("source") or [])


def _is_pip_cell(cell: dict) -> bool:
    s = _src(cell)
    return cell.get("cell_type") == "code" and (
        "bitsandbytes for quantized" in s[:240] or "pip\", \"install\"" in s[:400]
    )


def _is_repo_cell(cell: dict) -> bool:
    s = _src(cell)
    return (
        cell.get("cell_type") == "code"
        and "REPO_URL" in s
        and "DRY_RUN" in s
        and "REPO_ROOT" in s
    )


def _is_download_cell(cell: dict) -> bool:
    s = _src(cell).lower()
    if cell.get("cell_type") == "markdown":
        return "download" in s[:80] and ("drive" in s or "copy" in s or "`colab_out" in s)
    return "files.download" in s or "_out_files" in s or "_colab_files.download" in s


def strip_setup(cells: list[dict]) -> tuple[dict | None, list[dict]]:
    """Drop intro title + SETUP_PIP + SETUP_REPO; keep remaining body."""
    intro = None
    body: list[dict] = []
    skipped_pip = skipped_repo = False
    for i, cell in enumerate(cells):
        if i == 0 and cell.get("cell_type") == "markdown":
            intro = cell
            continue
        if not skipped_pip and _is_pip_cell(cell):
            skipped_pip = True
            continue
        if not skipped_repo and _is_repo_cell(cell):
            skipped_repo = True
            continue
        if _is_download_cell(cell):
            continue
        body.append(copy.deepcopy(cell))
    return intro, body


def wrap_source(src: str, flag: str, label: str) -> str:
    """Gate a code cell on ``flag``; peel ``from __future__`` to the top."""
    lines = src.splitlines(keepends=True)
    futures: list[str] = []
    rest: list[str] = []
    for line in lines:
        if line.lstrip().startswith("from __future__"):
            futures.append(line if line.endswith("\n") else line + "\n")
        else:
            rest.append(line)
    body = "".join(rest)
    if not body.strip():
        return "".join(futures) + body
    indented = ""
    for line in body.splitlines(keepends=True):
        if not line.endswith("\n"):
            line = line + "\n"
        if line.strip() == "":
            indented += "\n"
        else:
            indented += "    " + line
    return (
        "".join(futures)
        + f"if not ({flag}):\n"
        + f"    print('[skip] {label}  ({flag}=False)')\n"
        + "else:\n"
        + indented
    )


def wrap_body(cells: list[dict], flag: str, label: str) -> list[dict]:
    out: list[dict] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            out.append(cell)
            continue
        out.append(code(wrap_source(_src(cell), flag, label)))
    return out


def inject_after_models(cells: list[dict], snippet: str, needle: str) -> None:
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        s = _src(cell)
        if needle not in s:
            continue
        cell["source"] = _bn._split(s.rstrip() + "\n\n" + snippet.strip() + "\n")
        return


def replace_in_body(cells: list[dict], old: str, new: str) -> None:
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        s = _src(cell)
        if old not in s:
            continue
        cell["source"] = _bn._split(s.replace(old, new))


def download_cell(flag: str, label: str, filenames: list[str]) -> dict:
    files_lit = ", ".join(repr(f) for f in filenames)
    return code(
        f"""
if not ({flag}):
    print("[skip download] {label}")
else:
    from pathlib import Path as _P
    import shutil as _shutil
    _names = [{files_lit}]
    _paths = [OUT_DIR / n for n in _names]
    _present = [p for p in _paths if p.is_file()]
    print(f"[download {label}] present={{len(_present)}}/{{len(_paths)}} in {{OUT_DIR}}")
    for p in _paths:
        print(" ", "OK" if p.is_file() else "MISSING", p.name)
    _drive_dir = _P("/content/drive/MyDrive/rvc_colab_out")
    if _P("/content").exists() and not _P("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive")
        except Exception as _exc:
            print("[drive] mount skipped:", _exc)
    try:
        _drive_dir.mkdir(parents=True, exist_ok=True)
        for p in _present:
            _shutil.copy2(p, _drive_dir / p.name)
            print(f"[backup] {{p.name}} -> {{_drive_dir / p.name}}")
    except Exception as _exc:
        print("[backup] skipped:", _exc)
    try:
        from google.colab import files as _colab_files  # type: ignore
        for p in _present:
            _colab_files.download(str(p))
            print(f"[download] {{p.name}}")
    except Exception as _exc:
        print("[download] skipped (not Colab or blocked):", _exc)
""".strip(
            "\n"
        )
        + "\n"
    )


SUITE_KNOBS = r'''
# ═══════════════════════════════════════════════════════════════════════════
# K-suite master knobs (edit here; then Runtime → Run all)
# ═══════════════════════════════════════════════════════════════════════════
# Smoke first: LIMIT=2, DRY_RUN=True  →  then LIMIT=None, DRY_RUN=False
# Shared LIMIT/DRY_RUN/RESUME come from SETUP_REPO above.

RUN_K1_O5 = True       # teacher-forced likelihood (~4299 rows; Qwen primary)
RUN_K2_O6 = True       # quantization sensitivity (60 items; after K1)
RUN_K3_O15 = True      # surprisal contamination
RUN_K4_O8 = True       # mech↔behavior link (needs K1 CSV + O7 PASS)
RUN_K6_O14B = True     # naming likelihood (Qwen 1.5B/3B)
RUN_K7_DS16 = True     # recognition vs recall
RUN_K5_O16 = True      # open-model calibration — run last (sparse GT caveat)

# K1: skip gated Llama when no HF_TOKEN (Qwen arms are primary)
SKIP_LLAMA = None      # None = auto (skip if HF_TOKEN missing); True/False force

# K3: first pass = fp16 primaries only
RUN_OPTIONAL_SCALE = False

# K4: ALGO + GSM (O7 PASS already recorded for GSM)
INCLUDE_GSM = True

# K6 / K7 smoke overrides (None = full)
LIMIT_PAIRS = None           # O14b pairs (×3 arms)
LIMIT_PER_FAMILY = None      # DS16 problem_ids per family
K_DISTRACTORS = 4
MAX_NEW_TOKENS_BW = 512      # O14b greedy
N_BOOT = 5000
SEED = 42
FLOOR_ACC = 0.05
MIN_K_PCT = 20
DS16_MAX_NEW_TOKENS = {"GSM": 256, "ALGO": 192, "BW": 512}
VARIANTS = ("canonical", "W3")  # DS16 default; other arms redefine locally

if SKIP_LLAMA is None:
    SKIP_LLAMA = not bool(HF_TOKEN)
print("[suite] RUN flags:",
      {k: v for k, v in globals().items() if k.startswith("RUN_K")})
print(f"[suite] SKIP_LLAMA={SKIP_LLAMA}  RUN_OPTIONAL_SCALE={RUN_OPTIONAL_SCALE}  "
      f"INCLUDE_GSM={INCLUDE_GSM}  LIMIT={LIMIT}  DRY_RUN={DRY_RUN}")
'''


LLAMA_FILTER = '''
# Suite gate: drop Llama when SKIP_LLAMA (needs HF_TOKEN for gated weights)
if SKIP_LLAMA:
    _before = list(MODELS)
    MODELS = [
        m for m in MODELS
        if "llama" not in (m[0] if isinstance(m, (tuple, list)) else str(m)).lower()
    ]
    print(f"[suite] SKIP_LLAMA filtered MODELS {_before} → {MODELS}")
'''


def section(
    key: str,
    flag: str,
    title: str,
    blurb: str,
    raw_cells: list[dict],
    out_files: list[str],
    *,
    models_needle: str | None = None,
    replacements: list[tuple[str, str]] | None = None,
    preamble: str | None = None,
) -> list[dict]:
    intro, body = strip_setup(raw_cells)
    if replacements:
        for old, new in replacements:
            replace_in_body(body, old, new)
    if models_needle:
        inject_after_models(body, LLAMA_FILTER, models_needle)
    if preamble:
        body = [code(preamble)] + body
    gated = wrap_body(body, flag, key)
    cells: list[dict] = [
        md(f"""---
# {title}

{blurb}

**Flag:** `{flag}` · **Outputs (download separately):** {", ".join(f"`{f}`" for f in out_files)}
"""),
    ]
    if intro is not None:
        # Keep original title as a collapsed reference under the suite banner
        src = _src(intro).strip()
        # Avoid duplicating a full H1 — demote to note
        cells.append(
            md(
                "<details><summary>Arm README (from standalone notebook)</summary>\n\n"
                + src
                + "\n\n</details>"
            )
        )
    cells.extend(gated)
    cells.append(md(f"### Download — {key} only"))
    cells.append(download_cell(flag, key, out_files))
    return cells


def build() -> Path:
    cells: list[dict] = [
        md("""# K-suite — Colab T4 (K1–K7)

One runtime, **shared setup**, **separate downloadable CSVs** per arm.

| Order | Flag | Arm | Notes |
|------:|------|-----|-------|
| 1 | `RUN_K1_O5` | O5 teacher-forced likelihood | Highest priority. Smoke `LIMIT=2`, `DRY_RUN=True` first. Qwen primary; Llama skipped without `HF_TOKEN`. |
| 2 | `RUN_K2_O6` | O6 quantization sensitivity | After K1. Kill: median rank shift > 50 drops 4-bit ranks. |
| 3 | `RUN_K3_O15` | O15 surprisal contamination | First pass: `RUN_OPTIONAL_SCALE=False`. |
| 4 | `RUN_K4_O8` | O8 mech↔behavior link | Needs K1 CSV + O7 PASS (GSM). ALGO + GSM. |
| 5 | `RUN_K6_O14B` | O14b naming likelihood | Qwen 1.5B/3B; independent of O5 grid. |
| 6 | `RUN_K7_DS16` | DS-16 recognition/recall | Qwen 1.5B/3B; can+W3. |
| 7 | `RUN_K5_O16` | O16 open-model calibration | **Last.** Sparse corpus GT → expect uninformative AUC; still closes the loop. |

### Workflow
1. Set Colab secrets: `HF_TOKEN` (optional, Llama only), `GITHUB_TOKEN` if private clone.
2. Edit **master knobs** in the next code cell (`LIMIT`, `DRY_RUN`, `RUN_K*`).
3. **Runtime → Run all** (or run section-by-section).
4. After each arm (or at the end), use that arm’s **Download** cell — files stay distinct.

Land downloads in the repo per `colab/README.md` (`results/raw/` / `results/derived/`).
"""),
        code(SETUP_PIP),
        code(SETUP_REPO + "\n" + SUITE_KNOBS),
    ]

    cells += section(
        "K1_O5",
        "RUN_K1_O5",
        "K1 — O5 teacher-forced likelihood",
        "Unblocks C6, O8, and idle cells. ~full P1 grid × models. "
        "`SKIP_LLAMA` drops gated Llama when no token.",
        NB3,
        ["O5_teacher_forced_likelihood.csv"],
        models_needle="meta-llama/Llama-3.1-8B-Instruct",
    )

    cells += section(
        "K2_O6",
        "RUN_K2_O6",
        "K2 — O6 quantization sensitivity",
        "60 stratified items on Qwen2.5-1.5B. Run right after K1.",
        NB4,
        [
            "O6_quantization_sensitivity.csv",
            "O6_quantization_sensitivity_items.csv",
            "O6_quantization_sensitivity_summary.txt",
            "O6_subsample_manifest.json",
        ],
    )

    cells += section(
        "K3_O15",
        "RUN_K3_O15",
        "K3 — O15 surprisal contamination",
        "Typicality-vs-contamination reframe. Suite sets `RUN_OPTIONAL_SCALE=False` "
        "for the first pass (fp16 primaries only).",
        NB_O15,
        ["O15_surprisal_contamination.csv"],
    )

    cells += section(
        "K4_O8",
        "RUN_K4_O8",
        "K4 — O8 mechanistic↔behavioral link",
        "Requires `O5_teacher_forced_likelihood.csv` in `colab_out/` (from K1) and "
        "O7 PASS for GSM. Suite forces `INCLUDE_GSM=True`.",
        NB6,
        [
            "O8_mech_behavior_link.csv",
            "O8_layer_profile.csv",
            "O8_w3_binary_scores.csv",
            "O8_framing.txt",
        ],
        replacements=[
            (
                "INCLUDE_GSM = None  # None=auto from O7; True/False force",
                "INCLUDE_GSM = True  # suite: ALGO+GSM (O7 PASS); edit master knobs to override",
            ),
        ],
    )

    cells += section(
        "K6_O14B",
        "RUN_K6_O14B",
        "K6 — O14b naming likelihood",
        "Controlled BW naming bank; Qwen 1.5B/3B teacher-forced + greedy.",
        NB_O14B,
        ["O14b_naming_likelihood.csv", "O14b_naming_analysis.csv"],
        preamble="MAX_NEW_TOKENS = int(MAX_NEW_TOKENS_BW)\nprint('[K6] MAX_NEW_TOKENS', MAX_NEW_TOKENS, 'LIMIT_PAIRS', LIMIT_PAIRS)\n",
    )

    cells += section(
        "K7_DS16",
        "RUN_K7_DS16",
        "K7 — DS-16 recognition vs recall",
        "Memory-signature arm; distractors from `scripts/consolidate/ds16_distractors.py`.",
        NB_DS16,
        ["DS16_recognition_recall.csv", "DS16_gap_correlations.csv"],
        preamble=(
            "MAX_NEW_TOKENS = dict(DS16_MAX_NEW_TOKENS)\n"
            "VARIANTS = ('canonical', 'W3')\n"
            "print('[K7] MAX_NEW_TOKENS', MAX_NEW_TOKENS, 'LIMIT_PER_FAMILY', LIMIT_PER_FAMILY, "
            "'K_DISTRACTORS', K_DISTRACTORS)\n"
        ),
    )

    cells += section(
        "K5_O16",
        "RUN_K5_O16",
        "K5 — O16 open-model calibration (last)",
        "Close the loop. With ~0 exact / 2 near-exact corpus members, surprisal ROC "
        "has almost no positives — expect an uninformative AUC; still run to finish.",
        NB_O16,
        ["O16_open_model_scores.csv"],
    )

    cells += [
        md("""## Final — download every present suite artifact

Runs regardless of individual `RUN_K*` (downloads whatever files exist in `colab_out/`).
"""),
        code(
            r'''
from pathlib import Path as _P
import shutil as _shutil

_ALL = [
    "O5_teacher_forced_likelihood.csv",
    "O6_quantization_sensitivity.csv",
    "O6_quantization_sensitivity_items.csv",
    "O6_quantization_sensitivity_summary.txt",
    "O6_subsample_manifest.json",
    "O15_surprisal_contamination.csv",
    "O8_mech_behavior_link.csv",
    "O8_layer_profile.csv",
    "O8_w3_binary_scores.csv",
    "O8_framing.txt",
    "O14b_naming_likelihood.csv",
    "O14b_naming_analysis.csv",
    "DS16_recognition_recall.csv",
    "DS16_gap_correlations.csv",
    "O16_open_model_scores.csv",
]
_paths = [OUT_DIR / n for n in _ALL]
_present = [p for p in _paths if p.is_file()]
print(f"[suite final] {len(_present)}/{len(_ALL)} files in {OUT_DIR}")
for p in _paths:
    print(" ", "OK" if p.is_file() else "·", p.name)

_drive_dir = _P("/content/drive/MyDrive/rvc_colab_out")
if _P("/content").exists() and not _P("/content/drive/MyDrive").exists():
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
    except Exception as _exc:
        print("[drive] mount skipped:", _exc)
try:
    _drive_dir.mkdir(parents=True, exist_ok=True)
    for p in _present:
        _shutil.copy2(p, _drive_dir / p.name)
        print(f"[backup] {p.name}")
except Exception as _exc:
    print("[backup] skipped:", _exc)
try:
    from google.colab import files as _colab_files  # type: ignore
    for p in _present:
        _colab_files.download(str(p))
except Exception as _exc:
    print("[download] skipped:", _exc)
'''
        ),
    ]

    path = OUT / "k_suite_colab.ipynb"
    path.write_text(json.dumps(nb(cells, "k_suite_colab.ipynb"), indent=1) + "\n")
    print(f"wrote {path} ({len(cells)} cells)")
    return path


if __name__ == "__main__":
    build()
