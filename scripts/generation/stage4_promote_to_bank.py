"""
scripts/generation/stage4_promote_to_bank.py

Append verified staging rows into the three question bank files,
normalizing both old and new rows to exactly 12 columns.

Usage:
    python scripts/generation/stage4_promote_to_bank.py [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from probes.common.io import QUESTION_BANK_COLUMNS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage4")

STAGING_VERIFIED = REPO_ROOT / "data" / "staging" / "verified_rows.csv"
STAGING_GSM_W5 = REPO_ROOT / "data" / "staging" / "gsm_w5_manual_review.csv"
BANK_BW = REPO_ROOT / "data" / "problems" / "question_bank_bw.csv"
BANK_GSM = REPO_ROOT / "data" / "problems" / "question_bank_gsm.csv"
BANK_ALGO = REPO_ROOT / "data" / "problems" / "question_bank_algo.csv"

STAGING_DROP_COLS = {
    "status",
    "selection_reason",
    "generator_model",
    "manual_review_reason",
}

VALID_VARIANT_TYPES = {"canonical", "W1", "W2", "W3", "W4", "W5", "W6"}

FAMILY_BANKS = {
    "planning_suite": BANK_BW,
    "arithmetic_reasoning": BANK_GSM,
    "algorithmic": BANK_ALGO,
}


def _normalize_bank_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "variant_type" in out.columns:
        out["variant_type"] = out["variant_type"].astype(str).apply(
            lambda v: re.sub(r"^w([1-6])$", r"W\1", v.strip(), flags=re.IGNORECASE)
            if re.match(r"^w[1-6]$", v.strip(), re.IGNORECASE)
            else v.strip()
        )
    if "problem_family" in out.columns:
        out["problem_family"] = out["problem_family"].astype(str).apply(
            lambda v: "algorithmic"
            if "algorithmic" in v.lower()
            else v.strip().lower()
        )
    if "contamination_pole" in out.columns:
        out["contamination_pole"] = (
            out["contamination_pole"].astype(str).str.strip().str.lower()
        )
    return out


def _to_bank_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in QUESTION_BANK_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[[c for c in QUESTION_BANK_COLUMNS if c in out.columns]]
    return out[QUESTION_BANK_COLUMNS]


def _load_new_rows() -> pd.DataFrame:
    if not STAGING_VERIFIED.exists():
        raise FileNotFoundError(f"Missing {STAGING_VERIFIED}")
    verified = pd.read_csv(STAGING_VERIFIED, dtype=str).fillna("")
    frames = [verified]
    if STAGING_GSM_W5.exists():
        gsm_w5 = pd.read_csv(STAGING_GSM_W5, dtype=str).fillna("")
        frames.append(gsm_w5)
    else:
        log.warning(f"GSM W5 file not found: {STAGING_GSM_W5}")
    combined = pd.concat(frames, ignore_index=True)
    drop = [c for c in STAGING_DROP_COLS if c in combined.columns]
    if drop:
        combined = combined.drop(columns=drop)
    combined = _to_bank_columns(combined)
    return _normalize_bank_rows(combined)


def _split_by_family(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pf = df["problem_family"].astype(str).str.strip().str.lower()
    splits = {
        "planning_suite": df[pf == "planning_suite"].copy(),
        "arithmetic_reasoning": df[pf == "arithmetic_reasoning"].copy(),
        "algorithmic": df[pf == "algorithmic"].copy(),
    }
    for fam, part in splits.items():
        log.info(f"  new rows — {fam}: {len(part)}")
        if len(part) == 0:
            log.warning(f"  WARNING: zero new rows for {fam}")
    return splits


def _check_collisions(
    bank_path: Path, new_df: pd.DataFrame, fam_label: str
) -> list[str]:
    old = pd.read_csv(bank_path, dtype=str).fillna("")
    existing_ids = set(old["problem_id"].astype(str).str.strip())
    new_ids = set(new_df["problem_id"].astype(str).str.strip())
    collisions = sorted(existing_ids & new_ids)
    if collisions:
        log.error(
            f"COLLISION ({fam_label}): {len(collisions)} problem_id(s) already in bank: "
            f"{collisions[:20]}{'...' if len(collisions) > 20 else ''}"
        )
    return collisions


def _audit_bank(path: Path, expected_total: int, fam_label: str) -> None:
    df = pd.read_csv(path, dtype=str).fillna("")
    if list(df.columns) != list(QUESTION_BANK_COLUMNS):
        raise ValueError(
            f"{fam_label}: column order mismatch.\n"
            f"  got:      {list(df.columns)}\n"
            f"  expected: {QUESTION_BANK_COLUMNS}"
        )
    if (df["problem_id"].astype(str).str.strip() == "").any():
        raise ValueError(f"{fam_label}: empty problem_id rows found")
    vts = set(df["variant_type"].astype(str).str.strip())
    bad_vt = vts - VALID_VARIANT_TYPES
    if bad_vt:
        raise ValueError(f"{fam_label}: invalid variant_type values: {bad_vt}")
    dup = df.duplicated(subset=["problem_id", "variant_type"], keep=False)
    if dup.any():
        dups = df.loc[dup, ["problem_id", "variant_type"]].drop_duplicates()
        raise ValueError(f"{fam_label}: duplicate (problem_id, variant_type):\n{dups}")
    if len(df) != expected_total:
        raise ValueError(
            f"{fam_label}: row count {len(df)} != expected {expected_total}"
        )
    log.info(f"AUDIT PASS: {len(df)} total rows ({fam_label})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 — promote staging to question banks")
    parser.add_argument("--dry-run", action="store_true", help="Collision check only; no writes")
    args = parser.parse_args()

    log.info("STEP 1 — Load new rows")
    new_all = _load_new_rows()
    log.info(f"  combined new rows: {len(new_all)}")

    log.info("STEP 2 — Normalize new rows (done in loader)")

    log.info("STEP 3 — Split new rows by family")
    splits = _split_by_family(new_all)

    log.info("STEP 4 — Collision check")
    all_collisions: dict[str, list[str]] = {}
    for fam, bank_path in FAMILY_BANKS.items():
        all_collisions[fam] = _check_collisions(bank_path, splits[fam], fam)
    total_collisions = sum(len(v) for v in all_collisions.values())
    if total_collisions > 0:
        log.error(f"Aborting: {total_collisions} total collision(s) across families")
        sys.exit(1)
    log.info("  zero collisions — safe to append")

    if args.dry_run:
        log.info("DRY RUN — would append:")
        for fam, bank_path in FAMILY_BANKS.items():
            old_n = len(pd.read_csv(bank_path, dtype=str))
            new_n = len(splits[fam])
            sample = splits[fam]["problem_id"].head(5).tolist() if new_n else []
            log.info(
                f"  {fam}: +{new_n} rows ({old_n} → {old_n + new_n}), "
                f"sample ids: {sample}"
            )
        log.info(f"  total new rows: {len(new_all)}")
        return

    log.info("STEP 5-6 — Normalize old rows, append, write")
    summary: dict[str, tuple[int, int, int]] = {}
    for fam, bank_path in FAMILY_BANKS.items():
        old_df = pd.read_csv(bank_path, dtype=str).fillna("")
        old_n = len(old_df)
        old_norm = _normalize_bank_rows(_to_bank_columns(old_df))
        new_part = splits[fam]
        new_n = len(new_part)
        merged = pd.concat([old_norm, new_part], ignore_index=True)
        merged = merged[QUESTION_BANK_COLUMNS]
        merged.to_csv(bank_path, index=False, encoding="utf-8")
        summary[fam] = (old_n, new_n, old_n + new_n)
        log.info(f"  wrote {bank_path.name}: {old_n} + {new_n} = {old_n + new_n}")

    log.info("STEP 7 — Post-write audit")
    for fam, bank_path in FAMILY_BANKS.items():
        old_n, new_n, total = summary[fam]
        _audit_bank(bank_path, total, fam)

    log.info("STEP 8 — Summary")
    labels = {
        "planning_suite": "BW",
        "arithmetic_reasoning": "GSM",
        "algorithmic": "ALGO",
    }
    total_new = 0
    for fam, (old_n, new_n, total) in summary.items():
        label = labels[fam]
        print(f"  {label:4} appended {new_n} rows (old {old_n} → new total {total})")
        total_new += new_n
    print(f"  Total new rows appended: {total_new}")
    print("  Banks are ready for sweep.")


if __name__ == "__main__":
    main()
