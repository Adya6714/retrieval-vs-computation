#!/usr/bin/env python3
"""T1 noise-floor derived metrics (HP-16 step 5).

HP Track T entry point. Supports --resume and --dry-run.
Do not invent numbers; raw CSVs are append-only truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace(" ", "_")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_rows(path: Path, fieldnames: list[str], rows: list[dict], resume_key=None) -> int:
    """Append rows; if resume_key, skip keys already present."""
    ensure_parent(path)
    done = set()
    if path.exists() and path.stat().st_size > 0 and resume_key is not None:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                done.add(resume_key(row))
    write_header = not path.exists() or path.stat().st_size == 0
    n = 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in rows:
            if resume_key is not None and resume_key(row) in done:
                continue
            w.writerow({k: row.get(k, "") for k in fieldnames})
            n += 1
    return n


def main() -> None:
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-glob", default="results/raw/T1_noise_floor_*.csv")
    ap.add_argument("--out", default="results/derived/T1_noise_floor_metrics.csv")
    ap.add_argument("--report", default="docs/trackT/T1_REPORT.md")
    args = ap.parse_args()

    paths = sorted(REPO_ROOT.glob(args.raw_glob))
    out = REPO_ROOT / args.out
    report = REPO_ROOT / args.report
    ensure_parent(out)
    ensure_parent(report)

    cols = [
        "model", "variant", "n_items", "output_divergence_rate",
        "verdict_flip_rate", "flip_rate_given_canonical_correct", "flipped_ids",
    ]
    if not paths:
        # empty derived with header only
        with out.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()
        report.write_text(
            "# T1_REPORT\n\nNo raw T1 files found. Run `T1_noise_floor.py` first.\n"
        )
        print(f"[T1 metrics] no raws; wrote empty {out}")
        return

    frames = [pd.read_csv(p) for p in paths]
    raw = pd.concat(frames, ignore_index=True)
    ref = raw[
        (raw["batch_size"].astype(str) == "1")
        & (raw["attn_impl"] == "eager")
        & (raw["dtype"].astype(str).str.contains("float32"))
    ].copy()
    rows = []
    for (model, variant), g in raw.groupby(["model", "variant_type"]):
        ref_g = ref[(ref["model"] == model) & (ref["variant_type"] == variant)]
        if ref_g.empty:
            continue
        ref_map = dict(zip(ref_g["problem_id"], ref_g["response_sha16"]))
        ver_map = dict(zip(ref_g["problem_id"], ref_g["verified"].astype(str)))
        # divergence vs reference hash across non-ref configs
        other = g[~((g["batch_size"].astype(str) == "1") & (g["attn_impl"] == "eager") & (g["dtype"].astype(str).str.contains("float32")))]
        n_items = g["problem_id"].nunique()
        div = 0
        flips = 0
        flips_can = 0
        flipped = []
        n_pairs = 0
        n_can = 0
        for _, r in other.iterrows():
            pid = r["problem_id"]
            if pid not in ref_map:
                continue
            n_pairs += 1
            if str(r["response_sha16"]) != str(ref_map[pid]):
                div += 1
            if str(r["verified"]) != str(ver_map.get(pid, "")):
                flips += 1
                flipped.append(pid)
                if str(ver_map.get(pid, "")).lower() in {"1", "true", "yes"}:
                    flips_can += 1
                    n_can += 1
        rows.append({
            "model": model,
            "variant": variant,
            "n_items": n_items,
            "output_divergence_rate": (div / n_pairs) if n_pairs else "",
            "verdict_flip_rate": (flips / n_pairs) if n_pairs else "",
            "flip_rate_given_canonical_correct": (flips_can / n_can) if n_can else "",
            "flipped_ids": ";".join(sorted(set(flipped))),
        })
    pd.DataFrame(rows, columns=cols).to_csv(out, index=False)
    report.write_text(
        "# T1_REPORT\n\n"
        f"Raws: {', '.join(p.name for p in paths)}\n\n"
        f"See `{args.out}` for tables.\n"
    )
    print(f"[T1 metrics] wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
