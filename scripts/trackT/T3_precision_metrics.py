#!/usr/bin/env python3
"""T3 derived precision + band tables (HP-18 step 5).

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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-metrics", default="results/derived/T3_precision_metrics.csv")
    ap.add_argument("--out-band", default="results/derived/T3_band_map.csv")
    ap.add_argument("--figure", default="results/figures/T3_band_map.pdf")
    ap.add_argument("--report", default="docs/trackT/T3_REPORT.md")
    args = ap.parse_args()

    import pandas as pd
    raws = sorted(REPO_ROOT.glob("results/raw/T3_P1_*.csv"))
    out_m = REPO_ROOT / args.out_metrics
    out_b = REPO_ROOT / args.out_band
    fig = REPO_ROOT / args.figure
    report = REPO_ROOT / args.report
    for p in (out_m, out_b, fig, report):
        ensure_parent(p)

    mcols = ["model", "precision", "variant", "accuracy", "R_W3", "R_W4", "phi_vs_fp16", "n"]
    bcols = ["model", "band", "variant", "delta_acc_vs_fp16", "n"]
    if not raws:
        pd.DataFrame(columns=mcols).to_csv(out_m, index=False)
        pd.DataFrame(columns=bcols).to_csv(out_b, index=False)
        report.write_text("# T3_REPORT\n\nNo T3 raws yet.\n")
        print("[T3 metrics] empty tables")
        return

    # Tables-only scaffold; fill after non-dry runs. Do not invent accuracies.
    pd.DataFrame(columns=mcols).to_csv(out_m, index=False)
    pd.DataFrame(columns=bcols).to_csv(out_b, index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_h, ax = plt.subplots(figsize=(4, 3))
        ax.set_title("T3 band map (pending data)")
        ax.set_xlabel("variant")
        ax.set_ylabel("band")
        fig_h.savefig(fig)
        plt.close(fig_h)
    except Exception as e:
        print("[T3 metrics] figure skipped:", e)
    report.write_text(
        "# T3_REPORT\n\n"
        f"Raws: {', '.join(p.name for p in raws)}\n\n"
        "Metrics pending Acc_can floors on non-dry-run arms.\n"
    )
    print(f"[T3 metrics] wrote {out_m}, {out_b}")


if __name__ == "__main__":
    main()
