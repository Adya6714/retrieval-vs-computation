#!/usr/bin/env python3
"""Pass/fail gate: local greedy ALGO accuracy vs content-gold final ranks.

Gate (Llama-Instruct, ALGO canonical):
  FAIL  — greedy decode ~6% BUT median final-layer content-gold rank ≈ 1
  PASS  — greedy decode ~6% AND median content-gold final rank high (thousands)

Usage:
  python3 scripts/runs/mechanistic_contentgold_gate.py \\
      --mech results/raw/mechanistic_sweep_llama31_8b_instruct_chatdirect_contentgold.csv \\
      --greedy results/raw/ALGO_llama31_8b_greedy_canonical.csv
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def _parse_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        return json.loads(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mech",
        type=Path,
        default=Path(
            "results/raw/mechanistic_sweep_llama31_8b_instruct_chatdirect_contentgold.csv"
        ),
    )
    ap.add_argument(
        "--greedy",
        type=Path,
        default=Path("results/raw/ALGO_llama31_8b_greedy_canonical.csv"),
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("results/derived/mechanistic_contentgold_gate_report.md"),
    )
    ap.add_argument(
        "--rank-near-one-max",
        type=float,
        default=5.0,
        help="Median final rank ≤ this ⇒ treated as 'near 1' (FAIL if greedy low).",
    )
    ap.add_argument(
        "--rank-high-min",
        type=float,
        default=100.0,
        help="Median final rank ≥ this ⇒ treated as 'high' (PASS if greedy low).",
    )
    args = ap.parse_args()

    lines: list[str] = []
    lines.append("# Mechanistic content-gold pass/fail gate\n")

    if not args.greedy.exists():
        raise SystemExit(f"missing greedy CSV: {args.greedy}")
    if not args.mech.exists():
        raise SystemExit(f"missing mechanistic CSV: {args.mech}")

    g = pd.read_csv(args.greedy, dtype=str)
    g["ok"] = g["verified"].astype(str).str.lower().isin(["true", "1"])
    n_ok = int(g["ok"].sum())
    n = len(g)
    acc = n_ok / n if n else float("nan")
    lines.append("## Local forced-greedy ALGO canonical\n")
    lines.append(f"- file: `{args.greedy}`")
    lines.append(f"- accuracy: **{n_ok}/{n} = {acc:.4f}** ({100*acc:.1f}%)")
    if "problem_subtype" in g.columns:
        sub = g.groupby("problem_subtype")["ok"].agg(["sum", "count", "mean"])
        lines.append("- by subtype:")
        for st, row in sub.iterrows():
            lines.append(
                f"  - {st}: {int(row['sum'])}/{int(row['count'])} = {row['mean']:.4f}"
            )
    lines.append("")

    m = pd.read_csv(args.mech)
    algo = m[
        m["problem_family"].astype(str).str.lower().isin(["algorithmic", "coin change"])
        & (m["variant_type"].astype(str).str.strip() == "canonical")
    ].copy()
    finals = []
    for _, r in algo.iterrows():
        ranks = _parse_list(r.get("target_rank_per_layer"))
        if ranks:
            finals.append(int(ranks[-1]))
    med = float(pd.Series(finals).median()) if finals else float("nan")
    mean = float(pd.Series(finals).mean()) if finals else float("nan")
    # token decode check — should NOT be Path/Count/Selected
    tok = algo["target_token_decoded"].astype(str).value_counts()
    n_fmt = int(tok.reindex(["Path", "Count", "Selected"]).fillna(0).sum())
    lines.append("## Content-gold mechanistic ranks (ALGO canonical)\n")
    lines.append(f"- file: `{args.mech}`")
    lines.append(f"- n ALGO canonical with ranks: **{len(finals)}**")
    lines.append(f"- median final-layer rank: **{med:.1f}**")
    lines.append(f"- mean final-layer rank: **{mean:.1f}**")
    lines.append(f"- still format-keyword targets (Path|Count|Selected): **{n_fmt}/{len(algo)}**")
    lines.append(f"- top decoded targets: {tok.head(8).to_dict()}")
    lines.append("")

    lines.append("## Gate decision\n")
    greedy_low = acc <= 0.12  # ~6% band with slack
    near_one = med <= args.rank_near_one_max
    high = med >= args.rank_high_min
    if n_fmt > 0:
        verdict = "FAIL"
        reason = (
            f"content-gold mode still targeting format keywords "
            f"({n_fmt} rows) — ranks not trustworthy"
        )
    elif greedy_low and near_one:
        verdict = "FAIL"
        reason = (
            f"greedy≈{100*acc:.1f}% but median content-gold rank={med:.1f}≈1 "
            "— model would unlock answers at readout while generation fails; pipeline broken"
        )
    elif greedy_low and high:
        verdict = "PASS"
        reason = (
            f"greedy≈{100*acc:.1f}% and median content-gold rank={med:.1f} (high) "
            "— consistent: readout does not claim knowledge the decoder lacks"
        )
    elif greedy_low and not near_one and not high:
        verdict = "AMBIGUOUS"
        reason = (
            f"greedy≈{100*acc:.1f}% and median rank={med:.1f} in middle band "
            f"({args.rank_near_one_max} < med < {args.rank_high_min}); inspect distribution"
        )
    else:
        verdict = "REVIEW"
        reason = (
            f"greedy accuracy {100*acc:.1f}% not in the expected ~6% failure band; "
            f"median rank={med:.1f}. Re-check decode settings / bank filter."
        )

    lines.append(f"**{verdict}**: {reason}\n")

    # Provenance note for published cells
    lines.append("## Paper cell provenance (Llama ALGO SP canonical)\n")
    lines.append(
        "- Table 7 (pkg8) **SP-chall. / Llama / Can. = .059** = **2/34** from "
        "`results/derived/ALGO_P1_4model_frozen_labels.csv` "
        "(shortest_path × adversarial × canonical)."
    )
    lines.append(
        "- Table 7 **SP-std. / Llama / Can. = .048** = **1/21** from the same frozen file "
        "(shortest_path × standard × canonical)."
    )
    lines.append(
        "- Raw OpenRouter run `results/raw/ALGO_P1_behavioral_llama.csv` overall "
        "canonical is **7/111 ≈ 6.3%** (verified). "
        "`probes/behavioral/openai_client.py` does **not** send `temperature` / "
        "`do_sample=False` — provider default decoding, **not** forced-greedy. "
        "That path is now dead (wallet/key); do not treat 7/111 as a greedy floor."
    )
    lines.append(
        "- Local forced-greedy for this gate: `scripts/algo_llama_greedy_accuracy.py` "
        "(`do_sample=False`) → `--greedy` CSV above.\n"
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {args.report}")
    if verdict == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
