#!/usr/bin/env python3
"""J7: re-emit Table 3, Table 7, and PAPER_NUMBER_DELTAS from rescored P1."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.stats import wilson_ci  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402
# Frozen 61-ID challenging pool (same list as rebuild/compute_rebuild.py PAPER_ADV).
PAPER_ADV = {
    "SP": [
        "SP_003", "SP_004", "SP_005", "SP_019", "SP_020", "SP_021", "SP_023",
        "SP_024", "SP_026", "SP_027", "SP_028", "SP_029", "SP_030", "SP_037",
        "SP_038", "SP_039", "SP_040", "SP_042", "SP_044", "SP_045", "SP_046",
        "SP_047", "SP_048", "SP_062", "SP_063", "SP_064", "SP_065", "SP_066",
        "SP_068", "SP_069", "SP_070", "SP_071", "SP_072", "SP_073",
    ],
    "CC": [f"CC_{i:02d}" for i in range(1, 11)],
    "WIS": [
        "WIS_003", "WIS_004", "WIS_013", "WIS_014", "WIS_015", "WIS_016",
        "WIS_017", "WIS_018", "WIS_019", "WIS_020", "WIS_023", "WIS_024",
        "WIS_025", "WIS_026", "WIS_027", "WIS_028", "WIS_029",
    ],
}
PAPER_ADV_ALL = set(PAPER_ADV["SP"] + PAPER_ADV["CC"] + PAPER_ADV["WIS"])

DER = REPO_ROOT / "results" / "derived"
PAPER_T = REPO_ROOT / "paper" / "tables"
SHORT = {
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-flash": "Gemini",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
TAGS = {
    "Claude": "claude",
    "Gemini": "gemini",
    "GPT-4o": "gpt4o",
    "Llama": "llama",
    "o4-mini": "o1mini",
}
T3_MODELS = ["Claude", "GPT-4o", "Llama", "Gemini", "o4-mini"]
T3_TEX = {
    "Claude": "Claude",
    "GPT-4o": "GPT-4o",
    "Llama": "Llama-8B",
    "Gemini": "Gemini-2.5",
    "o4-mini": r"\omini",
}
T7_GSM_MODELS = ["Claude", "GPT-4o", "Gemini", "Llama", "o4-mini"]
T7_ALGO_MODELS = ["Claude", "GPT-4o", "Gemini", "Llama"]
T7_BW_MODELS = ["Claude", "GPT-4o", "Gemini", "Llama", "o4-mini"]
T7_ALGO_SLICES = [
    "CC-chall.",
    "CC-std.",
    "SP-chall.",
    "SP-std.",
    "WIS-chall.",
    "WIS-std.",
]
VARIANTS = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _algo_slice(pid: str) -> str:
    if pid.startswith("CC"):
        sub = "CC"
    elif pid.startswith("SP"):
        sub = "SP"
    elif pid.startswith("WIS"):
        sub = "WIS"
    else:
        return ""
    kind = "chall" if pid in PAPER_ADV_ALL else "std"
    return f"{sub}-{kind}."


def _load_algo() -> pd.DataFrame:
    parts = []
    for tag in TAGS.values():
        path = DER / f"ALGO_P1_behavioral_{tag}_rescored.csv"
        df = pd.read_csv(path, dtype=str).fillna("")
        df = df[_is_true(df["included"])]
        df = df[df["model"].isin(SHORT)]
        df["variant_type"] = df["variant_type"].map(normalize_variant)
        df = filter_excluded(df, family="ALGO")
        df["ok"] = _is_true(df["rescored_correct"])
        df["model_short"] = df["model"].map(SHORT)
        df["slice"] = df["problem_id"].map(_algo_slice)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["problem_id", "variant_type", "model_short"], keep="last")


def _load_gsm() -> pd.DataFrame:
    parts = []
    for tag in TAGS.values():
        path = DER / f"GSM_P1_behavioral_{tag}_rescored.csv"
        df = pd.read_csv(path, dtype=str).fillna("")
        df = df[_is_true(df["included"])]
        df = df[df["model"].isin(SHORT)]
        df["variant_type"] = df["variant_type"].map(normalize_variant)
        df = filter_excluded(df, family="GSM")
        df["ok"] = _is_true(df["rescored_correct"])
        df["model_short"] = df["model"].map(SHORT)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["problem_id", "variant_type", "model_short"], keep="last")


def _load_bw() -> pd.DataFrame:
    bank = pd.read_csv(REPO_ROOT / "data/problems/question_bank_bw.csv", dtype=str)
    ids = set(
        bank.loc[
            bank["variant_type"].str.strip().str.lower() == "canonical", "problem_id"
        ].astype(str)
    )
    parts = []
    for name in [
        "BW_P1_behavioral_rescored.csv",
        "BW_P1_behavioral_gemini_rescored.csv",
        "BW_P1_behavioral_o1mini_rescored.csv",
    ]:
        df = pd.read_csv(DER / name, dtype=str).fillna("")
        df = df[_is_true(df["included"])]
        df = df[df["model"].isin(SHORT)]
        df = df[df["problem_id"].isin(ids)]
        df["variant_type"] = df["variant_type"].map(normalize_variant)
        df = filter_excluded(df, family="BW")
        df["ok"] = _is_true(df["rescored_correct"])
        df["model_short"] = df["model"].map(SHORT)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["problem_id", "variant_type", "model_short"], keep="last")


def _acc(sub: pd.DataFrame) -> tuple[float | None, int, int]:
    n = int(len(sub))
    if n == 0:
        return None, 0, 0
    k = int(sub["ok"].sum())
    return k / n, k, n


def _round_disp(x: float | None, nd: int) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x) + 1e-12, nd)


def _fmt_acc(x: float | None) -> str:
    x = _round_disp(x, 3)
    if x is None:
        return "---"
    if x >= 1:
        return f"{x:.3f}"
    return f"{x:.3f}".replace("0.", ".", 1)


def _fmt_ci(lo: float, hi: float) -> str:
    def p(v: float) -> str:
        v = _round_disp(v, 2)
        s = f"{v:.2f}"
        return s[1:] if s.startswith("0") else s

    return f"[{p(lo)},{p(hi)}]"


def _cell(df: pd.DataFrame, model: str, variant: str, slice_name: str | None = None) -> tuple[float | None, int, int]:
    sub = df[(df["model_short"] == model) & (df["variant_type"] == variant)]
    if slice_name:
        sub = sub[sub["slice"] == slice_name]
    return _acc(sub)


def _omit_algo(slice_name: str, variant: str) -> bool:
    if variant == "W5" and not slice_name.startswith("SP"):
        return True
    return False


def write_table3(gsm: pd.DataFrame) -> list[dict]:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{GSM Probe~1 ($n{=}44$ for Claude/Gemini/\omini; $n{=}20$ partial",
        r"  for GPT-4o/Llama; 95\% Wilson CIs). The headline inversion is powered on",
        r"  ALGO SP-adv ($n{=}34$; Section~\ref{sec:inversion}); GSM rows calibrate",
        r"  near-matched canonical accuracy (0.800--0.850) with wide CIs for the",
        r"  $n{=}20$ cells.}",
        r"  \label{tab:gsm_p1}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Model & $\mathrm{Acc}_{\mathrm{can}}$ & $\mathrm{Acc}_{W_3}$ & $\Delta$ & $\Rwiii$ \\",
        r"    \midrule",
    ]
    new_cells: list[dict] = []
    for m in T3_MODELS:
        a_can, k_can, n_can = _cell(gsm, m, "canonical")
        a_w3, k_w3, n_w3 = _cell(gsm, m, "W3")
        lo_c, hi_c = wilson_ci(k_can, n_can)
        lo_w, hi_w = wilson_ci(k_w3, n_w3)
        can_d = _round_disp(a_can, 3)
        w3_d = _round_disp(a_w3, 3)
        delta = None if can_d is None or w3_d is None else _round_disp(can_d - w3_d, 3)
        rw = None if not can_d or w3_d is None else _round_disp(w3_d / can_d, 3)
        label = T3_TEX[m]
        lines.append(
            f"    {label:<10} & {_fmt_acc(a_can)} {_fmt_ci(lo_c, hi_c)} "
            f"& {_fmt_acc(a_w3)} {_fmt_ci(lo_w, hi_w)} "
            f"& {_fmt_acc(delta)} & {_fmt_acc(rw)} \\\\"
        )
        new_cells.extend(
            [
                {"location": "table3", "family": "GSM", "slice": "--", "model": m, "variant": "canonical", "metric": "accuracy", "new_value": can_d, "n": n_can, "nd": 3},
                {"location": "table3", "family": "GSM", "slice": "--", "model": m, "variant": "canonical", "metric": "ci_low", "new_value": _round_disp(lo_c, 2), "n": n_can, "nd": 2},
                {"location": "table3", "family": "GSM", "slice": "--", "model": m, "variant": "canonical", "metric": "ci_high", "new_value": _round_disp(hi_c, 2), "n": n_can, "nd": 2},
                {"location": "table3", "family": "GSM", "slice": "--", "model": m, "variant": "W3", "metric": "accuracy", "new_value": w3_d, "n": n_w3, "nd": 3},
                {"location": "table3", "family": "GSM", "slice": "--", "model": m, "variant": "W3", "metric": "ci_low", "new_value": _round_disp(lo_w, 2), "n": n_w3, "nd": 2},
                {"location": "table3", "family": "GSM", "slice": "--", "model": m, "variant": "W3", "metric": "ci_high", "new_value": _round_disp(hi_w, 2), "n": n_w3, "nd": 2},
                {"location": "table3", "family": "GSM", "slice": "--", "model": m, "variant": "W3", "metric": "delta", "new_value": delta, "n": min(n_can, n_w3), "nd": 3},
                {"location": "table3", "family": "GSM", "slice": "--", "model": m, "variant": "W3", "metric": "retention", "new_value": rw, "n": min(n_can, n_w3), "nd": 3},
            ]
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    (PAPER_T / "table3_gsm_p1.tex").write_text("\n".join(lines), encoding="utf-8")
    return new_cells


def write_table7(gsm: pd.DataFrame, algo: pd.DataFrame, bw: pd.DataFrame) -> list[dict]:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \caption{Per-variant accuracy by family, slice, and model (Probe~1). GSM",
        r"  uses rescored \texttt{included=True} rows; GPT-4o and Llama have $n{=}20$",
        r"  (bank-valid GSM\_001--020 only). ALGO uses the released verifier",
        r"  rescored offline (SP $W_3$ node mapping restored; trailing Path line",
        r"  preferred) on the frozen 61-ID challenging/standard split; \omini{} is",
        r"  omitted from ALGO rows. BW is bank-restricted to the $n{=}65$",
        r"  PlanBench IDs used throughout. ALGO and BW $W_6$ exclude byte-identical",
        r"  copies (25 ALGO WIS generator bug; 8 BW true duplicates). GSM $W_6$ is",
        r"  retained for models with bank rows (Claude/Gemini/\omini{} on GSM\_041--064).",
        r"  GPT-4o and Llama $W_6$ cells are omitted (\texttt{missing\_bank\_row}: Table~7",
        r"  .800/.450 were computed on GSM\_001--020 W6 in raw logs; no bank version",
        r"  ever defined W6 on those IDs). BW $W_5$",
        r"  drops MBW\_496--500 (identical to canonical; n=60). ALGO accuracies are",
        r"  still reported on the frozen 61-ID split of the 110 canonical IDs, but",
        r"  near-duplicate detection (normalized text + identical gold) collapses",
        r"  the 110-problem bank to effective $n{=}51$ (14 clone families covering",
        r"  73 problems); see \texttt{results/derived/bank\_clone\_audit.csv}.",
        r"  All remaining cells are the output of the released per-instance",
        r"  verifiers (Appendix~\ref{app:repro}).}",
        r"  \label{tab:pervariant}",
        r"  \begin{tabular}{lllccccccc}",
        r"    \toprule",
        r"    Family & Slice & Model & Can. & W1 & W2 & W3 & W4 & W5 & W6 \\",
        r"    \midrule",
    ]
    new_cells: list[dict] = []
    t7_tex = {
        "Claude": "Claude",
        "GPT-4o": "GPT-4o",
        "Gemini": "Gemini",
        "Llama": "Llama",
        "o4-mini": r"\omini",
    }

    def emit_row(family: str, slice_name: str, model: str, cells: list[str]) -> None:
        sl = slice_name if slice_name != "--" else "--"
        lines.append(
            f"    {family} & {sl} & {t7_tex[model]:<10} & " + " & ".join(cells) + r" \\"
        )

    def pack(family: str, slice_name: str, model: str, df: pd.DataFrame, omit=None) -> list[str]:
        out = []
        for vt in VARIANTS:
            if omit and omit(slice_name, vt):
                acc, k, n = None, 0, 0
                text = "---"
            else:
                acc, k, n = _cell(df, model, vt, None if slice_name == "--" else slice_name)
                text = _fmt_acc(acc)
            out.append(text)
            new_cells.append(
                {
                    "location": "table7",
                    "family": family,
                    "slice": slice_name,
                    "model": model,
                    "variant": vt,
                    "metric": "accuracy",
                    "new_value": None if text == "---" else _round_disp(acc, 3),
                    "n": n,
                    "display": text,
                    "nd": 3,
                }
            )
        return out

    for m in T7_GSM_MODELS:
        emit_row("GSM", "--", m, pack("GSM", "--", m, gsm, omit=lambda _s, vt: vt == "W6" and m in {"GPT-4o", "Llama"}))
    lines.append(r"    \midrule")
    for sl in T7_ALGO_SLICES:
        for m in T7_ALGO_MODELS:
            emit_row("ALGO", sl, m, pack("ALGO", sl, m, algo, omit=_omit_algo))
    lines.append(r"    \midrule")
    for m in T7_BW_MODELS:
        emit_row("BW", "--", m, pack("BW", "--", m, bw))
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    (PAPER_T / "table7_pervariant.tex").write_text("\n".join(lines), encoding="utf-8")
    return new_cells


def _parse_old_table3(text: str) -> dict[tuple, float | None]:
    out: dict[tuple, float | None] = {}
    alias = {
        "Claude": "Claude",
        "GPT-4o": "GPT-4o",
        "Llama-8B": "Llama",
        "Gemini-2.5": "Gemini",
        r"\omini": "o4-mini",
    }
    for line in text.splitlines():
        if "&" not in line or "Acc" in line or "toprule" in line:
            continue
        raw = line.strip().rstrip("\\").strip()
        m = re.match(
            r"(\\omini|[\w.\-]+)\s+&\s+(\.?\d+|---)(?:\s+(\[[^\]]+\]))?\s+&\s+(\.?\d+|---)(?:\s+(\[[^\]]+\]))?\s+&\s+(\.?\d+|---)\s+&\s+(\.?\d+|---)",
            raw,
        )
        if not m:
            continue
        model = alias.get(m.group(1), m.group(1))
        def num(s: str | None) -> float | None:
            if not s or s == "---":
                return None
            return float(s) if s.startswith("1") or s.startswith("0") else float("0" + s)

        def cis(blob: str | None) -> tuple[float | None, float | None]:
            if not blob:
                return None, None
            a, b = blob.strip("[]").split(",")
            return num(a.strip()), num(b.strip())

        out[("table3", "GSM", "--", model, "canonical", "accuracy")] = num(m.group(2))
        lo, hi = cis(m.group(3))
        out[("table3", "GSM", "--", model, "canonical", "ci_low")] = lo
        out[("table3", "GSM", "--", model, "canonical", "ci_high")] = hi
        out[("table3", "GSM", "--", model, "W3", "accuracy")] = num(m.group(4))
        lo, hi = cis(m.group(5))
        out[("table3", "GSM", "--", model, "W3", "ci_low")] = lo
        out[("table3", "GSM", "--", model, "W3", "ci_high")] = hi
        out[("table3", "GSM", "--", model, "W3", "delta")] = num(m.group(6))
        out[("table3", "GSM", "--", model, "W3", "retention")] = num(m.group(7))
    return out


def _parse_old_table7(text: str) -> dict[tuple, float | None]:
    out: dict[tuple, float | None] = {}
    alias = {r"\omini": "o4-mini"}
    for line in text.splitlines():
        if line.count("&") < 8:
            continue
        if "Family" in line or "toprule" in line or "midrule" in line:
            continue
        parts = [p.strip() for p in line.strip().rstrip("\\").split("&")]
        if len(parts) < 10:
            continue
        fam, sl, model = parts[0], parts[1], parts[2]
        model = alias.get(model, model)
        sl = sl if sl else "--"
        for vt, cell in zip(VARIANTS, parts[3:10]):
            if cell == "---":
                val: float | None = None
            elif cell.startswith("1") or cell.startswith("0"):
                val = float(cell)
            else:
                val = float("0" + cell)
            out[("table7", fam, sl, model, vt, "accuracy")] = val
    return out


def write_deltas(new_cells: list[dict], old_t3: str, old_t7: str) -> None:
    old = {}
    old.update(_parse_old_table3(old_t3))
    old.update(_parse_old_table7(old_t7))
    rows = []
    for c in new_cells:
        key = (c["location"], c["family"], c["slice"], c["model"], c["variant"], c["metric"])
        old_v = old.get(key, "MISSING_IN_OLD")
        new_v = c.get("new_value")
        if old_v == "MISSING_IN_OLD":
            changed = True
            old_s = ""
        else:
            nd = int(c.get("nd") or 3)
            if old_v is None and new_v is None:
                changed = False
            elif old_v is None or new_v is None:
                changed = True
            else:
                changed = _round_disp(float(old_v), nd) != _round_disp(float(new_v), nd)
            old_s = "" if old_v is None else old_v
        if not changed:
            continue
        rows.append(
            {
                "location": c["location"],
                "family": c["family"],
                "slice": c["slice"],
                "model": c["model"],
                "variant": c["variant"],
                "metric": c["metric"],
                "old_value": old_s if old_s != "" else ("---" if old_v is None else old_v),
                "new_value": "---" if new_v is None else new_v,
                "delta": ""
                if (old_v in (None, "MISSING_IN_OLD") or new_v is None)
                else float(new_v) - float(old_v),
                "n": c.get("n", ""),
                "note": "stale in current draft" if old_v != "MISSING_IN_OLD" else "added",
            }
        )
    out = pd.DataFrame(rows)
    path = DER / "PAPER_NUMBER_DELTAS.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out)} changed cells)")
    if not out.empty:
        print(out.to_string(index=False))


def main() -> None:
    import subprocess

    old_t3 = subprocess.check_output(
        ["git", "show", "HEAD:paper/tables/table3_gsm_p1.tex"], cwd=REPO_ROOT
    ).decode()
    old_t7 = subprocess.check_output(
        ["git", "show", "HEAD:paper/tables/table7_pervariant.tex"], cwd=REPO_ROOT
    ).decode()
    gsm = _load_gsm()
    algo = _load_algo()
    bw = _load_bw()
    cells = write_table3(gsm) + write_table7(gsm, algo, bw)
    write_deltas(cells, old_t3, old_t7)
    print("Wrote", PAPER_T / "table3_gsm_p1.tex")
    print("Wrote", PAPER_T / "table7_pervariant.tex")


if __name__ == "__main__":
    main()
