#!/usr/bin/env python3
"""J6: Probe 1 failure patterns from included=True rescored rows. Counts only."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.variants import normalize_variant  # noqa: E402

DERIVED = REPO_ROOT / "results" / "derived"
BANKS = {
    "ALGO": REPO_ROOT / "data/problems/question_bank_algo.csv",
    "BW": REPO_ROOT / "data/problems/question_bank_bw.csv",
    "GSM": REPO_ROOT / "data/problems/question_bank_gsm.csv",
}
PAPER_MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-flash": "Gemini",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
N_PAPER_MODELS = len(PAPER_MODELS)
OUT = DERIVED / "P1_failure_patterns.csv"


def _family_from_pid(pid: str) -> str:
    p = str(pid)
    if p.startswith("GSM"):
        return "GSM"
    if p.startswith("BW") or p.startswith("MBW"):
        return "BW"
    if p.startswith(("CC", "SP", "WIS")):
        return "ALGO"
    return ""


def _parse_notes_domain(notes: str) -> str:
    text = str(notes or "")
    if "chosen_domain" not in text:
        return ""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            blob, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(blob, dict) and blob.get("chosen_domain"):
            return str(blob["chosen_domain"]).strip().lower()
    return ""


def _phi(a: int, b: int, c: int, d: int) -> float:
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0:
        return float("nan")
    return (a * d - b * c) / denom


def _row(**kwargs) -> dict:
    base = {
        "section": "",
        "family": "",
        "key": "",
        "count": "",
        "n_problems": "",
        "n_models": "",
        "ids": "",
        "stat": "",
        "note": "",
    }
    base.update(kwargs)
    return base


def _load_included() -> pd.DataFrame:
    parts = []
    for path in sorted(DERIVED.glob("*_P1_*rescored.csv")):
        if "review" in path.name.lower():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "included" not in df.columns:
            continue
        df = df[df["included"].str.strip().str.lower() == "true"].copy()
        df = df[df["model"].isin(PAPER_MODELS)].copy()
        if path.name.startswith("ALGO_"):
            src_fam = "ALGO"
        elif path.name.startswith("GSM_"):
            src_fam = "GSM"
        elif path.name.startswith("BW_"):
            src_fam = "BW"
        else:
            continue
        df["family"] = df["problem_id"].map(_family_from_pid)
        df = df[df["family"] == src_fam].copy()
        if df.empty:
            continue
        df["model_short"] = df["model"].map(PAPER_MODELS)
        df["variant"] = df["variant_type"].map(normalize_variant)
        df["ok"] = df["rescored_correct"].str.strip().str.lower().eq("true")
        df["fail"] = ~df["ok"]
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["family", "problem_id", "variant", "model_short"], keep="last")


def _banks() -> pd.DataFrame:
    frames = []
    for fam, path in BANKS.items():
        b = pd.read_csv(path, dtype=str).fillna("")
        b["family"] = fam
        b["variant"] = b["variant_type"].map(normalize_variant)
        b["text_len"] = b["problem_text"].astype(str).str.len()
        b["w3_domain"] = b["notes"].map(_parse_notes_domain)
        frames.append(b)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    p1 = _load_included()
    bank = _banks()
    bank_key = bank.set_index(["family", "problem_id", "variant"])
    planbench = set(
        bank.loc[(bank.family == "BW") & (bank.variant == "canonical"), "problem_id"]
    )
    rows: list[dict] = []

    can = p1[p1["variant"] == "canonical"].copy()

    for fam, g in can.groupby("family"):
        wide = g.pivot_table(
            index="problem_id", columns="model_short", values="fail", aggfunc="max"
        )
        n_present = wide.notna().sum(axis=1)
        n_fail = wide.fillna(False).sum(axis=1)
        complete = n_present == N_PAPER_MODELS
        shared_all5 = complete & (n_fail == N_PAPER_MODELS)
        shared_present = (n_present >= 2) & (n_fail == n_present)
        specific = (n_present >= 2) & (n_fail == 1)

        shared_ids = sorted(wide.index[shared_all5].tolist())
        present_ids = sorted(wide.index[shared_present].tolist())
        spec_ids = sorted(wide.index[specific].tolist())

        rows.append(
            _row(
                section="shared_hard_canonical",
                family=fam,
                key="fail_all_five_paper_models",
                count=int(shared_all5.sum()),
                n_problems=int(complete.sum()),
                n_models=N_PAPER_MODELS,
                ids=",".join(shared_ids),
                note="canonical included=True; all 5 paper models have a row and all fail",
            )
        )
        rows.append(
            _row(
                section="shared_hard_canonical",
                family=fam,
                key="fail_all_models_with_a_row",
                count=int(shared_present.sum()),
                n_problems=int((n_present >= 2).sum()),
                n_models=N_PAPER_MODELS,
                ids=",".join(present_ids),
                note="canonical included=True; every paper model that has a row fails (min 2 models)",
            )
        )
        rows.append(
            _row(
                section="model_specific_canonical",
                family=fam,
                key="fail_exactly_one_model",
                count=int(specific.sum()),
                n_problems=int((n_present >= 2).sum()),
                n_models=N_PAPER_MODELS,
                ids=",".join(spec_ids),
                note="canonical included=True; exactly one of the models with a row fails",
            )
        )
        for m in PAPER_MODELS.values():
            if m not in wide.columns:
                n_only = 0
                only_ids: list[str] = []
            else:
                mask = specific & wide[m].fillna(False)
                n_only = int(mask.sum())
                only_ids = sorted(wide.index[mask].tolist())
            rows.append(
                _row(
                    section="model_specific_canonical",
                    family=fam,
                    key=f"fail_only_{m}",
                    count=n_only,
                    n_problems=int((n_present >= 2).sum()),
                    n_models=N_PAPER_MODELS,
                    ids=",".join(only_ids),
                    note="",
                )
            )

    w3 = p1[p1["variant"] == "W3"]
    merged = can[["family", "problem_id", "model_short", "fail"]].merge(
        w3[["family", "problem_id", "model_short", "fail"]],
        on=["family", "problem_id", "model_short"],
        suffixes=("_can", "_w3"),
    )
    for (model, fam), g in merged.groupby(["model_short", "family"]):
        a = int((~g.fail_can & ~g.fail_w3).sum())
        b = int((~g.fail_can & g.fail_w3).sum())
        c = int((g.fail_can & ~g.fail_w3).sum())
        d = int((g.fail_can & g.fail_w3).sum())
        phi = _phi(a, b, c, d)
        rows.append(
            _row(
                section="can_vs_w3",
                family=fam,
                key=model,
                count=len(g),
                n_problems=len(g),
                n_models=1,
                ids=f"ok_ok={a};ok_fail={b};fail_ok={c};fail_fail={d}",
                stat="" if phi != phi else f"{phi:.6f}",
                note="2x2 can_fail x w3_fail; stat=phi; cells=ok_ok,ok_fail,fail_ok,fail_fail",
            )
        )

    meta = []
    for _, r in can.iterrows():
        key = (r["family"], r["problem_id"], "canonical")
        if key not in bank_key.index:
            continue
        br = bank_key.loc[key]
        if isinstance(br, pd.DataFrame):
            br = br.iloc[0]
        meta.append(
            {
                "family": r["family"],
                "model_short": r["model_short"],
                "fail": int(r["fail"]),
                "text_len": int(br["text_len"]),
                "difficulty": str(br.get("difficulty") or ""),
                "subtype": str(br.get("problem_subtype") or ""),
            }
        )
    md = pd.DataFrame(meta)
    for fam, g in md.groupby("family"):
        if g["text_len"].nunique() > 1 and g["fail"].nunique() == 2:
            r_pb, p_pb = stats.pointbiserialr(g["fail"], g["text_len"])
        else:
            r_pb, p_pb = float("nan"), float("nan")
        rows.append(
            _row(
                section="correlate_length",
                family=fam,
                key="pointbiserial_fail_vs_text_len",
                count=int(g["fail"].sum()),
                n_problems=int(len(g)),
                n_models=int(g["model_short"].nunique()),
                stat="" if r_pb != r_pb else f"r={r_pb:.4f};p={p_pb:.4g}",
                note="canonical included rows pooled over models",
            )
        )
        for diff, gd in g.groupby("difficulty"):
            if not str(diff).strip():
                continue
            rows.append(
                _row(
                    section="fail_by_difficulty",
                    family=fam,
                    key=str(diff),
                    count=int(gd["fail"].sum()),
                    n_problems=int(len(gd)),
                    n_models=int(gd["model_short"].nunique()),
                    stat=f"{gd['fail'].mean():.4f}",
                    note="fail rate in stat",
                )
            )
        for sub, gs in g.groupby("subtype"):
            if not str(sub).strip():
                continue
            rows.append(
                _row(
                    section="fail_by_subtype",
                    family=fam,
                    key=str(sub),
                    count=int(gs["fail"].sum()),
                    n_problems=int(len(gs)),
                    n_models=int(gs["model_short"].nunique()),
                    stat=f"{gs['fail'].mean():.4f}",
                    note="fail rate in stat",
                )
            )

    def _bw_w3_only(frame: pd.DataFrame, section_suffix: str, note: str) -> None:
        bw = frame[(frame.family == "BW") & frame.fail_can & ~frame.fail_w3]
        bw_w3 = bank[(bank.family == "BW") & (bank.variant == "W3")].drop_duplicates(
            "problem_id"
        ).set_index("problem_id")
        domain_pids: dict[str, set[str]] = defaultdict(set)
        domain_pairs: dict[str, int] = defaultdict(int)
        classic = mystery = 0
        unique_pids = sorted(bw["problem_id"].unique())
        for pid in unique_pids:
            if str(pid).startswith("MBW_"):
                mystery += 1
            else:
                classic += 1
            if pid not in bw_w3.index:
                domain_pids["unknown"].add(pid)
                continue
            dom = str(bw_w3.loc[pid]["w3_domain"] or "") or "unknown"
            domain_pids[dom].add(pid)
        for _, r in bw.iterrows():
            pid = r["problem_id"]
            if pid not in bw_w3.index:
                domain_pairs["unknown"] += 1
            else:
                dom = str(bw_w3.loc[pid]["w3_domain"] or "") or "unknown"
                domain_pairs[dom] += 1
        rows.append(
            _row(
                section=f"bw_w3_only_success{section_suffix}",
                family="BW",
                key="n_problem_ids",
                count=int(bw["problem_id"].nunique()),
                n_problems=int(bw["problem_id"].nunique()),
                n_models=int(bw["model_short"].nunique()) if len(bw) else 0,
                ids=",".join(unique_pids),
                note=f"classic={classic};mystery={mystery}; {note}",
            )
        )
        rows.append(
            _row(
                section=f"bw_w3_only_success{section_suffix}",
                family="BW",
                key="n_model_problem_pairs",
                count=int(len(bw)),
                n_problems=int(bw["problem_id"].nunique()),
                n_models=int(bw["model_short"].nunique()) if len(bw) else 0,
                note=note,
            )
        )
        for dom in sorted(set(domain_pids) | set(domain_pairs)):
            rows.append(
                _row(
                    section=f"bw_w3_only_success_domain{section_suffix}",
                    family="BW",
                    key=dom,
                    count=len(domain_pids[dom]),
                    n_problems=int(bw["problem_id"].nunique()),
                    n_models=int(bw["model_short"].nunique()) if len(bw) else 0,
                    ids=f"pairs={domain_pairs[dom]}",
                    stat=str(domain_pairs[dom]),
                    note="count=unique problem_ids; stat=model-problem pairs; chosen_domain from W3 notes",
                )
            )

    _bw_w3_only(merged, "", "all included BW/MBW pids")
    _bw_w3_only(
        merged[merged["problem_id"].isin(planbench)],
        "_planbench65",
        "restricted to 65 PlanBench canonical bank IDs",
    )

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(rows)} rows)")
    show = pd.DataFrame(rows)
    keep = show.section.isin(
        [
            "shared_hard_canonical",
            "model_specific_canonical",
            "bw_w3_only_success",
            "bw_w3_only_success_planbench65",
            "bw_w3_only_success_domain",
            "bw_w3_only_success_domain_planbench65",
            "can_vs_w3",
        ]
    )
    print(show[keep].to_string(index=False))


if __name__ == "__main__":
    main()
