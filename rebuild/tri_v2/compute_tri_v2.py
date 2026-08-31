#!/usr/bin/env python3
"""Head-to-head triangulation rules on the 440 panel and 169 complete-case.

Writes only under rebuild/tri_v2/. Does not touch results/ or paper/.
"""
from __future__ import annotations

import sys
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REBUILD = ROOT / "rebuild"
OUT = REBUILD / "tri_v2"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REBUILD))

from triangulation_rule import (  # noqa: E402
    APPENDIX_CCI_COMPUTATION_MIN,
    APPENDIX_CCI_RETRIEVAL_MAX,
    APPENDIX_CONTAM_PERCENTILE,
    CANONICAL_RETRIEVAL_MIN,
    CCI_COMPUTATION_MIN,
    CCI_THRESHOLDS,
    CONTAM_PERCENTILES,
    CONTAM_SPLIT,
    W3_COMPUTATION_MIN,
    W3_CUTOFFS,
    W3_RETRIEVAL_MAX,
    count_labels,
    label_appendix_three_signal,
    label_default,
    label_sweep_cell,
    label_with_thresholds,
)


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    four = pd.read_csv(REBUILD / "triangulation_4model_labels.csv")
    for c in ("VAR_canonical", "VAR_W3", "instance_contamination_score", "ACI", "instance_rank_pct"):
        four[c] = pd.to_numeric(four[c], errors="coerce")
    for c in (
        "greedy_succeeds",
        "missing_core",
        "missing_phase2",
        "parse_failure_or_missing",
        "in_paper_4model",
    ):
        four[c] = _as_bool(four[c])
    assert len(four) == 440
    w3_ok = four["VAR_W3"].notna()
    cci_ok = four["ACI"].notna()
    prox_ok = four["instance_contamination_score"].notna()
    parse_ok = ~four["parse_failure_or_missing"]
    complete = w3_ok & cci_ok & prox_ok & parse_ok
    cc = four.loc[complete].copy()
    assert len(cc) == 169, f"expected 169 complete-case, got {len(cc)}"
    four["complete_case"] = complete
    return four, cc


def appendix_votes(
    df: pd.DataFrame,
    *,
    p75: float | None = None,
    floor: float | None = None,
) -> pd.DataFrame:
    """Signed votes matching label_appendix_three_signal, optionally with frozen p75/floor."""
    w3 = pd.to_numeric(df["VAR_W3"], errors="coerce")
    cci = pd.to_numeric(df["ACI"], errors="coerce")
    contam = pd.to_numeric(df["instance_contamination_score"], errors="coerce")
    if p75 is None:
        p75 = float(contam.quantile(APPENDIX_CONTAM_PERCENTILE / 100.0))
    if floor is None:
        floor = float(contam.min()) if contam.notna().any() else 0.0

    sig_w3 = np.where(w3 == 1, 1, np.where(w3 == 0, -1, 0)).astype(int)
    sig_cci = np.zeros(len(df), dtype=int)
    cci_ok = cci.notna().to_numpy()
    sig_cci[cci_ok & (cci.to_numpy() <= APPENDIX_CCI_RETRIEVAL_MAX)] = -1
    sig_cci[cci_ok & (cci.to_numpy() >= APPENDIX_CCI_COMPUTATION_MIN)] = 1
    sig_c = np.zeros(len(df), dtype=int)
    cok = contam.notna().to_numpy()
    cv = contam.to_numpy()
    sig_c[cok & (cv >= p75)] = -1
    sig_c[cok & (np.abs(cv - floor) <= 1e-12)] = 1
    # if a value is both floor and >= p75 (degenerate), last assignment wins;
    # match the function: p75 checked first, then floor overwrites when equal.
    # The original checks p75 first then floor as elif-equivalent via sequential if.
    # Original: if cv >= p75: -1 elif abs(cv-floor)<=eps: +1
    # Sequential assignment above applies floor second, which OVERWRITES p75
    # when floor==p75. Reproduce original order:
    sig_c = np.zeros(len(df), dtype=int)
    high = cok & (cv >= p75)
    at_floor = cok & (np.abs(cv - floor) <= 1e-12)
    sig_c[high] = -1
    sig_c[at_floor & ~high] = 1

    out = df.copy()
    out["sig_w3"] = sig_w3
    out["sig_cci"] = sig_cci
    out["sig_contam"] = sig_c
    out["cci_present"] = cci_ok
    out["vote_tuple"] = [
        f"({int(a)},{int(b)},{int(c)})" for a, b, c in zip(sig_w3, sig_cci, sig_c)
    ]
    out["appendix_p75"] = p75
    out["appendix_floor"] = floor
    return out


def _counts_md(c: dict) -> str:
    return (
        f"{c['retrieval']} / {c['computation']} / {c['mixed']} / {c['ambiguous']} "
        f"(n={c['n']})"
    )


def _pattern_kind(w3: int, cci: int, cont: int) -> str:
    sigs = (w3, cci, cont)
    if all(s == -1 for s in sigs):
        return "unanimous_retrieval"
    if all(s == 1 for s in sigs):
        return "unanimous_computation"
    if -1 in sigs and 1 in sigs:
        if 0 in sigs:
            return "conflict_partial"
        return "conflict_full"
    if all(s == 0 for s in sigs):
        return "all_neutral"
    return "one_sided_or_neutral"


def run_a_b(four: pd.DataFrame, cc: pd.DataFrame) -> dict:
    votes_440 = appendix_votes(four)
    p75_440 = float(votes_440["appendix_p75"].iloc[0])
    floor_440 = float(votes_440["appendix_floor"].iloc[0])
    votes_cc_native = appendix_votes(cc)
    p75_cc = float(votes_cc_native["appendix_p75"].iloc[0])
    floor_cc = float(votes_cc_native["appendix_floor"].iloc[0])
    votes_cc_frozen = appendix_votes(cc, p75=p75_440, floor=floor_440)

    lab_app_440 = label_appendix_three_signal(four)
    lab_app_cc = label_appendix_three_signal(cc)
    # sanity: votes-derived labels vs function
    def _from_votes(v: pd.DataFrame) -> pd.Series:
        labs = []
        for _, r in v.iterrows():
            sigs = [int(r.sig_w3), int(r.sig_cci), int(r.sig_contam)]
            if bool(r.cci_present) and sigs == [-1, -1, -1]:
                labs.append("retrieval")
            elif bool(r.cci_present) and sigs == [1, 1, 1]:
                labs.append("computation")
            elif -1 in sigs and 1 in sigs:
                labs.append("mixed")
            else:
                labs.append("ambiguous")
        return pd.Series(labs, index=v.index)

    derived_440 = _from_votes(votes_440)
    assert (derived_440.to_numpy() == lab_app_440.to_numpy()).all(), "vote decoder ≠ appendix function on 440"
    derived_cc = _from_votes(votes_cc_native)
    assert (derived_cc.to_numpy() == lab_app_cc.to_numpy()).all(), "vote decoder ≠ appendix function on 169"

    c_app_440 = count_labels(lab_app_440)
    c_app_cc = count_labels(lab_app_cc)
    lab_app_cc_frozen = _from_votes(votes_cc_frozen)
    c_app_cc_frozen = count_labels(lab_app_cc_frozen)

    lab_ex_440 = label_default(four)
    lab_ex_cc = label_default(cc)
    c_ex_440 = count_labels(lab_ex_440)
    c_ex_cc = count_labels(lab_ex_cc)

    four = four.copy()
    four["label_executed"] = lab_ex_440.to_numpy()
    four["label_appendix"] = lab_app_440.to_numpy()
    cc = cc.copy()
    cc["label_executed"] = lab_ex_cc.to_numpy()
    cc["label_appendix"] = lab_app_cc.to_numpy()
    cc["label_appendix_frozen_p75"] = lab_app_cc_frozen.to_numpy()

    votes_440 = votes_440.copy()
    votes_440["label_appendix"] = lab_app_440.to_numpy()
    votes_440["label_executed"] = lab_ex_440.to_numpy()
    votes_440["pattern_kind"] = [
        _pattern_kind(int(a), int(b), int(c))
        for a, b, c in zip(votes_440["sig_w3"], votes_440["sig_cci"], votes_440["sig_contam"])
    ]
    votes_cc_native = votes_cc_native.copy()
    votes_cc_native["label_appendix"] = lab_app_cc.to_numpy()
    votes_cc_native["label_executed"] = lab_ex_cc.to_numpy()
    votes_cc_native["pattern_kind"] = [
        _pattern_kind(int(a), int(b), int(c))
        for a, b, c in zip(
            votes_cc_native["sig_w3"], votes_cc_native["sig_cci"], votes_cc_native["sig_contam"]
        )
    ]
    votes_cc_frozen = votes_cc_frozen.copy()
    votes_cc_frozen["label_appendix"] = lab_app_cc_frozen.to_numpy()
    votes_cc_frozen["label_executed"] = lab_ex_cc.to_numpy()
    votes_cc_frozen["pattern_kind"] = [
        _pattern_kind(int(a), int(b), int(c))
        for a, b, c in zip(
            votes_cc_frozen["sig_w3"], votes_cc_frozen["sig_cci"], votes_cc_frozen["sig_contam"]
        )
    ]

    def _contingency(v: pd.DataFrame, panel: str) -> pd.DataFrame:
        rows = []
        for w3, cci, cont in product((-1, 0, 1), repeat=3):
            mask = (v["sig_w3"] == w3) & (v["sig_cci"] == cci) & (v["sig_contam"] == cont)
            n = int(mask.sum())
            if n == 0:
                continue
            sub = v.loc[mask]
            vc = sub["label_appendix"].value_counts()
            rows.append({
                "panel": panel,
                "sig_w3": w3,
                "sig_cci": cci,
                "sig_contam": cont,
                "vote_tuple": f"({w3},{cci},{cont})",
                "n": n,
                "pattern_kind": _pattern_kind(w3, cci, cont),
                "n_retrieval": int(vc.get("retrieval", 0)),
                "n_computation": int(vc.get("computation", 0)),
                "n_mixed": int(vc.get("mixed", 0)),
                "n_ambiguous": int(vc.get("ambiguous", 0)),
            })
        return pd.DataFrame(rows).sort_values(["n"], ascending=False)

    cont_440 = _contingency(votes_440, "full_440")
    cont_cc = _contingency(votes_cc_native, "complete_case_169")
    cont_cc_frozen = _contingency(votes_cc_frozen, "complete_case_169_p75_frozen")
    cont_all = pd.concat([cont_440, cont_cc, cont_cc_frozen], ignore_index=True)
    cont_all.to_csv(OUT / "B_appendix_vote_contingency.csv", index=False)

    mixed_440 = votes_440[votes_440["label_appendix"] == "mixed"]
    mixed_cc = votes_cc_native[votes_cc_native["label_appendix"] == "mixed"]
    mixed_cc_frozen = votes_cc_frozen[votes_cc_frozen["label_appendix"] == "mixed"]

    def _mixed_decomp(v: pd.DataFrame, panel: str) -> pd.DataFrame:
        g = (
            v.groupby(["vote_tuple", "pattern_kind"], as_index=False)
            .size()
            .rename(columns={"size": "n"})
            .sort_values("n", ascending=False)
        )
        g.insert(0, "panel", panel)
        return g

    mx = pd.concat(
        [
            _mixed_decomp(mixed_440, "full_440"),
            _mixed_decomp(mixed_cc, "complete_case_169"),
            _mixed_decomp(mixed_cc_frozen, "complete_case_169_p75_frozen"),
        ],
        ignore_index=True,
    )
    mx.to_csv(OUT / "B_appendix_mixed_sign_patterns.csv", index=False)

    votes_440.to_csv(OUT / "appendix_votes_440.csv", index=False)
    votes_cc_native.to_csv(OUT / "appendix_votes_complete_case.csv", index=False)

    counts_rows = [
        {"panel": "full_440", "rule": "executed", **c_ex_440},
        {"panel": "full_440", "rule": "appendix_symmetric", **c_app_440},
        {"panel": "complete_case_169", "rule": "executed", **c_ex_cc},
        {"panel": "complete_case_169", "rule": "appendix_symmetric", **c_app_cc},
        {
            "panel": "complete_case_169",
            "rule": "appendix_symmetric_p75_frozen_from_440",
            **c_app_cc_frozen,
        },
    ]
    pd.DataFrame(counts_rows).to_csv(OUT / "A_label_counts.csv", index=False)

    return {
        "four": four,
        "cc": cc,
        "votes_440": votes_440,
        "votes_cc": votes_cc_native,
        "c_app_440": c_app_440,
        "c_app_cc": c_app_cc,
        "c_app_cc_frozen": c_app_cc_frozen,
        "c_ex_440": c_ex_440,
        "c_ex_cc": c_ex_cc,
        "cont_440": cont_440,
        "cont_cc": cont_cc,
        "cont_cc_frozen": cont_cc_frozen,
        "mixed_patterns": mx,
        "p75_440": p75_440,
        "floor_440": floor_440,
        "p75_cc": p75_cc,
        "floor_cc": floor_cc,
        "lab_ex_440": lab_ex_440,
        "lab_ex_cc": lab_ex_cc,
        "lab_app_440": lab_app_440,
        "lab_app_cc": lab_app_cc,
        "lab_app_cc_frozen": lab_app_cc_frozen,
    }


def executed_condition_flags(df: pd.DataFrame) -> pd.DataFrame:
    rank = pd.to_numeric(df["instance_rank_pct"], errors="coerce")
    can = pd.to_numeric(df["VAR_canonical"], errors="coerce")
    w3 = pd.to_numeric(df["VAR_W3"], errors="coerce")
    aci = pd.to_numeric(df["ACI"], errors="coerce")
    greed = df["greedy_succeeds"]
    greed_ok = greed.fillna(False).astype(bool) & greed.notna()
    high_contam = rank > CONTAM_SPLIT
    low_contam = rank <= CONTAM_SPLIT

    out = df.copy()
    out["ret_ok_canonical"] = can > CANONICAL_RETRIEVAL_MIN
    out["ret_ok_w3"] = w3 < W3_RETRIEVAL_MAX
    out["ret_ok_contam"] = high_contam.fillna(False)
    out["ret_ok_greedy"] = greed_ok
    out["comp_ok_w3"] = w3 > W3_COMPUTATION_MIN
    out["comp_ok_cci"] = aci > CCI_COMPUTATION_MIN
    out["comp_ok_contam"] = low_contam.fillna(False)

    # executed-native three-signal signs (W3 / CCI@0.5 / rank median)
    sig_w3 = np.where(w3 > W3_COMPUTATION_MIN, 1, np.where(w3 < W3_RETRIEVAL_MAX, -1, 0))
    sig_cci = np.where(aci.isna(), 0, np.where(aci > CCI_COMPUTATION_MIN, 1, -1))
    sig_c = np.where(rank.isna(), 0, np.where(rank > CONTAM_SPLIT, -1, 1))
    out["exec_sig_w3"] = sig_w3.astype(int)
    out["exec_sig_cci"] = sig_cci.astype(int)
    out["exec_sig_contam"] = sig_c.astype(int)
    out["exec_vote_tuple"] = [
        f"({int(a)},{int(b)},{int(c)})" for a, b, c in zip(sig_w3, sig_cci, sig_c)
    ]
    out["exec_three_unanimous_retrieval"] = (sig_w3 == -1) & (sig_cci == -1) & (sig_c == -1)
    out["exec_three_unanimous_computation"] = (sig_w3 == 1) & (sig_cci == 1) & (sig_c == 1)
    out["retrieval_side_w3_contam"] = out["ret_ok_w3"] & out["ret_ok_contam"]
    out["computation_side_all_three"] = out["comp_ok_w3"] & out["comp_ok_cci"] & out["comp_ok_contam"]
    # W3 cut gap (0.2, 0.5]: empty on binary 0/1
    out["w3_in_asymmetric_gap"] = (w3 >= W3_RETRIEVAL_MAX) & (w3 <= W3_COMPUTATION_MIN)
    return out


def run_c(four: pd.DataFrame, cc: pd.DataFrame, packed: dict) -> dict:
    lab = packed["lab_ex_440"]
    flags = executed_condition_flags(four)
    flags["label_executed"] = lab.to_numpy()
    flags["label_appendix"] = packed["lab_app_440"].to_numpy()
    mixed = flags[flags["label_executed"] == "mixed"].copy()
    assert len(mixed) == 157

    ret_conds = ["ret_ok_canonical", "ret_ok_w3", "ret_ok_contam", "ret_ok_greedy"]
    comp_conds = ["comp_ok_w3", "comp_ok_cci", "comp_ok_contam"]

    fail_rows = []
    for name, col in [
        ("retrieval: canonical ≤ 0.5", "ret_ok_canonical"),
        ("retrieval: W3 not < 0.2", "ret_ok_w3"),
        ("retrieval: contamination rank not > 0.5", "ret_ok_contam"),
        ("retrieval: greedy_succeeds is not True", "ret_ok_greedy"),
        ("computation: W3 not > 0.5", "comp_ok_w3"),
        ("computation: ACI not > 0.5", "comp_ok_cci"),
        ("computation: contamination rank not ≤ 0.5", "comp_ok_contam"),
    ]:
        n_fail = int((~mixed[col]).sum())
        fail_rows.append({
            "condition": name,
            "n_mixed_failing": n_fail,
            "frac_of_mixed": n_fail / len(mixed),
        })
    fail_df = pd.DataFrame(fail_rows)
    fail_df.to_csv(OUT / "C_executed_mixed_failed_conditions.csv", index=False)

    def _fail_key(r) -> str:
        rf = [c.replace("ret_ok_", "R:") for c in ret_conds if not bool(r[c])]
        cf = [c.replace("comp_ok_", "C:") for c in comp_conds if not bool(r[c])]
        return " | ".join(rf + cf) if (rf or cf) else "(none — should be impossible)"

    mixed["failure_signature"] = mixed.apply(_fail_key, axis=1)
    sig_counts = (
        mixed.groupby("failure_signature", as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values("n", ascending=False)
    )
    sig_counts.to_csv(OUT / "C_executed_mixed_failure_signatures.csv", index=False)

    # appendix votes on executed-mixed (440-panel p75)
    votes = packed["votes_440"]
    mixed_votes = votes.loc[mixed.index]
    app_unan_ret = int((mixed_votes["pattern_kind"] == "unanimous_retrieval").sum())
    app_unan_comp = int((mixed_votes["pattern_kind"] == "unanimous_computation").sum())

    n_exec_unan_ret = int(mixed["exec_three_unanimous_retrieval"].sum())
    n_exec_unan_comp = int(mixed["exec_three_unanimous_computation"].sum())
    n_w3_gap = int(mixed["w3_in_asymmetric_gap"].sum())

    # retrieval-aligned on W3+contam (the retrieval conjunction minus canonical/greedy)
    ret_side = mixed["retrieval_side_w3_contam"]
    n_ret_side = int(ret_side.sum())
    n_ret_side_greedy_only = int((ret_side & mixed["ret_ok_canonical"] & ~mixed["ret_ok_greedy"]).sum())
    n_ret_side_canonical_only = int((ret_side & mixed["ret_ok_greedy"] & ~mixed["ret_ok_canonical"]).sum())
    n_ret_side_both_extra = int((ret_side & ~mixed["ret_ok_greedy"] & ~mixed["ret_ok_canonical"]).sum())

    n_comp_almost = int(mixed["computation_side_all_three"].sum())  # should be 0
    w3_fail = mixed["VAR_W3"] == 0
    low_contam = ~mixed["ret_ok_contam"]
    n_structural_hole = int((w3_fail & low_contam).sum())

    # same-direction but mixed because of greedy / extra retrieval fields / W3 asymmetry
    n_same_dir_exec = n_exec_unan_ret + n_exec_unan_comp
    n_same_dir_app = app_unan_ret + app_unan_comp

    summary = pd.DataFrame([
        {"metric": "n_executed_mixed", "value": len(mixed)},
        {"metric": "n_mixed_appendix_unanimous_retrieval", "value": app_unan_ret},
        {"metric": "n_mixed_appendix_unanimous_computation", "value": app_unan_comp},
        {"metric": "n_mixed_executed_native_unanimous_retrieval", "value": n_exec_unan_ret},
        {"metric": "n_mixed_executed_native_unanimous_computation", "value": n_exec_unan_comp},
        {"metric": "n_mixed_W3_in_asymmetric_gap_0.2_to_0.5", "value": n_w3_gap},
        {"metric": "n_mixed_retrieval_side_W3_and_high_contam", "value": n_ret_side},
        {"metric": "n_mixed_retrieval_side_failed_greedy_only", "value": n_ret_side_greedy_only},
        {"metric": "n_mixed_retrieval_side_failed_canonical_only", "value": n_ret_side_canonical_only},
        {"metric": "n_mixed_retrieval_side_failed_greedy_and_canonical", "value": n_ret_side_both_extra},
        {"metric": "n_mixed_with_all_computation_conjunctions", "value": n_comp_almost},
        {"metric": "n_mixed_structural_hole_W3_fail_and_low_contam", "value": n_structural_hole},
        {"metric": "n_mixed_same_direction_executed_native_three_signals", "value": n_same_dir_exec},
        {"metric": "n_mixed_same_direction_appendix_three_signals", "value": n_same_dir_app},
    ])
    summary.to_csv(OUT / "C_executed_mixed_same_direction.csv", index=False)

    keep_cols = [
        "problem_id", "model", "VAR_canonical", "VAR_W3", "ACI",
        "instance_contamination_score", "instance_rank_pct", "greedy_succeeds",
        "label_executed", "label_appendix",
        "ret_ok_canonical", "ret_ok_w3", "ret_ok_contam", "ret_ok_greedy",
        "comp_ok_w3", "comp_ok_cci", "comp_ok_contam",
        "exec_vote_tuple", "failure_signature",
        "exec_three_unanimous_retrieval", "exec_three_unanimous_computation",
        "w3_in_asymmetric_gap",
    ]
    mixed[keep_cols].to_csv(OUT / "C_executed_mixed_instances.csv", index=False)

    # complete-case mixed is the same 157
    mixed_cc = mixed[mixed.index.isin(cc.index)]
    assert len(mixed_cc) == 157

    return {
        "mixed": mixed,
        "fail_df": fail_df,
        "sig_counts": sig_counts,
        "summary": summary,
        "n_exec_unan_ret": n_exec_unan_ret,
        "n_exec_unan_comp": n_exec_unan_comp,
        "n_app_unan_ret": app_unan_ret,
        "n_app_unan_comp": app_unan_comp,
        "n_w3_gap": n_w3_gap,
        "n_ret_side": n_ret_side,
        "n_ret_side_greedy_only": n_ret_side_greedy_only,
        "n_ret_side_canonical_only": n_ret_side_canonical_only,
        "n_ret_side_both_extra": n_ret_side_both_extra,
        "n_comp_almost": n_comp_almost,
        "n_structural_hole": n_structural_hole,
    }


def _w3_regime(cut: float) -> str:
    """On binary {0,1} VAR_W3, three inequivalent cutoffs."""
    if cut <= 0.0:
        return "cut=0 (retrieval impossible; computation iff W3=1)"
    if cut >= 1.0:
        return "cut=1 (computation impossible; retrieval iff W3=0)"
    return "cut∈(0,1) (retrieval iff W3=0; computation iff W3=1)"


def run_d(four: pd.DataFrame) -> dict:
    rows = []
    label_hashes: dict[tuple, str] = {}
    count_tuples = []
    for cci_thr, w3_cut, pct in product(CCI_THRESHOLDS, W3_CUTOFFS, CONTAM_PERCENTILES):
        lab = label_sweep_cell(four, cci_thr=cci_thr, w3_cutoff=w3_cut, contam_pct=pct)
        c = count_labels(lab)
        key = (round(float(cci_thr), 2), float(w3_cut), int(pct))
        h = "|".join(lab.astype(str).tolist())
        rows.append({
            "cci_threshold": cci_thr,
            "w3_cutoff": w3_cut,
            "w3_regime": _w3_regime(w3_cut),
            "contam_percentile": pct,
            **{f"n_{k}": c[k] for k in ("retrieval", "computation", "mixed", "ambiguous")},
            "n": c["n"],
            "label_vector_id": hash(h) & 0xFFFFFFFF,
        })
        label_hashes[key] = h
        count_tuples.append(
            (c["retrieval"], c["computation"], c["mixed"], c["ambiguous"], c["n"])
        )
    sw = pd.DataFrame(rows)
    sw.to_csv(OUT / "D_sweep_270.csv", index=False)

    # Collapse W3 cutoffs identical on binary data
    n_nominal = len(sw)
    n_w3_raw = len(W3_CUTOFFS)
    regimes = sorted({_w3_regime(x) for x in W3_CUTOFFS})
    n_w3_distinct = len(regimes)
    n_collapsed = len(CCI_THRESHOLDS) * n_w3_distinct * len(CONTAM_PERCENTILES)

    # Empirical check: 0.25, 0.5, 0.75 produce identical label vectors at each (cci, pct)
    mid = (0.25, 0.5, 0.75)
    n_mid_mismatch = 0
    for cci_thr, pct in product(CCI_THRESHOLDS, CONTAM_PERCENTILES):
        hashes = [label_hashes[(round(float(cci_thr), 2), w, int(pct))] for w in mid]
        if len(set(hashes)) != 1:
            n_mid_mismatch += 1

    n_unique_vectors = len(set(label_hashes.values()))
    n_unique_counts = len(set(count_tuples))

    # Unique vectors after dropping redundant W3 cuts (keep 0.0, 0.5, 1.0)
    keep_w3 = (0.0, 0.5, 1.0)
    collapsed_hashes = {
        h for (cci, w, pct), h in label_hashes.items() if w in keep_w3
    }

    summary = pd.DataFrame([
        {"metric": "n_nominal_grid", "value": n_nominal},
        {"metric": "n_W3_cutoffs_listed", "value": n_w3_raw},
        {"metric": "n_W3_regimes_on_binary_VAR_W3", "value": n_w3_distinct},
        {"metric": "n_CCI_thresholds", "value": len(CCI_THRESHOLDS)},
        {"metric": "n_contam_percentiles", "value": len(CONTAM_PERCENTILES)},
        {"metric": "n_distinct_configs_after_W3_collapse", "value": n_collapsed},
        {"metric": "n_mid_cutoff_mismatches_0.25_vs_0.5_vs_0.75", "value": n_mid_mismatch},
        {"metric": "n_empirically_unique_label_vectors_in_270", "value": n_unique_vectors},
        {"metric": "n_empirically_unique_count_tuples_in_270", "value": n_unique_counts},
        {"metric": "n_empirically_unique_label_vectors_in_collapsed_162", "value": len(collapsed_hashes)},
    ])
    summary.to_csv(OUT / "D_sweep_collapse.csv", index=False)
    return {
        "n_nominal": n_nominal,
        "n_collapsed": n_collapsed,
        "n_w3_distinct": n_w3_distinct,
        "n_mid_mismatch": n_mid_mismatch,
        "n_unique_vectors": n_unique_vectors,
        "n_unique_counts": n_unique_counts,
        "n_unique_collapsed": len(collapsed_hashes),
        "regimes": regimes,
    }


def run_e(packed: dict) -> dict:
    four = packed["four"]
    cc = packed["cc"]

    def _xtab(df: pd.DataFrame, panel: str) -> pd.DataFrame:
        ct = pd.crosstab(df["label_executed"], df["label_appendix"], dropna=False)
        for lab in ("retrieval", "computation", "mixed", "ambiguous"):
            if lab not in ct.index:
                ct.loc[lab] = 0
            if lab not in ct.columns:
                ct[lab] = 0
        ct = ct.reindex(index=["retrieval", "computation", "mixed", "ambiguous"],
                        columns=["retrieval", "computation", "mixed", "ambiguous"]).fillna(0).astype(int)
        long = ct.reset_index().melt(id_vars=ct.index.name or "label_executed",
                                     var_name="label_appendix", value_name="n")
        long = long.rename(columns={long.columns[0]: "label_executed"})
        long.insert(0, "panel", panel)
        return ct, long

    xt_440, long_440 = _xtab(four, "full_440")
    xt_cc, long_cc = _xtab(cc, "complete_case_169")
    cc_frozen = cc.copy()
    cc_frozen["label_appendix"] = cc["label_appendix_frozen_p75"]
    xt_cc_f, long_cc_f = _xtab(cc_frozen, "complete_case_169_p75_frozen")
    pd.concat([long_440, long_cc, long_cc_f], ignore_index=True).to_csv(
        OUT / "E_label_crosstab.csv", index=False
    )

    n_change_440 = int((four["label_executed"] != four["label_appendix"]).sum())
    n_change_cc = int((cc["label_executed"] != cc["label_appendix"]).sum())
    n_change_cc_frozen = int((cc["label_executed"] != cc["label_appendix_frozen_p75"]).sum())
    # among complete-case, how many of the 440-changes fall inside
    four_cc = four[four["complete_case"]]
    n_change_440_inside_cc = int(
        (four_cc["label_executed"] != four_cc["label_appendix"]).sum()
    )

    # transition list (compact)
    trans = (
        four.assign(changed=four["label_executed"] != four["label_appendix"])
        .groupby(["label_executed", "label_appendix"], as_index=False)
        .size()
        .rename(columns={"size": "n_440"})
    )
    trans_cc = (
        cc.groupby(["label_executed", "label_appendix"], as_index=False)
        .size()
        .rename(columns={"size": "n_169"})
    )
    trans = trans.merge(trans_cc, on=["label_executed", "label_appendix"], how="outer").fillna(0)
    trans["n_440"] = trans["n_440"].astype(int)
    trans["n_169"] = trans["n_169"].astype(int)
    trans.to_csv(OUT / "E_label_transitions.csv", index=False)

    four[["problem_id", "model", "complete_case", "label_executed", "label_appendix"]].to_csv(
        OUT / "E_per_instance_labels.csv", index=False
    )
    cc[["problem_id", "model", "label_executed", "label_appendix", "label_appendix_frozen_p75"]].to_csv(
        OUT / "E_per_instance_labels_complete_case.csv", index=False
    )
    return {
        "xt_440": xt_440,
        "xt_cc": xt_cc,
        "xt_cc_frozen": xt_cc_f,
        "n_change_440": n_change_440,
        "n_change_cc": n_change_cc,
        "n_change_cc_frozen": n_change_cc_frozen,
        "n_change_440_inside_cc": n_change_440_inside_cc,
        "trans": trans,
    }


def _md_table(df: pd.DataFrame, cols: list[str] | None = None) -> list[str]:
    if cols is None:
        cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|"
    # first col left, rest right if numeric-looking
    lines = [header, "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                if abs(v) <= 1 and c.startswith("frac"):
                    cells.append(f"{v:.3f}")
                elif v == int(v):
                    cells.append(str(int(v)))
                else:
                    cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(a: dict, c: dict, d: dict, e: dict) -> None:
    def _kind_sum(cont: pd.DataFrame, kind: str) -> int:
        sub = cont[cont["pattern_kind"] == kind]
        return int(sub["n"].sum()) if len(sub) else 0

    cont_440 = a["cont_440"]
    cont_cc = a["cont_cc"]
    mx = a["mixed_patterns"]

    lines = [
        "# Triangulation v2 — rule comparison",
        "",
        "Executed rule = `label_default()` (asymmetric 5-field AND). Symmetric rule = `label_appendix_three_signal()` (signed votes on W3, CCI bands 0.10/0.67, contamination floor vs p75). Panels: 440-row 4-model, and the 169 complete-case subset from `rebuild/solidify/` (W3 + CCI + proximity present, parse succeeded). Contamination ranks for the executed rule are inherited from the 440 panel.",
        "",
        "Sign convention for the symmetric rule: **−1 = retrieval-ward**, **+1 = computation-ward**, **0 = neutral or missing**. Tuple order is `(W3, CCI, proximity)`.",
        "",
        "## A. Label counts",
        "",
        f"Appendix contamination p75 / floor on 440: {a['p75_440']:.4g} / {a['floor_440']:.4g}. On 169 (recomputed): {a['p75_cc']:.4g} / {a['floor_cc']:.4g}.",
        "",
        "| panel | rule | retrieval | computation | mixed | ambiguous | n |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| full 440 | executed | {a['c_ex_440']['retrieval']} | {a['c_ex_440']['computation']} | {a['c_ex_440']['mixed']} | {a['c_ex_440']['ambiguous']} | {a['c_ex_440']['n']} |",
        f"| full 440 | appendix symmetric | {a['c_app_440']['retrieval']} | {a['c_app_440']['computation']} | {a['c_app_440']['mixed']} | {a['c_app_440']['ambiguous']} | {a['c_app_440']['n']} |",
        f"| complete-case 169 | executed | {a['c_ex_cc']['retrieval']} | {a['c_ex_cc']['computation']} | {a['c_ex_cc']['mixed']} | {a['c_ex_cc']['ambiguous']} | {a['c_ex_cc']['n']} |",
        f"| complete-case 169 | appendix symmetric | {a['c_app_cc']['retrieval']} | {a['c_app_cc']['computation']} | {a['c_app_cc']['mixed']} | {a['c_app_cc']['ambiguous']} | {a['c_app_cc']['n']} |",
        f"| complete-case 169 | appendix, p75 frozen from 440 | {a['c_app_cc_frozen']['retrieval']} | {a['c_app_cc_frozen']['computation']} | {a['c_app_cc_frozen']['mixed']} | {a['c_app_cc_frozen']['ambiguous']} | {a['c_app_cc_frozen']['n']} |",
        "",
        f"Executed headline: **{_counts_md(a['c_ex_440'])}** vs **{_counts_md(a['c_ex_cc'])}**.",
        f"Symmetric headline: **{_counts_md(a['c_app_440'])}** vs **{_counts_md(a['c_app_cc'])}**.",
        "",
        "On complete-case the executed rule has **ambiguous = 0** (missing-data flags are gone). The symmetric rule can still call instances ambiguous when votes are one-sided or neutral (CCI in (0.10, 0.67), contamination between floor and p75).",
        "",
        f"**p75 collapse.** Recomputing p75 on the 169 subset yields p75 = floor = {a['p75_cc']:.4g}. Every non-negative contamination score is then ≥ p75, so every proximity vote becomes −1 (retrieval-ward). That is why appendix retrieval jumps 8 → 38 and mixed collapses 299 → 27. The frozen-p75 row (5 / 0 / 137 / 27) is the same vote thresholds as the 440 panel, subsetted — use that row to compare rules rather than thresholds.",
        "",
        "## B. Symmetric-rule vote contingency",
        "",
        "Mixed under the symmetric rule is **defined** as at least one −1 and at least one +1. So every mixed instance is a genuine sign conflict. Neutral/missing votes go to **ambiguous**, not mixed. That is the opposite of the executed rule, whose mixed pile is a residual.",
        "",
        "### Full 440 — pattern kinds",
        "",
        f"| kind | n |",
        f"|---|---:|",
        f"| unanimous retrieval (−1,−1,−1) | {_kind_sum(cont_440, 'unanimous_retrieval')} |",
        f"| unanimous computation (+1,+1,+1) | {_kind_sum(cont_440, 'unanimous_computation')} |",
        f"| conflict, all three nonzero | {_kind_sum(cont_440, 'conflict_full')} |",
        f"| conflict, one vote neutral (0) | {_kind_sum(cont_440, 'conflict_partial')} |",
        f"| one-sided or neutral (no conflict) | {_kind_sum(cont_440, 'one_sided_or_neutral')} |",
        f"| all zeros | {_kind_sum(cont_440, 'all_neutral')} |",
        "",
        "### Mixed sign patterns (symmetric rule)",
        "",
        "| panel | vote (W3, CCI, prox) | kind | n |",
        "|---|---|---|---:|",
    ]
    for _, r in mx.iterrows():
        lines.append(f"| {r.panel} | `{r.vote_tuple}` | {r.pattern_kind} | {int(r.n)} |")

    lines += [
        "",
        "Full 27-cell occupancy (only cells with n>0): `B_appendix_vote_contingency.csv`.",
        "",
        "### Complete-case 169 — pattern kinds (p75 recomputed = 0; proximity vote is always −1)",
        "",
        f"| kind | n |",
        f"|---|---:|",
        f"| unanimous retrieval | {_kind_sum(cont_cc, 'unanimous_retrieval')} |",
        f"| unanimous computation | {_kind_sum(cont_cc, 'unanimous_computation')} |",
        f"| conflict, all three nonzero | {_kind_sum(cont_cc, 'conflict_full')} |",
        f"| conflict, one vote neutral | {_kind_sum(cont_cc, 'conflict_partial')} |",
        f"| one-sided or neutral | {_kind_sum(cont_cc, 'one_sided_or_neutral')} |",
        f"| all zeros | {_kind_sum(cont_cc, 'all_neutral')} |",
        "",
        "Frozen-p75 mixed patterns are in the same table under `complete_case_169_p75_frozen`. Dominant 440 mixed cell: **(−1, 0, +1) = 223** — W3 fail vs contamination at floor, CCI silent. That is a two-signal conflict with a missing CCI vote, not a three-way fight.",
        "",
        "## C. Executed-rule mixed: which conjunction failed",
        "",
        f"Executed mixed n = **{len(c['mixed'])}** (all 157 sit inside the 169 complete-case subset).",
        "",
        "Retrieval needs ALL of: canonical > 0.5, W3 < 0.2, rank > 0.5, greedy_succeeds. Computation needs ALL of: W3 > 0.5, ACI > 0.5, rank ≤ 0.5. Mixed = not ambiguous and neither conjunction.",
        "",
        "| condition that failed | n of 157 mixed | fraction |",
        "|---|---:|---:|",
    ]
    for _, r in c["fail_df"].iterrows():
        lines.append(f"| {r.condition} | {int(r.n_mixed_failing)} | {r.frac_of_mixed:.3f} |")

    lines += [
        "",
        "### Same-direction but still mixed",
        "",
        "Three signals = W3, CCI, proximity. Two sign conventions:",
        "",
        "- **Executed-native:** W3 < 0.2 → −1, W3 > 0.5 → +1; ACI > 0.5 → +1 else −1; rank > 0.5 → −1 else +1.",
        "- **Appendix votes:** CCI bands 0.10/0.67; contamination floor / p75.",
        "",
        f"- Mixed with executed-native unanimous retrieval (−1,−1,−1): **{c['n_exec_unan_ret']}**.",
        f"- Mixed with executed-native unanimous computation (+1,+1,+1): **{c['n_exec_unan_comp']}** (must be 0 — that conjunction *is* executed computation).",
        f"- Mixed with appendix unanimous retrieval: **{c['n_app_unan_ret']}**.",
        f"- Mixed with appendix unanimous computation: **{c['n_app_unan_comp']}**.",
        f"- Mixed with W3 in the asymmetric gap (0.2, 0.5]: **{c['n_w3_gap']}** (binary W3 makes this empty).",
        f"- Mixed with retrieval-side W3=0 and high contamination, blocked only by extra retrieval fields: n with that side = **{c['n_ret_side']}**; failed greedy only = **{c['n_ret_side_greedy_only']}**; failed canonical only = **{c['n_ret_side_canonical_only']}**; failed both = **{c['n_ret_side_both_extra']}**.",
        f"- Mixed that satisfy the full computation conjunction: **{c['n_comp_almost']}** (must be 0).",
        f"- **Structural hole:** W3 = 0 and contamination rank ≤ 0.5: **{c['n_structural_hole']}** of 157. Retrieval needs high contamination; computation needs W3 = 1. Neither conjunction can fire.",
        "",
        f"**Same-direction but still mixed: {c['n_exec_unan_ret']} instances** (executed-native all three retrieval-ward). All 8 fail `canonical > 0.5`; 1 of those also fails `greedy_succeeds`. W3-cut asymmetry (0.2 vs 0.5) creates **zero** mixed labels: VAR_W3 is 0/1. The 8 are mixed because of the extra retrieval fields (canonical, and in one case greedy), not because the three named signals disagreed.",
        "",
        "The bulk of mixed (128/157) is the structural hole, not greedy and not W3-cut asymmetry.",
        "",
        "Failure signatures (which subset of conditions failed): `C_executed_mixed_failure_signatures.csv`.",
        "",
        "## D. Sweep after collapsing identical W3 cutoffs",
        "",
        "Nominal grid: 18 CCI × 5 W3 × 3 contamination percentiles = **270**.",
        "",
        "VAR_W3 is binary. `label_sweep_cell` uses one cutoff on both sides (`W3 < cut` retrieval, `W3 > cut` computation):",
        "",
        "- cut = 0.0: retrieval never (W3 < 0 is empty); computation iff W3 = 1.",
        "- cut ∈ {0.25, 0.50, 0.75}: retrieval iff W3 = 0; computation iff W3 = 1. **These three are identical.**",
        "- cut = 1.0: computation never (W3 > 1 is empty); retrieval iff W3 = 0.",
        "",
        f"**Distinct configurations after collapsing W3: {d['n_collapsed']}** = 18 × 3 × 3.",
        f"Empirical check that 0.25/0.50/0.75 produce identical label vectors at every (CCI, contam) cell: mismatches = {d['n_mid_mismatch']}.",
        f"Empirically unique label vectors among the 270: {d['n_unique_vectors']}. Unique count-tuples: {d['n_unique_counts']}. Unique label vectors among the collapsed 162: {d['n_unique_collapsed']}.",
        "",
        "## E. Instances that change label",
        "",
        f"- Full 440: **{e['n_change_440']}** / 440 change label ({e['n_change_440']/440:.1%}).",
        f"- Complete-case 169, appendix p75 recomputed: **{e['n_change_cc']}** / 169 ({e['n_change_cc']/169:.1%}). This includes the p75-collapse artefact.",
        f"- Complete-case 169, appendix p75 frozen from 440: **{e['n_change_cc_frozen']}** / 169 ({e['n_change_cc_frozen']/169:.1%}). Prefer this for rule-vs-rule.",
        "",
        "Rows = executed, columns = appendix symmetric.",
        "",
        "### Full 440",
        "",
        "| executed \\ appendix | retrieval | computation | mixed | ambiguous |",
        "|---|---:|---:|---:|---:|",
    ]
    xt = e["xt_440"]
    for row in ("retrieval", "computation", "mixed", "ambiguous"):
        lines.append(
            f"| {row} | {int(xt.loc[row, 'retrieval'])} | {int(xt.loc[row, 'computation'])} | "
            f"{int(xt.loc[row, 'mixed'])} | {int(xt.loc[row, 'ambiguous'])} |"
        )
    xtc = e["xt_cc"]
    lines += [
        "",
        "### Complete-case 169",
        "",
        "| executed \\ appendix | retrieval | computation | mixed | ambiguous |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ("retrieval", "computation", "mixed", "ambiguous"):
        lines.append(
            f"| {row} | {int(xtc.loc[row, 'retrieval'])} | {int(xtc.loc[row, 'computation'])} | "
            f"{int(xtc.loc[row, 'mixed'])} | {int(xtc.loc[row, 'ambiguous'])} |"
        )
    xtf = e["xt_cc_frozen"]
    lines += [
        "",
        "### Complete-case 169, appendix p75 frozen from 440",
        "",
        "| executed \\ appendix | retrieval | computation | mixed | ambiguous |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ("retrieval", "computation", "mixed", "ambiguous"):
        lines.append(
            f"| {row} | {int(xtf.loc[row, 'retrieval'])} | {int(xtf.loc[row, 'computation'])} | "
            f"{int(xtf.loc[row, 'mixed'])} | {int(xtf.loc[row, 'ambiguous'])} |"
        )
    lines += [
        "",
        "Per-instance labels: `E_per_instance_labels.csv`.",
        "",
        "## Files",
        "",
        "- `A_label_counts.csv`",
        "- `B_appendix_vote_contingency.csv`, `B_appendix_mixed_sign_patterns.csv`",
        "- `C_executed_mixed_failed_conditions.csv`, `C_executed_mixed_failure_signatures.csv`, `C_executed_mixed_same_direction.csv`, `C_executed_mixed_instances.csv`",
        "- `D_sweep_collapse.csv`, `D_sweep_270.csv`",
        "- `E_label_crosstab.csv`, `E_label_transitions.csv`, `E_per_instance_labels.csv`",
        "",
    ]
    (OUT / "TRI_V2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    four, cc = load_panels()
    print("A/B appendix counts and votes…")
    a = run_a_b(four, cc)
    print(f"  appendix 440: {a['c_app_440']}")
    print(f"  appendix 169: {a['c_app_cc']}")
    print("C executed mixed decomposition…")
    c = run_c(four, cc, a)
    print(f"  mixed={len(c['mixed'])} exec_unan_ret={c['n_exec_unan_ret']} greedy_only={c['n_ret_side_greedy_only']}")
    print("D sweep collapse…")
    d = run_d(four)
    print(f"  270 → {d['n_collapsed']} after W3 collapse")
    print("E label changes…")
    e = run_e(a)
    print(f"  changed 440={e['n_change_440']} 169={e['n_change_cc']}")
    write_report(a, c, d, e)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
