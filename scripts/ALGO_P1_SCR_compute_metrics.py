#!/usr/bin/env python3
"""Compute Probe 1 algorithmic metrics from behavioral sweep outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.stats import bootstrap_ci


def _to_bool(val: object) -> bool:
    return str(val).strip().lower() == "true"


def _difficulty_to_numeric(value: str) -> int:
    mapping = {"easy": 1, "medium": 2, "hard": 3}
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown difficulty value for numeric mapping: {value!r}")
    return mapping[key]


def _normalize_answer(text: str) -> str:
    s = str(text).lower().replace("→", "->")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w\-\>\[\]\{\},:]", "", s)
    return s


def _bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_resamples: int
) -> tuple[float, float]:
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_resamples):
        ra = a[rng.integers(0, len(a), len(a))]
        rb = b[rng.integers(0, len(b), len(b))]
        diffs.append(float(np.mean(ra) - np.mean(rb)))
    return (
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def _add_metric(
    rows: list[dict],
    *,
    model: str,
    subtype: str,
    variant_type: str,
    metric_name: str,
    metric_value: float | str,
    ci_lower: float | str = "",
    ci_upper: float | str = "",
) -> None:
    rows.append(
        {
            "model": model,
            "subtype": subtype,
            "variant_type": variant_type,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    )


def _extract_first_decision(
    subtype: str, answer: str, params: dict
) -> str | None:
    text = str(answer)
    if subtype == "coin_change":
        nums = [int(x) for x in re.findall(r"\d+", text)]
        if not nums:
            return None
        return str(nums[0])
    if subtype == "shortest_path":
        norm = text.replace("→", "->")
        m = re.search(r"([a-z0-9 ]+(?:\s*->\s*[a-z0-9 ]+)+)", norm, flags=re.IGNORECASE)
        if not m:
            return None
        toks = [t.strip() for t in re.split(r"\s*->\s*", m.group(1)) if t.strip()]
        if len(toks) < 2:
            return None
        node_mapping = params.get("node_mapping", {})
        rev = {str(v).lower(): str(k) for k, v in node_mapping.items()}

        def norm_tok(t: str) -> str:
            x = t.lower().strip()
            if x in rev:
                return rev[x]
            x2 = re.sub(r"^\s*hub\s+", "", x).strip()
            if re.fullmatch(r"[a-z]", x2):
                key = f"hub {x2}"
                if key in rev:
                    return rev[key]
            if re.fullmatch(r"-?\d+", x):
                return x
            return t

        return f"{norm_tok(toks[0])}->{norm_tok(toks[1])}"
    if subtype == "wis":
        m = re.search(r"\{([^}]*)\}", text)
        if not m:
            return None
        toks = [t.strip() for t in m.group(1).split(",") if t.strip()]
        if not toks:
            return None
        item_mapping = params.get("item_mapping", {})
        rev = {str(v).lower(): str(k) for k, v in item_mapping.items()}
        first = toks[0].lower()
        return rev.get(first, toks[0])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-results", nargs="+", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    args = parser.parse_args()

    sweep_frames = [pd.read_csv(Path(p), dtype=str).fillna("") for p in args.sweep_results]
    sweep = pd.concat(sweep_frames, ignore_index=True)
    bank = pd.read_csv(Path(args.bank), dtype=str).fillna("")

    required_sweep = {
        "problem_id",
        "variant_type",
        "model",
        "model_answer",
        "ground_truth",
        "verified",
        "parse_status",
        "correct_alternative",
        "human_review_flag",
        "correct_canonical",
        "greedy_answer",
        "gave_greedy_answer",
        "difficulty_params_instance_type",
    }
    miss = required_sweep - set(sweep.columns)
    if miss:
        raise ValueError(f"Sweep results missing required columns: {sorted(miss)}")

    required_bank = {"problem_id", "variant_type", "problem_subtype", "difficulty", "difficulty_params"}
    miss_bank = required_bank - set(bank.columns)
    if miss_bank:
        raise ValueError(f"Bank missing required columns: {sorted(miss_bank)}")

    merged = sweep.merge(
        bank[["problem_id", "variant_type", "problem_subtype", "difficulty", "difficulty_params"]],
        on=["problem_id", "variant_type"],
        how="left",
        validate="many_to_one",
    )
    if merged["problem_subtype"].eq("").any():
        raise ValueError("Join failed for some rows: missing problem_subtype after merge.")

    parsed_params: list[dict] = []
    for _, r in merged.iterrows():
        try:
            params = json.loads(r["difficulty_params"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid difficulty_params JSON at {r['problem_id']}/{r['variant_type']}: {exc}"
            ) from exc
        if not isinstance(params, dict):
            raise ValueError(
                f"difficulty_params must decode to object at {r['problem_id']}/{r['variant_type']}"
            )
        parsed_params.append(params)
    merged["difficulty_params_obj"] = parsed_params

    # Required fields in difficulty_params for this metric layer.
    for _, r in merged.iterrows():
        sub = r["problem_subtype"]
        params = r["difficulty_params_obj"]
        if sub in {"coin_change", "wis"} and "greedy_succeeds" not in params:
            raise ValueError(f"Missing greedy_succeeds for {r['problem_id']}/{r['variant_type']}")
        if "instance_type" not in params:
            raise ValueError(f"Missing instance_type in difficulty_params for {r['problem_id']}/{r['variant_type']}")
        if "greedy_answer" not in params:
            raise ValueError(f"Missing greedy_answer in difficulty_params for {r['problem_id']}/{r['variant_type']}")

    # Optional contamination score for contamination-controlled regression.
    contamination_available = False
    if "contamination_score" in merged.columns:
        merged["contamination_score"] = pd.to_numeric(
            merged["contamination_score"], errors="coerce"
        )
        contamination_available = not merged["contamination_score"].isna().any()
    else:
        merged["contamination_score"] = merged["difficulty_params_obj"].map(
            lambda p: p.get("contamination_score", np.nan)
        )
        merged["contamination_score"] = pd.to_numeric(
            merged["contamination_score"], errors="coerce"
        )
        contamination_available = not merged["contamination_score"].isna().any()

    if not contamination_available:
        print(
            "contamination_score not found — skipping contamination-controlled regression"
        )
        print("Running metrics WITHOUT contamination control")

    merged["verified_bool"] = merged["verified"].map(_to_bool)
    merged["correct_canonical_bool"] = merged["correct_canonical"].map(_to_bool)
    merged["difficulty_numeric"] = merged["difficulty"].map(_difficulty_to_numeric)
    merged["subtype"] = merged["problem_subtype"]
    merged["instance_type"] = merged["difficulty_params_instance_type"]
    merged["greedy_succeeds"] = merged["difficulty_params_obj"].map(
        lambda p: bool(p.get("greedy_succeeds", False))
    )

    metric_rows: list[dict] = []

    # 4.1 VAR by variant_type/model/subtype and by instance_type/model.
    for (model, subtype, variant), g in merged.groupby(["model", "subtype", "variant_type"]):
        vals = g["verified_bool"].astype(float).tolist()
        mean = float(np.mean(vals))
        lo, hi = bootstrap_ci(vals, n_resamples=args.bootstrap_n)
        _add_metric(
            metric_rows,
            model=model,
            subtype=subtype,
            variant_type=variant,
            metric_name="VAR",
            metric_value=mean,
            ci_lower=lo,
            ci_upper=hi,
        )
    for (model, subtype, inst), g in merged.groupby(["model", "subtype", "instance_type"]):
        vals = g["verified_bool"].astype(float).tolist()
        mean = float(np.mean(vals))
        lo, hi = bootstrap_ci(vals, n_resamples=args.bootstrap_n)
        _add_metric(
            metric_rows,
            model=model,
            subtype=subtype,
            variant_type=inst,
            metric_name="VAR_by_instance_type",
            metric_value=mean,
            ci_lower=lo,
            ci_upper=hi,
        )

    # 4.2 CSS split by canonical correctness.
    css_rows = []
    for (model, pid), g in merged.groupby(["model", "problem_id"]):
        subtype = g["subtype"].iloc[0]
        canon = g[g["variant_type"] == "canonical"]
        if len(canon) != 1:
            raise ValueError(f"Expected one canonical row for {model}/{pid}")
        canon_ans = _normalize_answer(canon.iloc[0]["model_answer"])
        canon_correct = bool(canon.iloc[0]["verified_bool"])
        variants = g[g["variant_type"] != "canonical"]
        if variants.empty:
            continue
        match = variants["model_answer"].map(_normalize_answer).eq(canon_ans)
        css_rows.append(
            {
                "model": model,
                "subtype": subtype,
                "problem_id": pid,
                "css": float(match.mean()),
                "canonical_correct": canon_correct,
            }
        )
    css_df = pd.DataFrame(css_rows)
    for (model, subtype, cc), g in css_df.groupby(["model", "subtype", "canonical_correct"]):
        vals = g["css"].tolist()
        lo, hi = bootstrap_ci(vals, n_resamples=args.bootstrap_n)
        name = "CSS_correct_canonical" if cc else "CSS_incorrect_canonical"
        _add_metric(
            metric_rows,
            model=model,
            subtype=subtype,
            variant_type="",
            metric_name=name,
            metric_value=float(np.mean(vals)),
            ci_lower=lo,
            ci_upper=hi,
        )

    # 4.3 VRI
    for (model, subtype), g in merged.groupby(["model", "subtype"]):
        w2 = g[g["variant_type"] == "W2"]["verified_bool"].astype(float).to_numpy()
        w4 = g[g["variant_type"] == "W4"]["verified_bool"].astype(float).to_numpy()
        w3 = g[g["variant_type"] == "W3"]["verified_bool"].astype(float).to_numpy()
        if len(w2) == 0 or len(w4) == 0 or len(w3) == 0:
            continue
        v2 = float(np.mean(w2))
        v4 = float(np.mean(w4))
        vv = float(np.mean(w3))
        lo2, hi2 = bootstrap_ci(w2.tolist(), n_resamples=args.bootstrap_n)
        lo4, hi4 = bootstrap_ci(w4.tolist(), n_resamples=args.bootstrap_n)
        lov, hiv = bootstrap_ci(w3.tolist(), n_resamples=args.bootstrap_n)
        _add_metric(metric_rows, model=model, subtype=subtype, variant_type="W2", metric_name="VRI_W2", metric_value=v2, ci_lower=lo2, ci_upper=hi2)
        _add_metric(metric_rows, model=model, subtype=subtype, variant_type="W4", metric_name="VRI_W4", metric_value=v4, ci_lower=lo4, ci_upper=hi4)
        structural = (v2 + v4) / 2.0
        _add_metric(metric_rows, model=model, subtype=subtype, variant_type="", metric_name="VRI_structural", metric_value=structural)
        _add_metric(metric_rows, model=model, subtype=subtype, variant_type="W3", metric_name="VRI_vocabulary", metric_value=vv, ci_lower=lov, ci_upper=hiv)
        # gap CI from bootstrap difference over independent draws.
        lo_gap, hi_gap = _bootstrap_diff((w2 + w4) / 2.0, w3, args.bootstrap_n)
        _add_metric(
            metric_rows,
            model=model,
            subtype=subtype,
            variant_type="",
            metric_name="VRI_gap",
            metric_value=structural - vv,
            ci_lower=lo_gap,
            ci_upper=hi_gap,
        )

    # 4.4 GSS
    canon = merged[merged["variant_type"] == "canonical"].copy()
    for (model, subtype), g in canon.groupby(["model", "subtype"]):
        pos = g[g["greedy_succeeds"] == True]["verified_bool"].astype(float).to_numpy()
        neg = g[g["greedy_succeeds"] == False]["verified_bool"].astype(float).to_numpy()
        if len(pos) == 0 or len(neg) == 0:
            continue
        gss = float(np.mean(pos) - np.mean(neg))
        lo, hi = _bootstrap_diff(pos, neg, args.bootstrap_n)
        _add_metric(metric_rows, model=model, subtype=subtype, variant_type="canonical", metric_name="GSS", metric_value=gss, ci_lower=lo, ci_upper=hi)

    # 4.4B Regression
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        raise RuntimeError("statsmodels is required for regression metrics.") from exc
    reg_df = canon.copy()
    reg_df["VAR"] = reg_df["verified_bool"].astype(float)
    reg_df["greedy_succeeds_num"] = reg_df["greedy_succeeds"].astype(int)
    m1 = smf.ols(
        "VAR ~ greedy_succeeds_num + difficulty_numeric + C(subtype)", data=reg_df
    ).fit()
    b1 = float(m1.params["greedy_succeeds_num"])
    _add_metric(
        metric_rows,
        model="ALL",
        subtype="ALL",
        variant_type="canonical",
        metric_name="GSS_regression_basic",
        metric_value=b1,
    )

    if contamination_available:
        m2 = smf.ols(
            "VAR ~ greedy_succeeds_num + contamination_score + difficulty_numeric + C(subtype)",
            data=reg_df,
        ).fit()
        b2 = float(m2.params["greedy_succeeds_num"])
        _add_metric(
            metric_rows,
            model="ALL",
            subtype="ALL",
            variant_type="canonical",
            metric_name="GSS_regression_with_contamination",
            metric_value=b2,
        )
        _add_metric(
            metric_rows,
            model="ALL",
            subtype="ALL",
            variant_type="canonical",
            metric_name="GSS_regression_coef_delta",
            metric_value=b1 - b2,
        )

    # 4.5 Formalism gap
    for (model, subtype), g in merged.groupby(["model", "subtype"]):
        c = g[g["variant_type"] == "canonical"]["verified_bool"].astype(float).to_numpy()
        w4 = g[g["variant_type"] == "W4"]["verified_bool"].astype(float).to_numpy()
        if len(c) == 0 or len(w4) == 0:
            continue
        gap = float(np.mean(w4) - np.mean(c))
        lo, hi = _bootstrap_diff(w4, c, args.bootstrap_n)
        _add_metric(metric_rows, model=model, subtype=subtype, variant_type="W4", metric_name="Formalism_Gap", metric_value=gap, ci_lower=lo, ci_upper=hi)

    # 4.6 CFS
    for (model, subtype), g in merged.groupby(["model", "subtype"]):
        by_pid = []
        cfs_valid = True
        for pid, gp in g.groupby("problem_id"):
            can = gp[gp["variant_type"] == "canonical"]
            if len(can) != 1:
                continue
            if bool(can.iloc[0]["verified_bool"]):
                continue  # require VAR(canonical)==0 condition at instance level
            params = can.iloc[0]["difficulty_params_obj"]
            if subtype == "shortest_path":
                source = int(params["source"])
                edges = params["graph"]
                deg = sum(1 for e in edges if int(e["u"]) == source) + sum(
                    1 for e in edges if (not params.get("directed", True)) and int(e["v"]) == source
                )
                if deg == 1:
                    cfs_valid = False
                    continue
            decisions = []
            for _, r in gp.iterrows():
                d = _extract_first_decision(subtype, r["model_answer"], r["difficulty_params_obj"])
                if d is not None:
                    decisions.append(d)
            if not decisions:
                continue
            mode = pd.Series(decisions).value_counts().iloc[0]
            by_pid.append(float(mode / len(decisions)))
        if by_pid:
            lo, hi = bootstrap_ci(by_pid, n_resamples=args.bootstrap_n)
            _add_metric(metric_rows, model=model, subtype=subtype, variant_type="", metric_name="CFS", metric_value=float(np.mean(by_pid)), ci_lower=lo, ci_upper=hi)
        _add_metric(metric_rows, model=model, subtype=subtype, variant_type="", metric_name="cfs_valid", metric_value=str(cfs_valid))

    # 4.7 HDR
    adv = canon[canon["instance_type"] == "adversarial"].copy()
    for (model, subtype), g in adv.groupby(["model", "subtype"]):
        vals = g["gave_greedy_answer"].map(_to_bool).astype(float).tolist()
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals, n_resamples=args.bootstrap_n)
        _add_metric(metric_rows, model=model, subtype=subtype, variant_type="canonical", metric_name="HDR", metric_value=float(np.mean(vals)), ci_lower=lo, ci_upper=hi)

    # 4.8 VWC canonical only
    def vocab_count(ans: str, subtype: str) -> int:
        if subtype == "coin_change":
            return len(re.findall(r"\d+", ans))
        if subtype == "shortest_path":
            t = ans.replace("→", "->")
            m = re.search(r"([a-z0-9 ]+(?:\s*->\s*[a-z0-9 ]+)+)", t, flags=re.IGNORECASE)
            if not m:
                return 0
            return len([x for x in re.split(r"\s*->\s*", m.group(1)) if x.strip()])
        if subtype == "wis":
            m = re.search(r"\{([^}]*)\}", ans)
            if not m:
                return 0
            return len([x for x in m.group(1).split(",") if x.strip()])
        return 0

    for (model, subtype), g in canon.groupby(["model", "subtype"]):
        x = g["model_answer"].map(lambda s: vocab_count(str(s), subtype)).astype(float)
        y = g["verified_bool"].astype(float)
        if len(x) < 2 or x.nunique() <= 1:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        _add_metric(metric_rows, model=model, subtype=subtype, variant_type="canonical", metric_name="VWC", metric_value=corr)

    out_df = pd.DataFrame(metric_rows, columns=["model", "subtype", "variant_type", "metric_name", "metric_value", "ci_lower", "ci_upper"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote metrics: {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
