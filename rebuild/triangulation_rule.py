"""Canonical triangulation label rule: appendix-printed three-signal conjunction.

Executed path = appendix print. There is one published rule.

  retrieval   : W3 vote −1 AND CCI ≤ 0.10 AND contamination ≥ p75
  computation : W3 vote +1 AND CCI ≥ 0.67 AND contamination at floor
  mixed       : conflicting signs across the three signals
  ambiguous   : remainder (including missing CCI)

W3 is symmetric (0 vs 1). greedy_succeeds is not a conjunct.

The former 5-field AND (asymmetric W3 0.2/0.5, greedy_succeeds, CCI 0.5,
median split) is kept as ``label_legacy_five_field`` — a named sensitivity
variant, not the published numbers.

The 270-configuration sweep applies the appendix *structure* (signed votes,
mixed = conflict, no greedy) across CCI / W3-cut / contamination-percentile.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Named constants — canonical (appendix) default
# ---------------------------------------------------------------------------
APPENDIX_CCI_RETRIEVAL_MAX = 0.10
APPENDIX_CCI_COMPUTATION_MIN = 0.67
APPENDIX_CONTAM_PERCENTILE = 75
APPENDIX_W3_CUTOFF = 0.5  # symmetric: VAR_W3 < cut → retrieval vote; > cut → computation

# Legacy 5-field AND constants (sensitivity variant only)
CANONICAL_RETRIEVAL_MIN = 0.5
W3_RETRIEVAL_MAX = 0.2
W3_COMPUTATION_MIN = 0.5
CCI_COMPUTATION_MIN = 0.5
CONTAM_SPLIT = 0.5
MIN_VOTES_FOR_AMBIGUOUS_OVERRIDE = 0

# 270-config sweep grid (18 × 5 × 3 = 270)
CCI_THRESHOLDS: tuple[float, ...] = tuple(round(x, 2) for x in np.arange(0.05, 0.90 + 1e-9, 0.05))
W3_CUTOFFS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
CONTAM_PERCENTILES: tuple[int, ...] = (50, 75, 90)

REQUIRED_CORE_FIELDS: tuple[str, ...] = (
    "VAR_W3",
    "instance_contamination_score",
)

PAPER_COUNTS = {
    "retrieval": 15,
    "computation": 1,
    "mixed": 300,
    "ambiguous": 124,
    "n": 440,
}

LEGACY_COUNTS = {
    "retrieval": 8,
    "computation": 4,
    "mixed": 157,
    "ambiguous": 271,
    "n": 440,
}


def _as_bool_mask(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.fillna(False).to_numpy()
    return s.fillna(False).astype(bool).to_numpy()


def label_default(df: pd.DataFrame) -> pd.Series:
    """Canonical published rule: appendix three-signal conjunction."""
    return label_appendix_three_signal(df)


def label_legacy_five_field(df: pd.DataFrame) -> pd.Series:
    """Named sensitivity variant: asymmetric W3 0.2/0.5, greedy_succeeds, CCI 0.5, median split."""
    return label_with_thresholds(
        df,
        cci_thr=CCI_COMPUTATION_MIN,
        w3_retrieval_max=W3_RETRIEVAL_MAX,
        w3_computation_min=W3_COMPUTATION_MIN,
        contam_split=CONTAM_SPLIT,
        canonical_retrieval_min=CANONICAL_RETRIEVAL_MIN,
    )


def label_with_thresholds(
    df: pd.DataFrame,
    *,
    cci_thr: float,
    w3_retrieval_max: float,
    w3_computation_min: float,
    contam_split: float,
    canonical_retrieval_min: float = CANONICAL_RETRIEVAL_MIN,
) -> pd.Series:
    """Legacy 5-field AND with explicit numeric thresholds (sensitivity variant).

    Expected columns on ``df``:
      VAR_canonical, VAR_W3, ACI, instance_rank_pct, greedy_succeeds,
      missing_core, parse_failure_or_missing, missing_phase2
    """
    rank = pd.to_numeric(df["instance_rank_pct"], errors="coerce")
    high_contam = rank > contam_split
    low_contam = rank <= contam_split

    out = pd.Series("mixed", index=df.index, dtype=object)
    ambiguous = (
        _as_bool_mask(df["missing_core"])
        | _as_bool_mask(df["parse_failure_or_missing"])
        | _as_bool_mask(df["missing_phase2"])
    )
    out.loc[ambiguous] = "ambiguous"

    greed = df["greedy_succeeds"]
    greed_ok = greed.fillna(False).astype(bool) & greed.notna()

    can = pd.to_numeric(df["VAR_canonical"], errors="coerce")
    w3 = pd.to_numeric(df["VAR_W3"], errors="coerce")
    aci = pd.to_numeric(df["ACI"], errors="coerce")

    retrieval = (
        (can > canonical_retrieval_min)
        & (w3 < w3_retrieval_max)
        & high_contam
        & greed_ok
        & ~ambiguous
    )
    computation = (
        (w3 > w3_computation_min)
        & (aci > cci_thr)
        & low_contam
        & ~ambiguous
    )
    out.loc[retrieval] = "retrieval"
    # retrieval wins ties (same as ALGO_P3_SCR_triangulation.py)
    out.loc[~retrieval & computation] = "computation"
    return out


def label_sweep_cell(
    df: pd.DataFrame,
    *,
    cci_thr: float,
    w3_cutoff: float,
    contam_pct: int,
) -> pd.Series:
    """270-config appendix structure: one CCI retrieval-max, symmetric W3, contam percentile.

    CCI computation-min is paired as ``1 - cci_thr`` so the 18-point CCI axis
    sweeps a dead zone. The exact appendix bands (0.10 / 0.67) are not a grid
    point; they are ``label_appendix_three_signal``.
    """
    retrieval_max = min(cci_thr, 1.0 - cci_thr)
    computation_min = max(cci_thr, 1.0 - cci_thr)
    return label_appendix_with_thresholds(
        df,
        cci_retrieval_max=retrieval_max,
        cci_computation_min=computation_min,
        w3_cutoff=w3_cutoff,
        contam_pct=contam_pct,
    )


def label_legacy_sweep_cell(
    df: pd.DataFrame,
    *,
    cci_thr: float,
    w3_cutoff: float,
    contam_pct: int,
) -> pd.Series:
    """270-config parameterization of the legacy 5-field AND."""
    return label_with_thresholds(
        df,
        cci_thr=cci_thr,
        w3_retrieval_max=w3_cutoff,
        w3_computation_min=w3_cutoff,
        contam_split=contam_pct / 100.0,
    )


def label_appendix_with_thresholds(
    df: pd.DataFrame,
    *,
    cci_retrieval_max: float,
    cci_computation_min: float,
    w3_cutoff: float,
    contam_pct: int,
) -> pd.Series:
    """Appendix structure (signed votes, mixed=conflict) with explicit thresholds."""
    w3 = pd.to_numeric(df["VAR_W3"], errors="coerce")
    cci = pd.to_numeric(df["ACI"], errors="coerce")
    contam = pd.to_numeric(df["instance_contamination_score"], errors="coerce")
    p_high = float(contam.quantile(contam_pct / 100.0)) if contam.notna().any() else 0.0
    floor = float(contam.min()) if contam.notna().any() else 0.0

    labels = []
    for i in df.index:
        w3v = w3.loc[i]
        cciv = cci.loc[i]
        cv = contam.loc[i]
        if pd.isna(w3v):
            sig_w3 = 0
        elif w3v > w3_cutoff:
            sig_w3 = 1
        elif w3v < w3_cutoff:
            sig_w3 = -1
        else:
            sig_w3 = 0
        if pd.notna(cciv):
            if cciv <= cci_retrieval_max:
                sig_cci = -1
            elif cciv >= cci_computation_min:
                sig_cci = 1
            else:
                sig_cci = 0
        else:
            sig_cci = 0
        if pd.notna(cv):
            if cv >= p_high:
                sig_c = -1
            elif abs(cv - floor) <= 1e-12:
                sig_c = 1
            else:
                sig_c = 0
        else:
            sig_c = 0
        sigs = [sig_w3, sig_cci, sig_c]
        cci_ok = pd.notna(cciv)
        if cci_ok and sig_w3 == -1 and sig_cci == -1 and sig_c == -1:
            labels.append("retrieval")
        elif cci_ok and sig_w3 == 1 and sig_cci == 1 and sig_c == 1:
            labels.append("computation")
        elif any(s == -1 for s in sigs) and any(s == 1 for s in sigs):
            labels.append("mixed")
        else:
            labels.append("ambiguous")
    return pd.Series(labels, index=df.index, dtype=object)


def label_appendix_three_signal(df: pd.DataFrame) -> pd.Series:
    """Appendix-printed three-signal conjunction (canonical published rule).

    Strong retrieval / computation require W3, CCI, and contamination all
    aligned. Mixed = conflicting directions. Ambiguous = remainder
    (including missing CCI). No greedy_succeeds conjunct.
    """
    return label_appendix_with_thresholds(
        df,
        cci_retrieval_max=APPENDIX_CCI_RETRIEVAL_MAX,
        cci_computation_min=APPENDIX_CCI_COMPUTATION_MIN,
        w3_cutoff=APPENDIX_W3_CUTOFF,
        contam_pct=APPENDIX_CONTAM_PERCENTILE,
    )


def count_labels(labels: Iterable[str] | pd.Series) -> dict[str, int]:
    s = pd.Series(list(labels) if not isinstance(labels, pd.Series) else labels)
    vc = s.value_counts()
    return {
        "retrieval": int(vc.get("retrieval", 0) + vc.get("retrieval_signal", 0)),
        "computation": int(vc.get("computation", 0) + vc.get("computation_signal", 0)),
        "mixed": int(vc.get("mixed", 0)),
        "ambiguous": int(vc.get("ambiguous", 0)),
        "n": int(len(s)),
    }


def matches_paper_counts(counts: dict[str, int]) -> bool:
    return all(counts.get(k) == PAPER_COUNTS[k] for k in ("retrieval", "computation", "mixed", "ambiguous", "n"))
