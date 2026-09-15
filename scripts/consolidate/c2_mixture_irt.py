#!/usr/bin/env python3
"""C2 / DS-01: Latent-strategy mixture IRT on Probe-1 binary responses.

Pre-req check: O10 item variance must be large enough for IRT identification.
Families that fail are reported in C2_identifiability_check.csv and skipped.
Passing families get 1PL/2PL baselines + mixture Rasch (K=2,3,4) with BIC and
bootstrap LRT model selection.

Outputs (results/derived/):
  C2_identifiability_check.csv
  C2_irt_fit_comparison.csv
  C2_class_profiles.csv
  strategy_posteriors.csv
  C2_o12_kappa.csv  (O12 cross-check; NA if ALGO failed ID)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402

try:
    from girth import onepl_mml, twopl_mml, ability_eap
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"girth required for C2 baseline IRT: {exc}") from exc

DER = REPO_ROOT / "results" / "derived"
OUT_ID = DER / "C2_identifiability_check.csv"
OUT_FIT = DER / "C2_irt_fit_comparison.csv"
OUT_PROF = DER / "C2_class_profiles.csv"
OUT_POST = DER / "strategy_posteriors.csv"
OUT_KAPPA = DER / "C2_o12_kappa.csv"
OUT_ITEMS = DER / "C2_irt_item_params.csv"
OUT_CODING = DER / "C2_IRT_CODING.md"

PAPER_MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
VARIANTS = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]
# Prefer complete 7-variant patterns; fall back drops W6 then W5.
VARIANT_SETS = [
    ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"],
    ["canonical", "W1", "W2", "W3", "W4", "W5"],
    ["canonical", "W1", "W2", "W3", "W4"],
]

# O10 primary item-variance gate (user: BW≈0.002 / ALGO≈0.018 fail; GSM≈0.186 OK).
MIN_ITEM_VARIANCE = 0.02
MIN_ITEM_PROPORTION = 0.10

N_QUAD = 15
N_BOOT_LRT = 40
SEED = 42
K_GRID = (2, 3, 4)


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    return expit(np.clip(x, -30, 30))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_p1_family(family: str) -> pd.DataFrame:
    parts = []
    for path in sorted(DER.glob(f"{family}_P1_*rescored.csv")):
        if "review" in path.name.lower():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "included" not in df.columns:
            continue
        df = df[_is_true(df["included"])].copy()
        df = filter_excluded(df, family=family)
        df["variant"] = df["variant_type"].map(normalize_variant)
        ok = df["rescored_correct"] if "rescored_correct" in df.columns else df.get("verified", "")
        df["y"] = _is_true(ok).astype(float)
        df["model"] = df["model"].map(PAPER_MODELS)
        df = df[df["model"].isin(PAPER_MODELS.values())]
        df = df[df["variant"].isin(VARIANTS)]
        df["problem_id"] = df["problem_id"].astype(str).str.strip()
        parts.append(df[["problem_id", "variant", "model", "y"]])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        ["problem_id", "variant", "model"], keep="last"
    )


def build_response_matrix(long: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Persons = (model, problem_id); items = variants. Complete-case rows only."""
    for cols in VARIANT_SETS:
        wide = long.pivot_table(
            index=["model", "problem_id"], columns="variant", values="y", aggfunc="first"
        )
        for c in cols:
            if c not in wide.columns:
                wide[c] = np.nan
        sub = wide[cols].dropna(axis=0, how="any")
        if len(sub) >= 30 and len(cols) >= 4:
            meta = sub.index.to_frame(index=False)
            X = sub.to_numpy(dtype=float)
            return X, meta, cols
    raise RuntimeError("Could not build a complete-case response matrix (n>=30, J>=4).")


# ---------------------------------------------------------------------------
# Identifiability (O10)
# ---------------------------------------------------------------------------

def identifiability_check() -> pd.DataFrame:
    o10 = pd.read_csv(DER / "O10_variance_components.csv")
    o10["is_primary"] = o10["is_primary"].astype(str).str.lower().isin({"true", "1"})
    rows = []
    for fam in ("ALGO", "BW", "GSM"):
        sub = o10[(o10["family"] == fam) & o10["is_primary"] & (o10["component"] == "item")]
        if sub.empty:
            # fall back to any item row
            sub = o10[(o10["family"] == fam) & (o10["component"] == "item")]
        if sub.empty:
            rows.append(
                {
                    "family": fam,
                    "o10_design": "",
                    "item_variance": float("nan"),
                    "item_proportion": float("nan"),
                    "n_items_o10": float("nan"),
                    "threshold_variance": MIN_ITEM_VARIANCE,
                    "threshold_proportion": MIN_ITEM_PROPORTION,
                    "identified": False,
                    "decision": "no_o10_item_row",
                    "empirical_problem_p_sd": float("nan"),
                }
            )
            continue
        r = sub.iloc[0]
        var = float(r["variance"])
        prop = float(r["proportion"])
        identified = (var >= MIN_ITEM_VARIANCE) or (prop >= MIN_ITEM_PROPORTION)
        # Secondary: empirical SD of problem-level accuracy
        long = load_p1_family(fam)
        p_sd = float(long.groupby("problem_id")["y"].mean().std()) if len(long) else float("nan")
        rows.append(
            {
                "family": fam,
                "o10_design": str(r.get("design", "")),
                "item_variance": var,
                "item_proportion": prop,
                "n_items_o10": int(r.get("n_items", 0)),
                "threshold_variance": MIN_ITEM_VARIANCE,
                "threshold_proportion": MIN_ITEM_PROPORTION,
                "identified": bool(identified),
                "decision": "fit_mixture_irt" if identified else "skip_irt_not_identified",
                "empirical_problem_p_sd": p_sd,
                "note": (
                    "O10 primary item variance near zero → difficulty parameters "
                    "not identified for this bank (DS-01 caveat)."
                    if not identified
                    else "Passes O10 item-heterogeneity gate."
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quadrature helpers
# ---------------------------------------------------------------------------

def gh_nodes(n: int = N_QUAD) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.hermite.hermgauss(n)
    # transform for N(0,1): θ = √2 x, weight ∝ w/√π
    theta = x * np.sqrt(2.0)
    weights = w / np.sqrt(np.pi)
    return theta, weights


def pattern_ll_matrix(Y: np.ndarray, b: np.ndarray, theta: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return (N,) marginal log-likelihoods for Rasch difficulties b (vectorized)."""
    # Y: N x J, theta: Q, b: J
    # logits[q,j] = theta[q] - b[j]
    logits = theta[:, None] - b[None, :]  # Q x J
    p = np.clip(_sigmoid_stable(logits), 1e-12, 1 - 1e-12)
    log_p = np.log(p)
    log_q = np.log(1 - p)
    # ll_q[i,q] = sum_j y_ij log p_qj + (1-y) log(1-p)
    # = Y @ log_p.T + (1-Y) @ log_q.T
    ll_q = Y @ log_p.T + (1.0 - Y) @ log_q.T  # N x Q
    return logsumexp(np.log(weights)[None, :] + ll_q, axis=1)


def person_loglik_rasch(y: np.ndarray, b: np.ndarray, theta: np.ndarray, weights: np.ndarray) -> float:
    return float(pattern_ll_matrix(y[None, :], b, theta, weights)[0])


# ---------------------------------------------------------------------------
# Baseline 1PL / 2PL (girth)
# ---------------------------------------------------------------------------

def fit_baseline_irt(Y: np.ndarray, item_names: list[str], family: str) -> tuple[pd.DataFrame, dict]:
    """girth expects items x persons."""
    dataset = np.asarray(Y.T, dtype=int)  # J x N
    # girth uses 1/0; tag all observed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one = onepl_mml(dataset)
        two = twopl_mml(dataset)
    # ability EAP under 2PL for reporting n
    try:
        theta = ability_eap(dataset, two["Difficulty"], two.get("Discrimination"))
    except Exception:
        theta = np.full(Y.shape[0], np.nan)

    rows = []
    disc_1 = float(one.get("Discrimination", 1.0)) if np.ndim(one.get("Discrimination", 1.0)) == 0 else None
    for j, name in enumerate(item_names):
        rows.append(
            {
                "family": family,
                "model_type": "1PL",
                "item": name,
                "difficulty": float(np.asarray(one["Difficulty"]).ravel()[j]),
                "discrimination": float(disc_1) if disc_1 is not None else float(np.asarray(one["Discrimination"]).ravel()[j]),
            }
        )
        rows.append(
            {
                "family": family,
                "model_type": "2PL",
                "item": name,
                "difficulty": float(np.asarray(two["Difficulty"]).ravel()[j]),
                "discrimination": float(np.asarray(two["Discrimination"]).ravel()[j]),
            }
        )
    meta = {
        "n_persons": int(Y.shape[0]),
        "n_items": int(Y.shape[1]),
        "theta_sd_2pl": float(np.nanstd(theta)),
        "onepl_keys": list(one.keys()),
        "twopl_keys": list(two.keys()),
    }
    return pd.DataFrame(rows), meta


def baseline_marginal_ll(Y: np.ndarray, kind: str, item_params: pd.DataFrame, family: str) -> float:
    sub = item_params[(item_params.family == family) & (item_params.model_type == kind)]
    b = sub["difficulty"].to_numpy(dtype=float)
    theta, w = gh_nodes()
    if kind == "1PL":
        return float(pattern_ll_matrix(Y, b, theta, w).sum())
    # 2PL: σ(a(θ-b))
    a = sub["discrimination"].to_numpy(dtype=float)
    N = Y.shape[0]
    total = 0.0
    for i in range(N):
        y = Y[i]
        logits = a[None, :] * (theta[:, None] - b[None, :])
        p = np.clip(_sigmoid_stable(logits), 1e-12, 1 - 1e-12)
        ll_q = (y[None, :] * np.log(p) + (1 - y[None, :]) * np.log(1 - p)).sum(axis=1)
        total += float(logsumexp(np.log(w) + ll_q))
    return total


# ---------------------------------------------------------------------------
# Mixture Rasch (Rost / Mislevy–Verhelst style)
# ---------------------------------------------------------------------------

def _init_mixture(Y: np.ndarray, K: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    N, J = Y.shape
    # k-means-ish on response patterns
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=K, n_init=10, random_state=int(rng.integers(0, 1_000_000)))
    lab = km.fit_predict(Y)
    pi = np.array([(lab == c).mean() for c in range(K)], dtype=float)
    pi = np.clip(pi, 1e-3, None)
    pi /= pi.sum()
    B = np.zeros((K, J))
    for c in range(K):
        idx = lab == c
        if idx.sum() == 0:
            p = Y.mean(axis=0)
        else:
            p = Y[idx].mean(axis=0)
        p = np.clip(p, 0.05, 0.95)
        # b = -logit(p) under θ=0 center; then center
        B[c] = -np.log(p / (1 - p))
        B[c] -= B[c].mean()
    return pi, B


def _rasch_mstep_moment(Y: np.ndarray, gamma_c: np.ndarray) -> np.ndarray:
    """Fast difficulty update: weighted mean P → centered logit difficulties."""
    w = gamma_c + 1e-12
    p = (w[:, None] * Y).sum(axis=0) / w.sum()
    p = np.clip(p, 0.02, 0.98)
    b = -np.log(p / (1.0 - p))
    return b - b.mean()


def mixture_rasch_em(
    Y: np.ndarray,
    K: int,
    *,
    max_iter: int = 120,
    tol: float = 1e-5,
    seed: int = SEED,
    n_restarts: int = 3,
    refine_bfgs: bool = False,
) -> dict:
    theta, weights = gh_nodes()
    rng = np.random.default_rng(seed)
    best = None
    N, J = Y.shape

    for restart in range(n_restarts):
        pi, B = _init_mixture(Y, K, rng)
        ll_prev = -np.inf
        gamma = np.full((N, K), 1.0 / K)
        it = 0
        for it in range(max_iter):
            # E-step
            log_comp = np.empty((N, K))
            for c in range(K):
                log_comp[:, c] = np.log(pi[c] + 1e-300) + pattern_ll_matrix(
                    Y, B[c], theta, weights
                )
            log_marg = logsumexp(log_comp, axis=1)
            gamma = np.exp(log_comp - log_marg[:, None])
            ll = float(log_marg.sum())

            # M-step
            pi = gamma.mean(axis=0)
            pi = np.clip(pi, 1e-4, None)
            pi /= pi.sum()
            for c in range(K):
                B[c] = _rasch_mstep_moment(Y, gamma[:, c])

            if abs(ll - ll_prev) < tol * (1 + abs(ll)):
                break
            ll_prev = ll

        # Optional final polish (full-data fits only)
        if refine_bfgs:
            for c in range(K):
                w_c = gamma[:, c]

                def nll_sum0(b_free: np.ndarray, w_c=w_c) -> float:
                    b_full = np.zeros(J)
                    b_full[:-1] = b_free
                    b_full[-1] = -b_free.sum()
                    ll_i = pattern_ll_matrix(Y, b_full, theta, weights)
                    return -float(np.dot(w_c, ll_i))

                x0 = B[c, :-1].copy()
                res = minimize(nll_sum0, x0, method="L-BFGS-B", options={"maxiter": 25})
                b_full = np.zeros(J)
                b_full[:-1] = res.x
                b_full[-1] = -res.x.sum()
                B[c] = b_full
            log_comp = np.empty((N, K))
            for c in range(K):
                log_comp[:, c] = np.log(pi[c] + 1e-300) + pattern_ll_matrix(
                    Y, B[c], theta, weights
                )
            ll = float(logsumexp(log_comp, axis=1).sum())
            gamma = np.exp(log_comp - logsumexp(log_comp, axis=1)[:, None])

        n_params = (K - 1) + K * (J - 1)
        bic = -2 * ll + n_params * np.log(N)
        aic = -2 * ll + 2 * n_params
        cand = {
            "K": K,
            "pi": pi,
            "B": B,
            "ll": ll,
            "bic": bic,
            "aic": aic,
            "n_params": n_params,
            "n": N,
            "J": J,
            "gamma": gamma,
            "converged_iter": it,
            "restart": restart,
        }
        if best is None or cand["ll"] > best["ll"]:
            best = cand
    return best


def mixture_posteriors(Y: np.ndarray, fit: dict) -> np.ndarray:
    theta, weights = gh_nodes()
    N, K = Y.shape[0], fit["K"]
    log_comp = np.empty((N, K))
    for c in range(K):
        log_comp[:, c] = np.log(fit["pi"][c] + 1e-300) + pattern_ll_matrix(
            Y, fit["B"][c], theta, weights
        )
    log_marg = logsumexp(log_comp, axis=1)
    return np.exp(log_comp - log_marg[:, None])


def simulate_from_fit(fit: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    theta_nodes, _ = gh_nodes()
    K, J = fit["K"], fit["J"]
    Y = np.zeros((n, J))
    classes = rng.choice(K, size=n, p=fit["pi"])
    # sample θ ~ N(0,1)
    thetas = rng.normal(0, 1, size=n)
    for i in range(n):
        c = classes[i]
        p = _sigmoid_stable(thetas[i] - fit["B"][c])
        Y[i] = rng.binomial(1, p)
    return Y


def bootstrap_lrt(
    Y: np.ndarray,
    fit_null: dict,
    fit_alt: dict,
    *,
    n_boot: int = N_BOOT_LRT,
    seed: int = SEED,
) -> dict:
    """Parametric bootstrap LRT for H0: K_null vs H1: K_alt (K_alt = K_null+1 typically)."""
    lr_obs = 2.0 * (fit_alt["ll"] - fit_null["ll"])
    rng = np.random.default_rng(seed)
    N = Y.shape[0]
    boots = []
    for b in range(n_boot):
        Yb = simulate_from_fit(fit_null, N, rng)
        try:
            f0 = mixture_rasch_em(Yb, fit_null["K"], seed=seed + 10 + b, n_restarts=1, max_iter=60)
            f1 = mixture_rasch_em(Yb, fit_alt["K"], seed=seed + 1000 + b, n_restarts=1, max_iter=60)
            boots.append(2.0 * (f1["ll"] - f0["ll"]))
        except Exception:
            continue
    boots_a = np.asarray(boots, dtype=float)
    if len(boots_a) == 0:
        p = float("nan")
    else:
        p = float((np.sum(boots_a >= lr_obs) + 1) / (len(boots_a) + 1))
    return {
        "lr_obs": lr_obs,
        "p_boot": p,
        "n_boot_ok": int(len(boots_a)),
        "lr_boot_mean": float(np.mean(boots_a)) if len(boots_a) else float("nan"),
        "lr_boot_p95": float(np.percentile(boots_a, 95)) if len(boots_a) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Class profiles + labels
# ---------------------------------------------------------------------------

def characterize_classes(Y: np.ndarray, gamma: np.ndarray, item_names: list[str], fit: dict) -> pd.DataFrame:
    K = fit["K"]
    rows = []
    map_c = gamma.argmax(axis=1)
    for c in range(K):
        w = gamma[:, c]
        # soft profile
        soft = np.average(Y, axis=0, weights=w + 1e-12)
        hard = Y[map_c == c].mean(axis=0) if np.any(map_c == c) else soft
        profile = {item_names[j]: float(soft[j]) for j in range(len(item_names))}
        p_can = profile.get("canonical", float("nan"))
        p_w3 = profile.get("W3", float("nan"))
        p_w6 = profile.get("W6", float("nan"))
        p_surf = np.nanmean([profile.get(v, np.nan) for v in ("W1", "W2", "W4")])
        # labels (mutually preferential)
        if np.isfinite(p_w3) and (p_can - p_w3) >= 0.20:
            label = "w3_collapse"
        elif np.isfinite(p_w6) and (p_can - p_w6) >= 0.20:
            label = "w6_collapse"
        elif np.isfinite(p_surf) and p_surf >= 0.80 and (not np.isfinite(p_w3) or p_w3 >= 0.70):
            label = "surface_invariant"
        elif np.isfinite(p_can) and p_can < 0.40:
            label = "low_accuracy"
        else:
            label = "mixed_other"
        rows.append(
            {
                "class_id": c,
                "class_label": label,
                "mixing_weight": float(fit["pi"][c]),
                "n_map": int((map_c == c).sum()),
                "mean_posterior": float(w.mean()),
                **{f"p_{k}": v for k, v in profile.items()},
                **{f"hard_p_{item_names[j]}": float(hard[j]) for j in range(len(item_names))},
                **{f"diff_{item_names[j]}": float(fit["B"][c, j]) for j in range(len(item_names))},
            }
        )
    return pd.DataFrame(rows)


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    mask = (a != None) & (b != None)  # noqa: E711
    # also drop nan-like
    a = a[mask]
    b = b[mask]
    if len(a) < 2:
        return float("nan")
    labels = sorted(set(a) | set(b))
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for x, y in zip(a, b):
        mat[idx[x], idx[y]] += 1
    n = mat.sum()
    if n == 0:
        return float("nan")
    po = np.trace(mat) / n
    pe = (mat.sum(0) * mat.sum(1)).sum() / (n * n)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else float("nan")
    return float((po - pe) / (1 - pe))


def map_class_to_strategy(class_label: str) -> str:
    if class_label == "w3_collapse":
        return "retrieval"
    if class_label in {"surface_invariant"}:
        return "computation"
    if class_label == "w6_collapse":
        return "retrieval"  # param regeneration collapse also retrieval-ish
    if class_label == "low_accuracy":
        return "ambiguous"
    return "mixed"


def o12_and_tri_kappa(
    post: pd.DataFrame,
    class_profiles: pd.DataFrame,
    family: str,
    algo_identified: bool,
) -> pd.DataFrame:
    rows = []
    label_map = {
        int(r.class_id): map_class_to_strategy(str(r.class_label))
        for _, r in class_profiles.iterrows()
    }
    post = post.copy()
    post["strategy_mapped"] = post["map_class"].map(label_map)

    # O12 (ALGO only)
    o12_path = DER / "O12_label_stability.csv"
    if not algo_identified:
        rows.append(
            {
                "comparison": "mixture_MAP_vs_O12_modal",
                "family": "ALGO",
                "n": 0,
                "kappa": float("nan"),
                "note": "ALGO failed IRT identifiability; O12 kappa not computed (expected gate).",
            }
        )
    elif family == "ALGO" and o12_path.exists():
        o12 = pd.read_csv(o12_path, dtype=str)
        o12["model_short"] = o12["model"].map(PAPER_MODELS).fillna(o12["model"])
        # map O12 labels to coarse
        coarse = {
            "retrieval": "retrieval",
            "computation": "computation",
            "mixed": "mixed",
            "ambiguous": "ambiguous",
        }
        o12["o12_coarse"] = o12["modal_label"].map(coarse)
        m = post.merge(
            o12[["model_short", "problem_id", "o12_coarse"]],
            left_on=["model", "problem_id"],
            right_on=["model_short", "problem_id"],
            how="inner",
        )
        # drop ambiguous from kappa or keep
        m2 = m[m["o12_coarse"].isin(["retrieval", "computation", "mixed"])]
        m2 = m2[m2["strategy_mapped"].isin(["retrieval", "computation", "mixed"])]
        rows.append(
            {
                "comparison": "mixture_MAP_vs_O12_modal",
                "family": "ALGO",
                "n": int(len(m2)),
                "kappa": cohens_kappa(m2["strategy_mapped"].to_numpy(), m2["o12_coarse"].to_numpy()),
                "note": "Near-zero expected if threshold triangulation ≠ model-based classes.",
            }
        )
    else:
        rows.append(
            {
                "comparison": "mixture_MAP_vs_O12_modal",
                "family": "ALGO",
                "n": 0,
                "kappa": float("nan"),
                "note": "Mixture fit family is not ALGO; O12 is ALGO-only.",
            }
        )

    # triangulation_v2 on same family
    v2_path = DER / "triangulation_v2_labels.csv"
    if v2_path.exists():
        v2 = pd.read_csv(v2_path, dtype=str)
        v2 = v2[v2["family"].astype(str).str.upper() == family]
        v2["model"] = v2["model"].map(lambda m: PAPER_MODELS.get(m, m))
        # collapse weak_* 
        def crush(lab: str) -> str:
            lab = str(lab).lower()
            if "retrieval" in lab:
                return "retrieval"
            if "computation" in lab:
                return "computation"
            if lab == "mixed":
                return "mixed"
            return "other"

        v2["tri_coarse"] = v2["tri_v2_label"].map(crush)
        m = post.merge(
            v2[["model", "problem_id", "tri_coarse"]],
            on=["model", "problem_id"],
            how="inner",
        )
        m2 = m[m["tri_coarse"].isin(["retrieval", "computation", "mixed"])]
        m2 = m2[m2["strategy_mapped"].isin(["retrieval", "computation", "mixed"])]
        rows.append(
            {
                "comparison": "mixture_MAP_vs_triangulation_v2",
                "family": family,
                "n": int(len(m2)),
                "kappa": cohens_kappa(m2["strategy_mapped"].to_numpy(), m2["tri_coarse"].to_numpy()),
                "note": "Secondary cross-check on the fitted family.",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_coding_note(variant_items: list[str], family: str) -> None:
    text = f"""# C2 / DS-01 IRT coding

## Person / item definition

- **Person (respondent):** `(model, problem_id)` instance from rescored Probe-1.
- **Item (IRT item):** variant correctness indicators: {variant_items}.
- **Response:** binary `rescored_correct` (1=correct, 0=incorrect), complete-case rows only.

This treats variant surfaces as the measurement instrument for *strategy*, matching
Mislevy & Verhelst (strategy classes with class-specific item parameters). It is
**not** a problem-bank difficulty ranking across `{family}` items; that ranking is
blocked when O10 item variance is near zero (see `C2_identifiability_check.csv`).

## Models

- **1PL / 2PL baseline:** `girth.onepl_mml` / `girth.twopl_mml` on the person×variant matrix.
- **Mixture Rasch (K=2,3,4):** Rost / Mislevy–Verhelst style — class mixing weights `π_c`,
  class-specific Rasch difficulties `b_{{c,j}}` (sum-to-zero within class), latent ability
  `θ ~ N(0,1)` integrated by Gauss–Hermite quadrature (Q={N_QUAD}).
- **Selection:** minimum BIC among mixture K∈{1,2,3,4} (K=1 is the single-class Rasch null).
  Bootstrap LRT (B={N_BOOT_LRT}) for adjacent increments as a second criterion.
  If BIC and LRT disagree, **BIC is primary**; disagreement is reported in `C2_irt_fit_comparison.csv`.
  Do not prefer a larger K for narrative fit.

## Class labels (post-hoc profile rules)

Applied after fit; not used in estimation:

- `w3_collapse`: P(canonical)−P(W3) ≥ 0.20
- `w6_collapse`: P(canonical)−P(W6) ≥ 0.20
- `surface_invariant`: mean(W1,W2,W4) ≥ 0.80 and P(W3) ≥ 0.70
- `low_accuracy`: P(canonical) < 0.40
- else `mixed_other`
"""
    OUT_CODING.write_text(text, encoding="utf-8")


def fit_family(family: str, long: pd.DataFrame) -> dict:
    Y, meta, items = build_response_matrix(long)
    print(f"[{family}] matrix N={Y.shape[0]} J={Y.shape[1]} items={items}")
    write_coding_note(items, family)

    item_params, base_meta = fit_baseline_irt(Y, items, family)
    ll_1 = baseline_marginal_ll(Y, "1PL", item_params, family)
    ll_2 = baseline_marginal_ll(Y, "2PL", item_params, family)
    n, J = Y.shape
    # 1PL: J-1 free difficulties (+ discrimination fixed)
    bic_1 = -2 * ll_1 + (J - 1) * np.log(n)
    # 2PL: J discriminations + J-1 difficulties (location)
    bic_2 = -2 * ll_2 + (J + J - 1) * np.log(n)

    fit_rows = [
        {
            "family": family,
            "model": "1PL",
            "K": 1,
            "ll": ll_1,
            "bic": bic_1,
            "aic": -2 * ll_1 + 2 * (J - 1),
            "n_params": J - 1,
            "n_persons": n,
            "n_items": J,
            "selected": False,
            "lrt_vs_prev_p": float("nan"),
            "note": "baseline single-class Rasch/1PL",
        },
        {
            "family": family,
            "model": "2PL",
            "K": 1,
            "ll": ll_2,
            "bic": bic_2,
            "aic": -2 * ll_2 + 2 * (J + J - 1),
            "n_params": J + J - 1,
            "n_persons": n,
            "n_items": J,
            "selected": False,
            "lrt_vs_prev_p": float("nan"),
            "note": "baseline 2PL",
        },
    ]

    fits = {}
    for K in K_GRID:
        print(f"[{family}] mixture Rasch K={K} ...")
        fits[K] = mixture_rasch_em(Y, K, seed=SEED, n_restarts=5, refine_bfgs=True)
        fr = fits[K]
        fit_rows.append(
            {
                "family": family,
                "model": "mixture_rasch",
                "K": K,
                "ll": fr["ll"],
                "bic": fr["bic"],
                "aic": fr["aic"],
                "n_params": fr["n_params"],
                "n_persons": n,
                "n_items": J,
                "selected": False,
                "lrt_vs_prev_p": float("nan"),
                "note": f"restarts=4; converged_iter~{fr['converged_iter']}",
            }
        )

    # Bootstrap LRT for 2vs1 (1 = single Rasch via K=1 mixture), 3vs2, 4vs3
    print(f"[{family}] fitting K=1 mixture for LRT null ...")
    fit1 = mixture_rasch_em(Y, 1, seed=SEED, n_restarts=4, refine_bfgs=True)
    fit_rows.append(
        {
            "family": family,
            "model": "mixture_rasch",
            "K": 1,
            "ll": fit1["ll"],
            "bic": fit1["bic"],
            "aic": fit1["aic"],
            "n_params": fit1["n_params"],
            "n_persons": n,
            "n_items": J,
            "selected": False,
            "lrt_vs_prev_p": float("nan"),
            "note": "single-class mixture Rasch (LRT null)",
        }
    )
    fits[1] = fit1

    lrt_p = {}
    pairs = [(1, 2), (2, 3), (3, 4)]
    for k0, k1 in pairs:
        print(f"[{family}] bootstrap LRT K={k0} vs K={k1} (B={N_BOOT_LRT}) ...")
        res = bootstrap_lrt(Y, fits[k0], fits[k1], n_boot=N_BOOT_LRT, seed=SEED)
        lrt_p[k1] = res["p_boot"]
        for row in fit_rows:
            if row["model"] == "mixture_rasch" and row["K"] == k1:
                row["lrt_vs_prev_p"] = res["p_boot"]
                row["lrt_obs"] = res["lr_obs"]
                row["lrt_n_boot"] = res["n_boot_ok"]

    # Select K by BIC among mixture K=1..4 (primary). LRT path reported separately.
    mix_rows = [r for r in fit_rows if r["model"] == "mixture_rasch" and r["K"] in (1,) + K_GRID]
    best_bic = min(mix_rows, key=lambda r: r["bic"])
    K_star = int(best_bic["K"])

    # LRT path: start at 1, accept K+1 if p<0.05
    K_lrt = 1
    for k1 in (2, 3, 4):
        p = lrt_p.get(k1, 1.0)
        if np.isfinite(p) and p < 0.05:
            K_lrt = k1
        else:
            break

    for row in fit_rows:
        if row["model"] == "mixture_rasch" and row["K"] == K_star:
            row["selected"] = True
            disagree = ""
            if K_lrt != K_star:
                disagree = f" BIC≠LRT(LRT→{K_lrt});"
            row["note"] = (
                f"SELECTED by BIC (K∈1..4);{disagree} LRT-path K={K_lrt}; "
                + str(row.get("note", ""))
            )

    # Posteriors + profiles for K*
    fit_star = fits[K_star]
    gamma = mixture_posteriors(Y, fit_star)
    profiles = characterize_classes(Y, gamma, items, fit_star)
    profiles.insert(0, "family", family)
    profiles.insert(1, "K", K_star)

    post_rows = []
    map_c = gamma.argmax(axis=1)
    label_by_c = {
        int(r.class_id): str(r.class_label) for _, r in profiles.iterrows()
    }
    for i in range(len(meta)):
        row = {
            "family": family,
            "model": meta.iloc[i]["model"],
            "problem_id": meta.iloc[i]["problem_id"],
            "K": K_star,
            "map_class": int(map_c[i]),
            "map_class_label": label_by_c.get(int(map_c[i]), ""),
            "map_prob": float(gamma[i, map_c[i]]),
        }
        for c in range(K_star):
            row[f"p_class_{c}"] = float(gamma[i, c])
        post_rows.append(row)
    post = pd.DataFrame(post_rows)

    return {
        "item_params": item_params,
        "fit_rows": fit_rows,
        "profiles": profiles,
        "posteriors": post,
        "K_bic": K_star,
        "K_lrt": K_lrt,
        "items": items,
        "base_meta": base_meta,
        "Y": Y,
        "meta": meta,
    }


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    print("[C2] identifiability check (O10 item variance) ...")
    id_df = identifiability_check()
    id_df.to_csv(OUT_ID, index=False)
    print(id_df[["family", "item_variance", "item_proportion", "identified", "decision"]].to_string(index=False))

    algo_ok = bool(id_df.loc[id_df.family == "ALGO", "identified"].iloc[0])
    pass_fams = id_df.loc[id_df["identified"], "family"].tolist()
    if not pass_fams:
        print("No family passed identifiability; writing empty strategy outputs.")
        pd.DataFrame(id_df).to_csv(OUT_ID, index=False)
        pd.DataFrame().to_csv(OUT_FIT, index=False)
        pd.DataFrame().to_csv(OUT_PROF, index=False)
        pd.DataFrame().to_csv(OUT_POST, index=False)
        pd.DataFrame(
            [
                {
                    "comparison": "mixture_MAP_vs_O12_modal",
                    "family": "ALGO",
                    "n": 0,
                    "kappa": float("nan"),
                    "note": "No family identified; O12 kappa N/A.",
                }
            ]
        ).to_csv(OUT_KAPPA, index=False)
        return

    all_fit_rows = []
    all_profiles = []
    all_posts = []
    all_items = []
    kappa_parts = []

    for fam in pass_fams:
        long = load_p1_family(fam)
        if long.empty:
            print(f"[{fam}] no P1 data; skip")
            continue
        result = fit_family(fam, long)
        all_fit_rows.extend(result["fit_rows"])
        all_profiles.append(result["profiles"])
        all_posts.append(result["posteriors"])
        all_items.append(result["item_params"])
        kappa_parts.append(
            o12_and_tri_kappa(result["posteriors"], result["profiles"], fam, algo_ok)
        )
        print(
            f"[{fam}] selected K_BIC={result['K_bic']} K_LRT_path={result['K_lrt']}"
        )

    # Failed families: stub rows in fit comparison
    for _, r in id_df.iterrows():
        if not r["identified"]:
            all_fit_rows.append(
                {
                    "family": r["family"],
                    "model": "NOT_FIT",
                    "K": "",
                    "ll": float("nan"),
                    "bic": float("nan"),
                    "aic": float("nan"),
                    "n_params": "",
                    "n_persons": "",
                    "n_items": "",
                    "selected": False,
                    "lrt_vs_prev_p": float("nan"),
                    "note": (
                        f"Skipped: O10 item_var={r['item_variance']:.4f} "
                        f"prop={r['item_proportion']:.4f} below gate "
                        f"(var>={MIN_ITEM_VARIANCE} or prop>={MIN_ITEM_PROPORTION})."
                    ),
                }
            )

    pd.DataFrame(all_fit_rows).to_csv(OUT_FIT, index=False)
    if all_profiles:
        pd.concat(all_profiles, ignore_index=True).to_csv(OUT_PROF, index=False)
    else:
        pd.DataFrame().to_csv(OUT_PROF, index=False)
    if all_posts:
        pd.concat(all_posts, ignore_index=True).to_csv(OUT_POST, index=False)
    else:
        pd.DataFrame().to_csv(OUT_POST, index=False)
    if all_items:
        pd.concat(all_items, ignore_index=True).to_csv(OUT_ITEMS, index=False)
    if kappa_parts:
        pd.concat(kappa_parts, ignore_index=True).to_csv(OUT_KAPPA, index=False)

    print(f"[write] {OUT_ID}")
    print(f"[write] {OUT_FIT}")
    print(f"[write] {OUT_PROF}")
    print(f"[write] {OUT_POST}")
    print(f"[write] {OUT_ITEMS}")
    print(f"[write] {OUT_KAPPA}")


if __name__ == "__main__":
    main()
