#!/usr/bin/env python3
"""O14: Controlled Blocksworld naming intervention (A sequential / B scattered / C indexed).

Pre-registered interpretation (do not revise after seeing results):
  - If naming moves accuracy substantially → primary finding; L1 partly lexical.
  - If naming does not → L1 survives the confound with a controlled n=120 experiment.
Both outcomes are reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.behavioral.sampling import DEFAULT_TEMPERATURE  # noqa: E402
from probes.common.cluster_inference import bootstrap_p_two_sided  # noqa: E402
from probes.contamination.bw_instance_metrics import naming_is_sequential  # noqa: E402
from probes.contamination.verify import verify_answer  # noqa: E402
from scripts.generation.utils.variant_utils import (  # noqa: E402
    apply_mapping,
    fd_plan_to_bw_format,
    generate_random_bw_pddl,
    make_inverse_mapping,
    pddl_to_natural_language,
    run_fast_downward,
)

DER = REPO_ROOT / "results" / "derived"
BANK_CSV = REPO_ROOT / "data" / "problems" / "question_bank_bw.csv"
BANK_OUT = DER / "O14_naming_bank.jsonl"
RESULTS_OUT = DER / "O14_naming_results.csv"
ANALYSIS_OUT = DER / "O14_naming_analysis.csv"

N_PAIRS = 120
GEN_SEED = 14_014
EVAL_SEED = 14_014
N_BOOT = 5000

MODELS = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.1-8b-instruct",
    "openai/o4-mini",
    "deepseek/deepseek-r1-distill-llama-70b",
]
MODEL_SHORT = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
    "deepseek/deepseek-r1-distill-llama-70b": "DeepSeek",
}

ARM_A = "A_sequential"
ARM_B = "B_scattered"
ARM_C = "C_indexed"

# Split points for mapping only variable content (boilerplate has standalone "a").
_STATE_MARK = "Current state: "
_GOAL_MARK = " Goal: "
_RESPOND_MARK = " Respond with a numbered list"


# ---------------------------------------------------------------------------
# Bank helpers
# ---------------------------------------------------------------------------


def _canonical_num_blocks_dist() -> list[int]:
    """Empirical num_blocks from FD-ok canonical BW items, scaled to N_PAIRS."""
    from probes.contamination.bw_instance_metrics import extract_bw_metrics

    bank = pd.read_csv(BANK_CSV, dtype=str).fillna("")
    can = bank[bank["variant_type"].str.strip().str.lower().eq("canonical")]
    counts: Counter[int] = Counter()
    for _, row in can.iterrows():
        m = extract_bw_metrics(row["problem_text"], row["problem_id"])
        n = int(m.get("num_blocks") or 0)
        fd = m.get("fd_optimal_plan_length")
        if n >= 3 and fd is not None and str(m.get("fd_status", "")).lower() == "ok":
            counts[n] += 1
    if not counts:
        raise RuntimeError("No FD-ok canonical BW items found for size distribution")

    # Largest-remainder allocation to exactly N_PAIRS.
    total = sum(counts.values())
    raw = {k: counts[k] / total * N_PAIRS for k in sorted(counts)}
    floors = {k: int(np.floor(v)) for k, v in raw.items()}
    rem = N_PAIRS - sum(floors.values())
    frac_order = sorted(raw.keys(), key=lambda k: raw[k] - floors[k], reverse=True)
    for k in frac_order[:rem]:
        floors[k] += 1
    sizes: list[int] = []
    for k in sorted(floors):
        sizes.extend([k] * floors[k])
    assert len(sizes) == N_PAIRS
    return sizes


def _bank_letter_weights() -> dict[str, float]:
    """Letter frequencies from non-sequential canonical gold plans (a–l style)."""
    from probes.contamination.bw_instance_metrics import extract_bw_metrics

    bank = pd.read_csv(BANK_CSV, dtype=str).fillna("")
    can = bank[bank["variant_type"].str.strip().str.lower().eq("canonical")]
    letter_counts: Counter[str] = Counter()
    for _, row in can.iterrows():
        m = extract_bw_metrics(row["problem_text"], row["problem_id"])
        if m.get("naming_is_sequential") or int(m.get("num_blocks") or 0) < 3:
            continue
        for tok in re.findall(
            r"(?:pick-up|put-down|stack|unstack)\s+([a-z])(?:\s+([a-z]))?",
            str(row.get("correct_answer", "")).lower(),
        ):
            for t in tok:
                if t:
                    letter_counts[t] += 1
    if not letter_counts:
        # Fallback uniform over a–l (bank style).
        return {chr(ord("a") + i): 1.0 for i in range(12)}
    return {k: float(v) for k, v in letter_counts.items()}


def _sample_scattered(n: int, rng: np.random.Generator, weights: dict[str, float]) -> list[str]:
    """Sample n distinct letters; reject sequential a.. sets.

    Uses a–z with empirical bank weights on observed letters and a small
    floor on the rest so n=12 can avoid the unique sequential set {a..l}.
    """
    alphabet = [chr(ord("a") + i) for i in range(26)]
    floor = 0.05 * (min(weights.values()) if weights else 1.0)
    base_w = np.array([max(weights.get(ch, 0.0), floor) for ch in alphabet], dtype=float)

    for _ in range(10_000):
        chosen: list[str] = []
        avail = list(alphabet)
        avail_w = base_w.copy()
        for _i in range(n):
            avail_w = avail_w / avail_w.sum()
            idx = int(rng.choice(len(avail), p=avail_w))
            chosen.append(avail.pop(idx))
            avail_w = np.delete(avail_w, idx)
        if not naming_is_sequential(set(chosen)):
            return chosen
    raise RuntimeError(f"Failed to sample non-sequential letter set of size {n}")


def _split_prompt(text: str) -> tuple[str, str, str, str]:
    """Return (prefix, state, goal, suffix) for safe ID remapping."""
    i_state = text.find(_STATE_MARK)
    i_goal = text.find(_GOAL_MARK)
    i_resp = text.find(_RESPOND_MARK)
    if i_state < 0 or i_goal < 0 or i_resp < 0:
        raise ValueError("Prompt missing expected Current state / Goal / Respond markers")
    prefix = text[: i_state + len(_STATE_MARK)]
    state = text[i_state + len(_STATE_MARK) : i_goal]
    goal = text[i_goal + len(_GOAL_MARK) : i_resp]
    suffix = text[i_resp:]
    return prefix, state, goal, suffix


def _remap_prompt(text: str, mapping: dict[str, str]) -> str:
    prefix, state, goal, suffix = _split_prompt(text)
    return prefix + apply_mapping(state, mapping) + _GOAL_MARK + apply_mapping(goal, mapping) + suffix


def assert_only_block_ids_differ(
    text_a: str,
    text_b: str,
    answer_a: str,
    answer_b: str,
    mapping_a_to_b: dict[str, str],
    *,
    label: str,
) -> None:
    """Fail loudly unless B recovers A exactly via inverse ID map (content-safe)."""
    if text_a == text_b:
        raise AssertionError(f"{label}: texts are identical; expected ID change")
    if len(mapping_a_to_b) != len(set(mapping_a_to_b.values())):
        raise AssertionError(f"{label}: mapping is not bijective: {mapping_a_to_b}")

    inv = make_inverse_mapping(mapping_a_to_b)
    prefix_a, state_a, goal_a, suffix_a = _split_prompt(text_a)
    prefix_b, state_b, goal_b, suffix_b = _split_prompt(text_b)
    if prefix_a != prefix_b or suffix_a != suffix_b:
        raise AssertionError(
            f"{label}: boilerplate differs\nA-prefix={prefix_a!r}\nB-prefix={prefix_b!r}"
        )
    recovered_state = apply_mapping(state_b, inv)
    recovered_goal = apply_mapping(goal_b, inv)
    recovered_ans = apply_mapping(answer_b, inv)
    if recovered_state != state_a:
        raise AssertionError(
            f"{label}: state not ID-only\nA={state_a!r}\nrec={recovered_state!r}"
        )
    if recovered_goal != goal_a:
        raise AssertionError(
            f"{label}: goal not ID-only\nA={goal_a!r}\nrec={recovered_goal!r}"
        )
    if recovered_ans != answer_a:
        raise AssertionError(
            f"{label}: answer not ID-only\nA={answer_a!r}\nrec={recovered_ans!r}"
        )
    # Full-text check: only IDs differ inside state/goal.
    rebuilt_a = prefix_a + state_a + _GOAL_MARK + goal_a + suffix_a
    if rebuilt_a != text_a:
        raise AssertionError(f"{label}: internal split failed to rebuild A")


def generate_bank(*, n_pairs: int = N_PAIRS, seed: int = GEN_SEED) -> Path:
    DER.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    py_rng_base = int(seed)
    sizes = _canonical_num_blocks_dist()
    if n_pairs != N_PAIRS:
        # Allow smoke tests: take prefix of scaled dist, or resample.
        sizes = (sizes * ((n_pairs // len(sizes)) + 1))[:n_pairs]
    weights = _bank_letter_weights()

    records: list[dict] = []
    attempt = 0
    pair_idx = 0
    while pair_idx < n_pairs:
        n_blocks = sizes[pair_idx]
        attempt += 1
        gen_seed = py_rng_base + attempt * 9973 + n_blocks * 17
        domain, problem = generate_random_bw_pddl(n_blocks, gen_seed)
        plan, status = run_fast_downward(domain, problem, timeout=120)
        if status != "ok" or not plan:
            continue
        plan_len = len([ln for ln in plan.splitlines() if ln.strip()])
        expected = 2 * (n_blocks - 1)
        if plan_len != expected:
            # Flat→tower should be exactly 2*(n-1); skip anomalies.
            continue

        seq_blocks = [chr(ord("a") + i) for i in range(n_blocks)]
        text_a = pddl_to_natural_language(problem, n_blocks)
        ans_a = fd_plan_to_bw_format(plan)

        # Gold must verify before remapping.
        if not verify_answer(
            f"O14_tmp_A", ans_a, ans_a, "blocksworld", problem_text=text_a
        ):
            continue

        scattered = _sample_scattered(n_blocks, rng, weights)
        map_b = {seq_blocks[i]: scattered[i] for i in range(n_blocks)}
        map_c = {seq_blocks[i]: f"b{i + 1}" for i in range(n_blocks)}

        text_b = _remap_prompt(text_a, map_b)
        ans_b = apply_mapping(ans_a, map_b)
        text_c = _remap_prompt(text_a, map_c)
        ans_c = apply_mapping(ans_a, map_c)

        assert_only_block_ids_differ(
            text_a, text_b, ans_a, ans_b, map_b, label=f"pair{pair_idx:04d}_A_vs_B"
        )
        assert_only_block_ids_differ(
            text_a, text_c, ans_a, ans_c, map_c, label=f"pair{pair_idx:04d}_A_vs_C"
        )

        for arm, text, ans, blocks, mapping in (
            (ARM_A, text_a, ans_a, seq_blocks, {b: b for b in seq_blocks}),
            (ARM_B, text_b, ans_b, scattered, map_b),
            (ARM_C, text_c, ans_c, list(map_c.values()), map_c),
        ):
            ok = verify_answer(
                f"O14_{pair_idx:04d}_{arm}",
                ans,
                ans,
                "blocksworld",
                problem_text=text,
            )
            if not ok:
                raise AssertionError(
                    f"Gold failed verifier for pair={pair_idx} arm={arm}"
                )
            pair_id = f"O14_{pair_idx:04d}"
            records.append(
                {
                    "pair_id": pair_id,
                    "arm": arm,
                    "problem_id": f"{pair_id}_{arm}",
                    "num_blocks": n_blocks,
                    "plan_length": plan_len,
                    "fd_optimal_plan_length": plan_len,
                    "problem_text": text,
                    "correct_answer": ans,
                    "block_names": blocks,
                    "mapping_from_sequential": mapping,
                    "gen_seed": gen_seed,
                    "problem_family": "planning_suite",
                    "problem_subtype": "blocksworld",
                    "variant_type": arm,
                    "contamination_pole": "naming_intervention",
                    "difficulty": str(n_blocks),
                }
            )
        pair_idx += 1
        if pair_idx % 20 == 0:
            print(f"  generated {pair_idx}/{n_pairs} pairs", flush=True)

    with BANK_OUT.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} rows ({n_pairs} pairs × 3 arms) → {BANK_OUT}")
    return BANK_OUT


def load_bank(path: Path = BANK_OUT) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

RESULTS_COLUMNS = [
    "pair_id",
    "arm",
    "problem_id",
    "model",
    "model_short",
    "num_blocks",
    "plan_length",
    "behavioral_correct",
    "raw_response",
    "correct_answer",
    "temperature",
    "seed",
    "max_tokens",
    "decoding_notes",
]


def _existing_eval_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, dtype=str).fillna("")
    if "problem_id" not in df.columns or "model" not in df.columns:
        return set()
    done: set[tuple[str, str]] = set()
    for (pid, model), g in df.groupby(["problem_id", "model"], sort=False):
        raw = str(g.iloc[-1].get("raw_response", ""))
        if raw.startswith("ERROR:"):
            continue
        done.add((str(pid), str(model)))
    return done


def run_eval(
    *,
    dry_run: bool = False,
    resume: bool = True,
    models: list[str] | None = None,
    limit_pairs: int | None = None,
    output_path: Path | None = None,
) -> Path:
    load_dotenv(REPO_ROOT / ".env")
    DER.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_path) if output_path else RESULTS_OUT
    bank = load_bank()
    if limit_pairs is not None:
        keep = {f"O14_{i:04d}" for i in range(limit_pairs)}
        bank = [r for r in bank if r["pair_id"] in keep]

    models = models or list(MODELS)
    done = _existing_eval_keys(out_path) if resume else set()
    write_header = not out_path.exists() or out_path.stat().st_size == 0

    if dry_run:
        from probes.behavioral.mock_client import MockClient

        clients = {m: MockClient(default_response="pick-up a\nstack a b") for m in models}
        print("DRY RUN: MockClient — no API credits")
    else:
        import os

        if not os.environ.get("OPENROUTER_API_KEY"):
            raise EnvironmentError(
                "OPENROUTER_API_KEY not set. Add to .env or use --dry-run."
            )
        from probes.behavioral.openai_client import OpenRouterClient

        clients = {
            m: OpenRouterClient(model=m, temperature=DEFAULT_TEMPERATURE, seed=EVAL_SEED)
            for m in models
        }

    n_done = 0
    n_skip = 0
    n_err = 0
    with out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()

        for rec in bank:
            for model in models:
                key = (rec["problem_id"], model)
                if key in done:
                    n_skip += 1
                    continue
                client = clients[model]
                temp = float(getattr(client, "temperature", DEFAULT_TEMPERATURE))
                max_tok = getattr(client, "max_tokens", "")
                seed = getattr(client, "seed", EVAL_SEED)
                notes = (
                    f"greedy_temp={temp}; seed={seed}; max_tokens={max_tok}; "
                    f"client=OpenRouterClient; verifier=verify_answer(blocksworld)"
                )
                try:
                    if dry_run:
                        # Use gold so dry-run analysis is structurally valid.
                        raw = rec["correct_answer"]
                    else:
                        payload = client.complete(rec["problem_id"], rec["problem_text"])
                        raw = str(payload.get("response") or payload.get("text") or "")
                        # OpenRouterClient may swallow transport errors into response text.
                        if raw.startswith("ERROR:"):
                            raise RuntimeError(raw)
                    ok = verify_answer(
                        rec["problem_id"],
                        raw,
                        rec["correct_answer"],
                        "blocksworld",
                        problem_text=rec["problem_text"],
                    )
                except Exception as exc:  # noqa: BLE001 — resume-safe transport errors
                    raw = f"ERROR: {exc}"
                    ok = False
                    n_err += 1
                    notes = notes + f"; error={type(exc).__name__}"
                    writer.writerow(
                        {
                            "pair_id": rec["pair_id"],
                            "arm": rec["arm"],
                            "problem_id": rec["problem_id"],
                            "model": model,
                            "model_short": MODEL_SHORT.get(model, model),
                            "num_blocks": rec["num_blocks"],
                            "plan_length": rec["plan_length"],
                            "behavioral_correct": "false",
                            "raw_response": raw,
                            "correct_answer": rec["correct_answer"],
                            "temperature": temp,
                            "seed": seed if seed is not None else "",
                            "max_tokens": max_tok,
                            "decoding_notes": notes,
                        }
                    )
                    f.flush()
                    n_done += 1
                    if "401" in raw or "Unauthorized" in raw or "User not found" in raw:
                        print(
                            f"AUTH FAILURE for {model}; aborting this shard. "
                            "Refresh OPENROUTER_API_KEY and re-run eval with resume.",
                            flush=True,
                        )
                        print(
                            f"Eval aborted: wrote {n_done}, skipped {n_skip}, "
                            f"err {n_err} → {out_path}",
                            flush=True,
                        )
                        return out_path
                    continue

                writer.writerow(
                    {
                        "pair_id": rec["pair_id"],
                        "arm": rec["arm"],
                        "problem_id": rec["problem_id"],
                        "model": model,
                        "model_short": MODEL_SHORT.get(model, model),
                        "num_blocks": rec["num_blocks"],
                        "plan_length": rec["plan_length"],
                        "behavioral_correct": str(bool(ok)).lower(),
                        "raw_response": raw,
                        "correct_answer": rec["correct_answer"],
                        "temperature": temp,
                        "seed": seed if seed is not None else "",
                        "max_tokens": max_tok,
                        "decoding_notes": notes,
                    }
                )
                f.flush()
                n_done += 1
                if n_done % 25 == 0:
                    print(f"  eval wrote {n_done} (skipped {n_skip}, err {n_err})", flush=True)

    print(f"Eval complete: wrote {n_done}, skipped {n_skip}, errors {n_err} → {out_path}")
    return out_path


def merge_shards(shard_dir: Path | None = None) -> Path:
    """Merge per-model shard CSVs into O14_naming_results.csv (last row wins)."""
    shard_dir = shard_dir or (DER / "O14_shards")
    parts = sorted(shard_dir.glob("*.csv"))
    if not parts:
        raise FileNotFoundError(f"No shards in {shard_dir}")
    frames = [pd.read_csv(p, dtype=str).fillna("") for p in parts]
    df = pd.concat(frames, ignore_index=True)
    # Drop transport failures when a later success exists; keep last otherwise.
    df["_err"] = df["raw_response"].astype(str).str.startswith("ERROR:")
    df = df.sort_values(["problem_id", "model", "_err"])  # False (ok) last
    df = df.drop_duplicates(["problem_id", "model"], keep="last")
    df = df.drop(columns=["_err"])
    DER.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_OUT, index=False)
    n_err = int(df["raw_response"].astype(str).str.startswith("ERROR:").sum())
    print(f"Merged {len(parts)} shards → {RESULTS_OUT} ({len(df)} rows, {n_err} errors)")
    return RESULTS_OUT


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _cluster_mean_ci(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> dict:
    """Mean of per-cluster values with cluster bootstrap percentile CI."""
    # One value per cluster expected for paired deltas; still group safely.
    df = pd.DataFrame({"v": values, "c": cluster_ids})
    per = df.groupby("c", sort=False)["v"].mean()
    estimate = float(per.mean()) if len(per) else float("nan")
    clusters = per.index.to_numpy()
    vals = per.to_numpy(dtype=float)
    if len(vals) == 0:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_clustered": float("nan"),
            "n_clusters": 0,
        }
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = rng.choice(len(vals), size=len(vals), replace=True)
        boots[i] = float(np.mean(vals[draw]))
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "p_clustered": bootstrap_p_two_sided(boots),
        "n_clusters": int(len(clusters)),
    }


def _spearman_cluster(
    x: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> dict:
    from probes.common.cluster_inference import cluster_bootstrap_assoc

    return cluster_bootstrap_assoc(
        x, y, cluster_ids.tolist(), kind="spearman", n_boot=n_boot, seed=seed
    )


def run_analysis() -> Path:
    DER.mkdir(parents=True, exist_ok=True)
    if not RESULTS_OUT.exists():
        raise FileNotFoundError(f"Missing results: {RESULTS_OUT}")

    df = pd.read_csv(RESULTS_OUT, dtype=str).fillna("")
    df = df[~df["raw_response"].astype(str).str.startswith("ERROR:")].copy()
    df["ok"] = df["behavioral_correct"].str.strip().str.lower().eq("true").astype(int)
    df["num_blocks"] = pd.to_numeric(df["num_blocks"], errors="coerce")
    df["plan_length"] = pd.to_numeric(df["plan_length"], errors="coerce")

    # Last row per (pair, arm, model).
    df = df.drop_duplicates(["pair_id", "arm", "model"], keep="last")

    rows: list[dict] = []
    contrasts = [
        ("B_minus_A", ARM_B, ARM_A),
        ("C_minus_A", ARM_C, ARM_A),
        ("B_minus_C", ARM_B, ARM_C),
    ]

    for model, gmodel in df.groupby("model"):
        short = MODEL_SHORT.get(str(model), str(model))
        wide = gmodel.pivot_table(
            index="pair_id", columns="arm", values="ok", aggfunc="last"
        )
        meta = (
            gmodel.drop_duplicates("pair_id")
            .set_index("pair_id")[["num_blocks", "plan_length"]]
        )
        for contrast, arm_hi, arm_lo in contrasts:
            if arm_hi not in wide.columns or arm_lo not in wide.columns:
                continue
            sub = wide[[arm_hi, arm_lo]].dropna().join(meta, how="inner")
            if sub.empty:
                continue
            delta = (sub[arm_hi] - sub[arm_lo]).to_numpy(dtype=float)
            clusters = sub.index.to_numpy()
            ci = _cluster_mean_ci(delta, clusters, seed=hash(f"{model}{contrast}") % (2**31))
            acc_hi = float(sub[arm_hi].mean())
            acc_lo = float(sub[arm_lo].mean())
            rows.append(
                {
                    "analysis": "paired_accuracy_delta",
                    "contrast": contrast,
                    "model": model,
                    "model_short": short,
                    "n_pairs": int(len(sub)),
                    "acc_arm_hi": round(acc_hi, 6),
                    "acc_arm_lo": round(acc_lo, 6),
                    "delta_mean": round(ci["estimate"], 6),
                    "ci_low": round(ci["ci_low"], 6),
                    "ci_high": round(ci["ci_high"], 6),
                    "p_clustered": round(ci["p_clustered"], 6),
                    "n_clusters": ci["n_clusters"],
                    "moderator": "",
                    "spearman_r": "",
                    "spearman_ci_low": "",
                    "spearman_ci_high": "",
                    "spearman_p": "",
                    "interpretation_gate": (
                        "naming_moves_accuracy"
                        if abs(ci["estimate"]) >= 0.05
                        and (
                            ci["ci_low"] > 0
                            or ci["ci_high"] < 0
                        )
                        else "naming_null_compatible"
                    ),
                }
            )

            for moderator in ("num_blocks", "plan_length"):
                x = sub[moderator].to_numpy(dtype=float)
                y = delta
                sp = _spearman_cluster(
                    x, y, clusters, seed=hash(f"{model}{contrast}{moderator}") % (2**31)
                )
                rows.append(
                    {
                        "analysis": "delta_scales_with",
                        "contrast": contrast,
                        "model": model,
                        "model_short": short,
                        "n_pairs": int(len(sub)),
                        "acc_arm_hi": "",
                        "acc_arm_lo": "",
                        "delta_mean": round(float(np.mean(delta)), 6),
                        "ci_low": "",
                        "ci_high": "",
                        "p_clustered": "",
                        "n_clusters": sp["n_clusters"],
                        "moderator": moderator,
                        "spearman_r": round(sp["estimate"], 6)
                        if sp["estimate"] == sp["estimate"]
                        else "",
                        "spearman_ci_low": round(sp["ci_low"], 6)
                        if sp["ci_low"] == sp["ci_low"]
                        else "",
                        "spearman_ci_high": round(sp["ci_high"], 6)
                        if sp["ci_high"] == sp["ci_high"]
                        else "",
                        "spearman_p": round(sp["p_clustered"], 6)
                        if sp["p_clustered"] == sp["p_clustered"]
                        else "",
                        "interpretation_gate": "",
                    }
                )

        # Per-arm accuracy for reference.
        for arm, garm in gmodel.groupby("arm"):
            # cluster on pair (one row per pair already after pivot uniqueness)
            vals = garm.drop_duplicates("pair_id")["ok"].to_numpy(dtype=float)
            cids = garm.drop_duplicates("pair_id")["pair_id"].to_numpy()
            ci = _cluster_mean_ci(vals, cids, seed=hash(f"acc{model}{arm}") % (2**31))
            rows.append(
                {
                    "analysis": "arm_accuracy",
                    "contrast": str(arm),
                    "model": model,
                    "model_short": short,
                    "n_pairs": int(len(vals)),
                    "acc_arm_hi": round(ci["estimate"], 6),
                    "acc_arm_lo": "",
                    "delta_mean": "",
                    "ci_low": round(ci["ci_low"], 6),
                    "ci_high": round(ci["ci_high"], 6),
                    "p_clustered": "",
                    "n_clusters": ci["n_clusters"],
                    "moderator": "",
                    "spearman_r": "",
                    "spearman_ci_low": "",
                    "spearman_ci_high": "",
                    "spearman_p": "",
                    "interpretation_gate": "",
                }
            )

    # Pre-registered summary row across models (sign consistency).
    paired = [r for r in rows if r["analysis"] == "paired_accuracy_delta" and r["contrast"] == "B_minus_A"]
    if paired:
        moves = sum(1 for r in paired if r["interpretation_gate"] == "naming_moves_accuracy")
        rows.append(
            {
                "analysis": "preregistered_summary",
                "contrast": "B_minus_A",
                "model": "ALL",
                "model_short": "ALL",
                "n_pairs": "",
                "acc_arm_hi": "",
                "acc_arm_lo": "",
                "delta_mean": "",
                "ci_low": "",
                "ci_high": "",
                "p_clustered": "",
                "n_clusters": len(paired),
                "moderator": "",
                "spearman_r": "",
                "spearman_ci_low": "",
                "spearman_ci_high": "",
                "spearman_p": "",
                "interpretation_gate": (
                    f"models_with_substantial_naming_effect={moves}/{len(paired)}; "
                    "if majority move → reframe L1 as partly lexical; "
                    "if not → L1 survives controlled naming confound"
                ),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS_OUT, index=False)
    print(f"Wrote analysis → {ANALYSIS_OUT} ({len(out)} rows)")
    return ANALYSIS_OUT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "stage",
        choices=["generate", "eval", "analyze", "merge", "all"],
        help="Pipeline stage",
    )
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--output", type=str, default=None, help="Eval CSV path override")
    args = ap.parse_args()

    if args.stage in {"generate", "all"}:
        generate_bank(n_pairs=args.n_pairs)
    if args.stage in {"eval", "all"}:
        run_eval(
            dry_run=args.dry_run,
            resume=not args.no_resume,
            models=args.models,
            limit_pairs=args.limit_pairs,
            output_path=Path(args.output) if args.output else None,
        )
    if args.stage == "merge":
        merge_shards()
    if args.stage in {"analyze", "all"}:
        run_analysis()


if __name__ == "__main__":
    main()
