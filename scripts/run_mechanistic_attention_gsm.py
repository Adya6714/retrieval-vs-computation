#!/usr/bin/env python3
"""GSM attention-to-numbers mechanistic probe (canonical vs W3).

Independent signal from final-layer gold rank: for each GSM can/W3 item,
measure mean attention (avg over heads) from the final prompt token onto
numeric-value tokens in the problem statement, at every layer.

Reuses the same Llama-3.1-8B-Instruct chat-direct prompt construction as
``scripts/run_mechanistic_llama_gsm_sp.py``.

Outputs (default ``results/raw/mechanistic/``):
  mechanistic_attention_raw.csv
  mechanistic_attention_summary.csv
  mechanistic_attention_behavior_link.csv

Usage:
  python3 scripts/run_mechanistic_attention_gsm.py
  python3 scripts/run_mechanistic_attention_gsm.py --limit 2  # smoke
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_mechanistic_llama_gsm_sp import (  # noqa: E402
    DEFAULT_MODEL,
    GSM_BANK,
    GSM_P1_LLAMA,
    _family_instruction,
)

# Integers, decimals, optional leading $ (hotel-phone style "$0.9").
_NUMBER_RE = re.compile(r"\$?-?\d+(?:\.\d+)?")

RAW_COLS = [
    "problem_id",
    "variant_type",
    "layer",
    "n_layers",
    "mean_attention",
    "attention_mass",
    "n_numeric_tokens",
    "numeric_token_positions",
    "numeric_token_decoded",
    "seq_len",
    "model",
    "behavioral_correct",
    "caveat_validity_gate",
]


def _load_gsm_can_w3(limit: int | None) -> list[dict]:
    bank = pd.read_csv(REPO_ROOT / GSM_BANK, dtype=str).fillna("")
    rows: list[dict] = []
    for vt in ("canonical", "W3"):
        if vt == "canonical":
            sub = bank[bank["variant_type"].astype(str).str.strip() == "canonical"]
        else:
            sub = bank[bank["variant_type"].astype(str).str.strip().str.upper() == "W3"]
        for _, r in sub.iterrows():
            rows.append(
                {
                    "problem_id": str(r["problem_id"]).strip(),
                    "variant_type": "canonical" if vt == "canonical" else "W3",
                    "problem_text": str(r["problem_text"]),
                }
            )
    rows.sort(key=lambda d: (0 if d["variant_type"] == "canonical" else 1, d["problem_id"]))
    if limit is not None:
        # limit per variant
        capped: list[dict] = []
        counts = {"canonical": 0, "W3": 0}
        for d in rows:
            if counts[d["variant_type"]] >= limit:
                continue
            counts[d["variant_type"]] += 1
            capped.append(d)
        rows = capped
    return rows


def _load_p1_labels() -> dict[tuple[str, str], bool | None]:
    path = REPO_ROOT / GSM_P1_LLAMA
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    out: dict[tuple[str, str], bool | None] = {}
    for _, r in df.iterrows():
        pid = str(r["problem_id"]).strip()
        vt = str(r["variant_type"]).strip()
        if vt.lower() == "canonical":
            vt = "canonical"
        elif vt.upper().startswith("W"):
            vt = vt.upper()
        val = str(r.get("behavioral_correct", "")).strip().lower()
        if val in {"true", "1", "yes"}:
            out[(pid, vt)] = True
        elif val in {"false", "0", "no"}:
            out[(pid, vt)] = False
        else:
            out[(pid, vt)] = None
    return out


def _problem_char_span_in_prompt(prompt: str, problem_text: str) -> tuple[int, int] | None:
    """Locate problem_text inside the chat-templated prompt."""
    # Prefer the tagged "Problem:\n..." block used by the GSM harness.
    marker = "Problem:\n"
    idx = prompt.find(marker)
    if idx >= 0:
        start = idx + len(marker)
        # problem_text should follow; allow minor whitespace drift
        chunk = prompt[start:]
        if chunk.startswith(problem_text):
            return start, start + len(problem_text)
        # fallback: search raw problem_text
    pos = prompt.find(problem_text)
    if pos < 0:
        return None
    return pos, pos + len(problem_text)


def _numeric_token_indices(
    tokenizer,
    prompt: str,
    problem_text: str,
) -> tuple[list[int], list[str], int]:
    """Token indices overlapping numeric spans inside the problem statement."""
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    # Some Llama tokenizers need the slow path for offsets; fall back if missing.
    if "offset_mapping" not in encoded:
        # Manual alignment via cumulative decode (approximate but deterministic).
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # Rebuild offsets by decoding prefixes — expensive but n=88 is fine.
        offsets = []
        for i in range(len(input_ids)):
            # decode token alone for length estimate is unreliable; use prefix
            prefix = tokenizer.decode(input_ids[: i + 1], skip_special_tokens=False)
            # This can desync; prefer requiring offset_mapping.
            offsets.append((0, 0))
        raise RuntimeError("tokenizer missing offset_mapping; need a fast tokenizer")

    offsets = encoded["offset_mapping"][0].tolist()  # list of (start, end)
    input_ids = encoded["input_ids"][0]
    seq_len = int(input_ids.shape[0])

    span = _problem_char_span_in_prompt(prompt, problem_text)
    if span is None:
        return [], [], seq_len
    p_start, p_end = span
    problem_region = prompt[p_start:p_end]

    numeric_idx: list[int] = []
    decoded: list[str] = []
    for m in _NUMBER_RE.finditer(problem_region):
        # Skip bare minus signs / empty
        tok = m.group(0)
        if not re.search(r"\d", tok):
            continue
        abs_start = p_start + m.start()
        abs_end = p_start + m.end()
        for ti, (a, b) in enumerate(offsets):
            a, b = int(a), int(b)
            if b <= a:
                continue  # special tokens often (0,0)
            # overlap with number span
            if a < abs_end and b > abs_start:
                if ti not in numeric_idx:
                    numeric_idx.append(ti)
                    decoded.append(tokenizer.decode([int(input_ids[ti])]))
    return numeric_idx, decoded, seq_len


def _attention_to_numbers(
    attentions: tuple,
    numeric_idx: list[int],
) -> list[tuple[float, float]]:
    """Per layer: (mean_attention, attention_mass) from final token → numeric toks.

    attentions[layer]: (batch, n_heads, seq, seq)
    Query position = last prompt token (-1).
    mean_attention = mean over numeric positions of (mean over heads).
    attention_mass = sum over numeric positions of (mean over heads).
    """
    rows = []
    if not numeric_idx:
        return [(float("nan"), float("nan"))] * len(attentions)
    idx = np.asarray(numeric_idx, dtype=np.int64)
    for layer_attn in attentions:
        # layer_attn: torch.Tensor
        attn = layer_attn[0]  # (heads, seq, seq)
        # mean over heads → (seq_q, seq_k); take query=-1, keys=numeric
        mean_heads = attn.mean(dim=0)[-1, idx].float().cpu().numpy()
        mean_attention = float(mean_heads.mean())
        attention_mass = float(mean_heads.sum())
        rows.append((mean_attention, attention_mass))
    return rows


def _wilcoxon_per_layer(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    layers = sorted(raw["layer"].astype(int).unique())
    for layer in layers:
        sub = raw[raw["layer"].astype(int) == layer]
        can = sub[sub.variant_type == "canonical"].set_index("problem_id")
        w3 = sub[sub.variant_type == "W3"].set_index("problem_id")
        ids = sorted(set(can.index) & set(w3.index))
        a = pd.to_numeric(can.loc[ids, "mean_attention"], errors="coerce")
        b = pd.to_numeric(w3.loc[ids, "mean_attention"], errors="coerce")
        mask = a.notna() & b.notna()
        a, b = a[mask], b[mask]
        n = int(len(a))
        med_can = float(a.median()) if n else float("nan")
        med_w3 = float(b.median()) if n else float("nan")
        w_two = p_two = w_greater = p_greater = float("nan")
        if n >= 1 and not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True):
            try:
                w_two, p_two = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                # H1 exploratory: canonical attends more to numbers than W3 rename
                w_greater, p_greater = stats.wilcoxon(
                    a, b, zero_method="wilcox", alternative="greater"
                )
            except ValueError:
                pass
        rows.append(
            {
                "layer": layer,
                "n_paired": n,
                "median_mean_attention_canonical": med_can,
                "median_mean_attention_W3": med_w3,
                "delta_median_can_minus_W3": med_can - med_w3 if n else float("nan"),
                "wilcoxon_W_two_sided": w_two,
                "wilcoxon_p_two_sided": p_two,
                "wilcoxon_W_can_gt_W3": w_greater,
                "wilcoxon_p_can_gt_W3": p_greater,
                "row_type": "per_layer",
            }
        )

    # Final 3 layers summary (mean of mean_attention across those layers, then Wilcoxon)
    if layers:
        last3 = layers[-3:]
        def _final3_mean(df_vt: pd.DataFrame) -> pd.Series:
            sub = df_vt[df_vt["layer"].astype(int).isin(last3)]
            g = sub.groupby("problem_id")["mean_attention"].apply(
                lambda s: pd.to_numeric(s, errors="coerce").mean()
            )
            return g

        can_all = raw[raw.variant_type == "canonical"]
        w3_all = raw[raw.variant_type == "W3"]
        a = _final3_mean(can_all)
        b = _final3_mean(w3_all)
        ids = sorted(set(a.index) & set(b.index))
        a, b = a.loc[ids], b.loc[ids]
        mask = a.notna() & b.notna()
        a, b = a[mask], b[mask]
        n = int(len(a))
        med_can = float(a.median()) if n else float("nan")
        med_w3 = float(b.median()) if n else float("nan")
        w_two = p_two = w_greater = p_greater = float("nan")
        if n >= 1 and not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True):
            try:
                w_two, p_two = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                w_greater, p_greater = stats.wilcoxon(
                    a, b, zero_method="wilcox", alternative="greater"
                )
            except ValueError:
                pass
        rows.append(
            {
                "layer": f"final3_mean({last3[0]}-{last3[-1]})",
                "n_paired": n,
                "median_mean_attention_canonical": med_can,
                "median_mean_attention_W3": med_w3,
                "delta_median_can_minus_W3": med_can - med_w3 if n else float("nan"),
                "wilcoxon_W_two_sided": w_two,
                "wilcoxon_p_two_sided": p_two,
                "wilcoxon_W_can_gt_W3": w_greater,
                "wilcoxon_p_can_gt_W3": p_greater,
                "row_type": "final3_summary",
            }
        )
    return pd.DataFrame(rows)


def _behavior_link(raw: pd.DataFrame, n_layers: int) -> pd.DataFrame:
    """Spearman: attention-to-numbers vs behavioral correctness (P1 labels)."""
    # Use final-layer mean_attention; also final3 mean — report both as measured.
    rows = []
    last3 = list(range(max(0, n_layers - 3), n_layers))

    def _slice_frame(layer_sel, label: str) -> pd.DataFrame:
        if layer_sel == "final":
            sub = raw[raw["layer"].astype(int) == (n_layers - 1)].copy()
            sub["attn"] = pd.to_numeric(sub["mean_attention"], errors="coerce")
        else:
            sub = raw[raw["layer"].astype(int).isin(last3)].copy()
            sub["attn"] = pd.to_numeric(sub["mean_attention"], errors="coerce")
            sub = (
                sub.groupby(["problem_id", "variant_type", "behavioral_correct"], as_index=False)[
                    "attn"
                ]
                .mean()
            )
        return sub

    for slice_name, layer_sel in [
        ("final_layer", "final"),
        ("final3_mean", "final3"),
        ("canonical_final_layer", "final"),
        ("W3_final_layer", "final"),
        ("all_variants_final_layer", "final"),
    ]:
        sub = _slice_frame(layer_sel, slice_name)
        if slice_name == "canonical_final_layer":
            sub = sub[sub["variant_type"] == "canonical"]
        elif slice_name == "W3_final_layer":
            sub = sub[sub["variant_type"] == "W3"]
        # behavioral_correct may be empty string
        sub = sub.copy()
        sub["y"] = sub["behavioral_correct"].map(
            lambda v: 1.0 if str(v).lower() in {"true", "1", "yes"} else (
                0.0 if str(v).lower() in {"false", "0", "no"} else np.nan
            )
        )
        use = sub.dropna(subset=["attn", "y"])
        n = int(len(use))
        n_ok = int((use["y"] == 1).sum())
        n_bad = int((use["y"] == 0).sum())
        med_ok = float(use.loc[use["y"] == 1, "attn"].median()) if n_ok else float("nan")
        med_bad = float(use.loc[use["y"] == 0, "attn"].median()) if n_bad else float("nan")
        rho = p = float("nan")
        if n >= 3 and use["attn"].nunique() > 1 and use["y"].nunique() > 1:
            try:
                rho, p = stats.spearmanr(use["attn"], use["y"])
            except ValueError:
                pass
        rows.append(
            {
                "slice": slice_name,
                "n": n,
                "n_correct": n_ok,
                "n_incorrect": n_bad,
                "median_mean_attention_correct": med_ok,
                "median_mean_attention_incorrect": med_bad,
                "spearman_rho": float(rho) if rho == rho else float("nan"),
                "spearman_p": float(p) if p == p else float("nan"),
                "label_source": "GSM_P1_behavioral_llama.csv",
                "note": (
                    "Spearman of mean_attention vs behavioral_correct (1/0); "
                    "no threshold tuning; empty P1 labels dropped"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out-dir", default="results/raw/mechanistic")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--validity-gate-status",
        default="NOT_PASSED_OR_NOT_RUN",
        help="Recorded on every raw row (Prompt-2 gate was blocked / failed).",
    )
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "mechanistic_attention_raw.csv"
    sum_path = out_dir / "mechanistic_attention_summary.csv"
    beh_path = out_dir / "mechanistic_attention_behavior_link.csv"

    queue = _load_gsm_can_w3(args.limit)
    p1 = _load_p1_labels()
    print(f"[setup] model={args.model} n_items={len(queue)} out={out_dir}")
    print(
        f"[caveat] validity_gate={args.validity_gate_status} "
        "(Prompt 2 did not clear; results reported anyway per explicit Prompt 3 request)"
    )

    if args.dry_run:
        n_layers = 32
        rows = []
        for item in queue:
            for layer in range(n_layers):
                rows.append(
                    {
                        "problem_id": item["problem_id"],
                        "variant_type": item["variant_type"],
                        "layer": layer,
                        "n_layers": n_layers,
                        "mean_attention": 0.01,
                        "attention_mass": 0.05,
                        "n_numeric_tokens": 3,
                        "numeric_token_positions": "[]",
                        "numeric_token_decoded": "",
                        "seq_len": 0,
                        "model": args.model,
                        "behavioral_correct": p1.get(
                            (item["problem_id"], item["variant_type"]), ""
                        ),
                        "caveat_validity_gate": args.validity_gate_status,
                    }
                )
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.set_grad_enabled(False)
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        print(f"[model] loading {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Prefer fast tokenizer for offset_mapping
        if not getattr(tokenizer, "is_fast", False):
            print("[warn] tokenizer is not fast; offset_mapping may fail")

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype_map[args.dtype],
            device_map="auto",
            attn_implementation="eager",  # required for output_attentions
        )
        model.eval()
        n_layers = int(model.config.num_hidden_layers)
        device = next(model.parameters()).device
        print(f"[model] n_layers={n_layers} device={device}")

        rows = []
        t0 = time.time()
        for i, item in enumerate(queue):
            user_msg = f"{_family_instruction('gsm')}\n\nProblem:\n{item['problem_text']}"
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                add_generation_prompt=True,
                tokenize=False,
            )
            numeric_idx, decoded, seq_len = _numeric_token_indices(
                tokenizer, prompt, item["problem_text"]
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            # Sanity: seq length match
            if int(inputs["input_ids"].shape[1]) != seq_len:
                # re-tokenize consistently
                seq_len = int(inputs["input_ids"].shape[1])
                # recompute indices with same add_special_tokens=False path already used

            with torch.no_grad():
                out = model(
                    **inputs,
                    output_attentions=True,
                    use_cache=False,
                )
            per_layer = _attention_to_numbers(out.attentions, numeric_idx)
            del out, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            beh = p1.get((item["problem_id"], item["variant_type"]))
            beh_str = "" if beh is None else str(bool(beh))

            for layer_i, (mean_attn, mass) in enumerate(per_layer):
                rows.append(
                    {
                        "problem_id": item["problem_id"],
                        "variant_type": item["variant_type"],
                        "layer": layer_i,
                        "n_layers": n_layers,
                        "mean_attention": mean_attn,
                        "attention_mass": mass,
                        "n_numeric_tokens": len(numeric_idx),
                        "numeric_token_positions": str(numeric_idx),
                        "numeric_token_decoded": "|".join(decoded),
                        "seq_len": seq_len,
                        "model": args.model,
                        "behavioral_correct": beh_str,
                        "caveat_validity_gate": args.validity_gate_status,
                    }
                )
            print(
                f"[{i+1}/{len(queue)}] {item['variant_type']} {item['problem_id']} "
                f"n_num={len(numeric_idx)} final_mean_attn="
                f"{per_layer[-1][0] if per_layer else float('nan'):.6g} "
                f"elapsed={time.time()-t0:.0f}s"
            )

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RAW_COLS)
        w.writeheader()
        for r in rows:
            # normalize behavioral_correct
            if isinstance(r.get("behavioral_correct"), bool):
                r["behavioral_correct"] = str(r["behavioral_correct"])
            w.writerow({k: r.get(k, "") for k in RAW_COLS})

    raw_df = pd.DataFrame(rows)
    n_layers = int(raw_df["n_layers"].iloc[0])
    summary = _wilcoxon_per_layer(raw_df)
    summary.to_csv(sum_path, index=False)
    behavior = _behavior_link(raw_df, n_layers)
    behavior.to_csv(beh_path, index=False)

    print("\n=== ATTENTION SUMMARY (final3 / last layer) ===")
    print(summary[summary.row_type == "final3_summary"].to_string(index=False))
    last = summary[summary.layer == n_layers - 1]
    if len(last):
        print(last.to_string(index=False))
    print("\n=== BEHAVIOR LINK ===")
    print(behavior.to_string(index=False))
    print(f"\nwrote {raw_path}")
    print(f"wrote {sum_path}")
    print(f"wrote {beh_path}")


if __name__ == "__main__":
    main()
