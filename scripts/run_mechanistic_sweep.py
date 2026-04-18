"""
Layer 2 — requires GPU for real runs. Use --dry-run with
GPT-2 on CPU to verify the pipeline locally before committing GPU hours.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import yaml

from probes.common.io import QUESTION_BANK_PATH, QUESTION_BANK_COLUMNS



def main():
    parser = argparse.ArgumentParser(description="Layer 2 Mechanistic Sweep")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--probe", type=str, choices=["probe1", "probe2"], default="probe1")
    parser.add_argument(
        "--question-bank",
        type=str,
        default=QUESTION_BANK_PATH,
        help="Path to unified question bank CSV (used for probe1)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=True, help="Skip problem_ids already in output CSV")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--dry-run", action="store_true", help="Loads GPT-2 instead of Qwen, runs on first 2 problems only")
    parser.add_argument("--device", type=str, default="cuda", help="Override to CPU for dry-run")
    
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN: overriding model to gpt2 and device to cpu")
        args.model = "gpt2"
        args.device = "cpu"

    with open("configs/paths.yaml", "r") as f:
        paths = yaml.safe_load(f)

    problems_dir = Path(paths.get("problems_dir", "./data/problems"))
    results_dir = Path(paths.get("results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.probe == "probe1":
        instances_path = Path(args.question_bank)
    else:
        instances_path = problems_dir / f"{args.probe}_instances.csv"
    variants_path = problems_dir / f"{args.probe}_variants.csv"
    output_path = results_dir / f"{args.probe}_mechanistic.csv"

    if not instances_path.exists():
        print(f"Error: Instances file not found at {instances_path}")
        return

    df_instances = pd.read_csv(instances_path)
    if args.probe == "probe1":
        missing_cols = set(QUESTION_BANK_COLUMNS) - set(df_instances.columns)
        if missing_cols:
            raise ValueError(
                f"Question bank missing required columns: {sorted(missing_cols)}"
            )
        df_instances = df_instances[
            df_instances["variant_type"].astype(str).str.strip().str.lower() == "canonical"
        ]

    if args.family and "problem_family" in df_instances.columns:
        df_instances = df_instances[df_instances["problem_family"] == args.family]

    if args.limit is not None:
        df_instances = df_instances.head(args.limit)

    if args.dry_run:
        df_instances = df_instances.head(2)

    df_variants = pd.DataFrame()
    if variants_path.exists():
        df_variants = pd.read_csv(variants_path)

    if args.resume and output_path.exists() and output_path.stat().st_size > 0:
        try:
            processed = pd.read_csv(output_path)
            if "problem_id" in processed.columns:
                processed_ids = set(processed["problem_id"].astype(str).str.strip())
                df_instances = df_instances[~df_instances["problem_id"].astype(str).str.strip().isin(processed_ids)]
        except pd.errors.EmptyDataError:
            pass

    if df_instances.empty:
        print("No problems to process. Exiting.")
        return

    # Import mechanistic modules here so --dry-run flag is parsed first
    from probes.mechanistic import load_model, activations, logit_lens, similarity

    # Override MODEL_NAME using parameter mapping natively
    model = load_model.load_model(model_name=args.model, device=args.device)

    output_cols = [
        "problem_id",
        "problem_family",
        "model",
        "layer_cosine_similarities",
        "crystallization_layer",
        "n_layers_processed"
    ]

    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_cols)
        if write_header:
            writer.writeheader()
            outfile.flush()

        for _, row in df_instances.iterrows():
            problem_id = str(row["problem_id"]).strip()
            # Probe 2 may not have problem_family, default to blocksworld
            family = str(row.get("problem_family", "blocksworld")).strip()
            problem_text = str(row.get("problem_text", ""))
            correct_answer = str(row.get("correct_answer", ""))

            print(f"Processing problem_id: {problem_id}")

            try:
                # b. Run activations.extract_activations
                acts_canon = activations.extract_activations(model, problem_text, problem_id, "canonical")

                # c. Run logit_lens.run_logit_lens
                ll_res = logit_lens.run_logit_lens(model, problem_text)
                
                # Crystallization layer: first layer where the first token of the 
                # correct answer becomes the top predicted token.
                # For multi-token answers we use only the first token as a proxy.
                first_answer_token = correct_answer.strip().split()[0].lower() if correct_answer.strip() else ""
                cryst_layer = -1
                for r in ll_res:
                    if str(r.get("top_token", "")).strip().lower() == first_answer_token:
                        cryst_layer = r["layer"]
                        break

                n_layers = model.cfg.n_layers

                # d. Run similarity.layer_cosine_similarity with W1 variant
                sims_json = "[]"
                if not df_variants.empty:
                    w1_var = df_variants[
                        (df_variants["problem_id"].astype(str).str.strip() == problem_id) &
                        (df_variants["variant_type"].str.strip() == "W1")
                    ]
                    if w1_var.empty:
                        print(f"WARNING: No W1 variant found for {problem_id}.")
                    else:
                        variant_text = str(w1_var.iloc[0]["variant_text"])
                        acts_var = activations.extract_activations(model, variant_text, problem_id, "W1")
                        try:
                            sims = similarity.layer_cosine_similarity(acts_canon, acts_var)
                            sims_json = json.dumps(sims.tolist())
                        except Exception as e:
                            print(f"WARNING: similarity calc failed for {problem_id}: {e}")
                else:
                    print(f"WARNING: Variants file {variants_path} does not exist.")

                # e. Write result row
                writer.writerow({
                    "problem_id": problem_id,
                    "problem_family": family,
                    "model": args.model,
                    "layer_cosine_similarities": sims_json,
                    "crystallization_layer": cryst_layer,
                    "n_layers_processed": n_layers
                })
                outfile.flush()

            except Exception as e:
                print(f"ERROR: Failed on problem_id {problem_id}: {e}")
                continue
