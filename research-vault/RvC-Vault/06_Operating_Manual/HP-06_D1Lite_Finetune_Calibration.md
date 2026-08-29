# HP-06 — D1-lite: Ground-Truth Calibration via Seen/Unseen Fine-Tune
addresses: [[D01_Controlled_Exposure_Validation]] · phase: 1 flagship · needs: 1× A100-40GB (or 2×24GB), ~1 GPU-day, repo

PROMPT:
Goal: first-ever external validation of the convergence labels. Ground truth = which items the model was fine-tuned on.
Steps:
1. Split ALGO canonicals (110) 50/50 stratified by subtype (CC/SP/WIS) and canonical difficulty; SEEN set gets LoRA fine-tuning on canonical problem + full CoT solution (use repo gold solutions; 3 epochs, lr 1e-4, r=16 — log all hyperparams). Base: meta-llama/Llama-3.1-8B-Instruct.
2. Falsification arm (adopt from Xie et al. 2410.23123): a second LoRA fine-tuned on the same SEEN items with corrupted final answers. Labels on this arm must NOT read computation-consistent for seen items; if they do, the instrument is broken.
3. Rerun the full three-probe pipeline (P1 W1–W6, P2A/2B, P3 proximity unchanged) on SEEN and UNSEEN for: base model, clean-FT, corrupted-FT.
4. Analysis: ROC of retrieval-consistent label vs SEEN membership across the full threshold grid; report AUC, and sensitivity/specificity at the pre-registered thresholds (HP-04). Then freeze calibrated thresholds and update docs/LABEL_THRESHOLDS_PREREG.md v2.
5. Ride-along: compute Commitment Depth (HP-08 definition) on seen vs unseen — expect earlier commitment on seen.
Output: adapters, raw CSVs per arm, D1LITE_REPORT.md with ROC figure and the headline sentence "labels recover known exposure with X sensitivity / Y specificity (AUC Z)".
Validate: no test-set leakage into LoRA training beyond design; UNSEEN canonical accuracy should not degrade >5pp vs base (else FT damaged the model — rerun at lower lr); every pipeline setting identical across arms.
Gate: AUC ≥ 0.75 → design D1-mid frequency ladder; 0.6–0.75 → revise thresholds/signals and rerun; <0.6 → labels do not track exposure; program pivots to D2/D3 framing. Record gate outcome in [[01_Program_State]].
