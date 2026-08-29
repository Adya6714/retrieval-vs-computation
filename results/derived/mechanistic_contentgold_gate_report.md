# Mechanistic content-gold pass/fail gate

## Local forced-greedy ALGO canonical

- file: `results/raw/ALGO_llama31_8b_greedy_canonical.csv`
- accuracy: **6/110 = 0.0545** (5.5%)
- by subtype:
  - coin_change: 2/25 = 0.0800
  - shortest_path: 1/55 = 0.0182
  - wis: 3/30 = 0.1000

## Content-gold mechanistic ranks (ALGO canonical)

- file: `results/raw/mechanistic_sweep_llama31_8b_instruct_chatdirect_contentgold.csv`
- n ALGO canonical with ranks: **110**
- median final-layer rank: **14.0**
- mean final-layer rank: **140.2**
- still format-keyword targets (Path|Count|Selected): **0/110**
- top decoded targets: {'0': 63, '4': 14, '2': 9, '3': 5, '1': 4, '5': 3, '11': 2, '8': 2}

## Gate decision

**AMBIGUOUS**: greedy≈5.5% and median rank=14.0 in middle band (5.0 < med < 100.0); inspect distribution

## Paper cell provenance (Llama ALGO SP canonical)

- Table 7 (pkg8) **SP-chall. / Llama / Can. = .059** = **2/34** from `results/derived/ALGO_P1_4model_frozen_labels.csv` (shortest_path × adversarial × canonical).
- Table 7 **SP-std. / Llama / Can. = .048** = **1/21** from the same frozen file (shortest_path × standard × canonical).
- Raw OpenRouter run `results/raw/ALGO_P1_behavioral_llama.csv` overall canonical is **7/111 ≈ 6.3%** (verified). `probes/behavioral/openai_client.py` does **not** send `temperature` / `do_sample=False` — provider default decoding, **not** forced-greedy. That path is now dead (wallet/key); do not treat 7/111 as a greedy floor.
- Local forced-greedy for this gate: `scripts/algo_llama_greedy_accuracy.py` (`do_sample=False`) → `--greedy` CSV above.

