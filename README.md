# retrieval-vs-computation

**Research paper:** "Beyond Accuracy: Behavioral and Mechanistic Signatures of Retrieval-Like Processing in LLMs"

**Authors:** Adya, Shaswat, Nandini Banka — BITS Pilani (2026)

**Submission target:** BlackboxNLP 2026 (primary), GenBench 2026 (secondary)

---

## Research Question

When an LLM gets a multi-step reasoning problem right, is that correctness causally dependent on surface features of the input (words, phrasings, entity names seen in training) or on structural features (abstract relationships invariant under surface change)?

## Contribution

Behavioral signatures of retrieval-like processing co-occur with mechanistic signatures of retrieval-like processing at the per-instance level, and both correlate with training-data contamination — three independent lines of evidence converging on the same per-instance diagnosis.

## Three Probes

| Probe | Behavioral Component | Mechanistic Counterpart |
|-------|---------------------|------------------------|
| 1 — Surface Invariance | CSS across 6 surface variants | Layer-wise cosine similarity + activation patching (Qwen2.5-7B) |
| 2 — Plan-Execution Coupling | CCI / TEP on Blocksworld | Tuned/logit lens crystallization depth (Qwen2.5-7B) |
| 3 — Contamination Indexing | Infini-gram n-gram fingerprinting | Crystallization depth vs contamination score correlation |

## Repo Structure

```
probes/
  contamination/       # Probe 3: Infini-gram querying, fingerprinting
  probe1_surface/      # Probe 1: variant generation, CSS computation
  probe2_plan_exec/    # Probe 2: Blocksworld CCI/TEP, lens analysis
data/                  # gitignored except small CSVs; problem instances frozen here
notebooks/             # Figure generation, one notebook per probe
paper/                 # LaTeX source (ACL/EMNLP style)
scripts/               # Shared utilities (API wrappers, stat helpers)
results/               # CSV outputs — committed; one file per probe
```

## Setup

```bash
conda create -n rvc python=3.11
conda activate rvc
pip install -r requirements.txt
```

## Key Files

- `CHARTER.md` — research question, contribution, division of labor
- `data/probe1_instances.json` — frozen problem set for Probe 1
- `data/probe2_instances.json` — frozen Blocksworld instances for Probe 2
- `results/contamination_triage.csv` — Phase 1 go/no-go result
- `results/probe1_behavioral.csv` — Probe 1 CSS results
- `results/probe1_mechanistic.csv` — Probe 1 activation analysis
- `results/probe2_behavioral.csv` — Probe 2 CCI/TEP results
- `results/probe2_mechanistic.csv` — Probe 2 lens crystallization results
- `results/probe3_contamination.csv` — Full contamination sweep

## Status

- [ ] Phase 0: Infrastructure
- [ ] Phase 1: Contamination triage (go/no-go for Probe 3)
- [ ] Phase 2: Mechanistic tooling setup
- [ ] Phase 3: Probe 1
- [ ] Phase 4: Probes 2 & 3
- [ ] Phase 5: Writing
- [ ] Phase 6: Submission
