# v1.1 corrected rerun

Current measurement pipeline for the retrieval-vs-computation programme. Code lives in
[`rvc11/`](../../rvc11/). The v1 `probes/` + `scripts/` stack is kept as an archive; do not
rerun it for new claims.

| Doc | Role |
|-----|------|
| [RUNBOOK.md](RUNBOOK.md) | How to run: setup → QA → pilot → experiments |
| [AUDIT.md](AUDIT.md) | What was wrong in v1 and what v1.1 changes |
| Root stubs | [`V11_RUNBOOK.md`](../../V11_RUNBOOK.md) · [`V11_AUDIT.md`](../../V11_AUDIT.md) |

**Status (13 Nov 2026):** pipeline shipped and dry-run tested. No paid model run yet.
Model IDs in `configs/v11/models.yaml` are still placeholders.

---

## Experiments designed

Settings: [`configs/v11/study.yaml`](../../configs/v11/study.yaml). Bank: [`data/v11/`](../../data/v11/)
(1,379 usable items; 1,179 verified / 200 unverified; 54 quarantined).

| ID | Make target | Question | Design |
|----|-------------|----------|--------|
| **P1** `p1_surface` | `make v11-p1 RUN=main` | Does accuracy survive a surface change? | All usable items × {canonical, W1–W6} × thinking on/off × 5 samples; paired vs canonical |
| **E1** `e1_instruction` | `make v11-e1 RUN=main` | Was the v1 drop caused by the instruction, not the rewrite? | {none, show_work, answer_only} × thinking × {canonical, W3, W4} × 3 samples (v2 H7 / instruction-drift contrast) |
| **E2** `e2_noise` | `make v11-e2 RUN=main` | How big must an effect be to be real? | 20 verified canonical problems per family × 20 samples; pass^k at k=1,4,8 and minimum detectable paired difference |
| **P2** `p2_plan` → `p2-continue` | `make v11-p2 RUN=main` | Is a wrong intermediate value corrected or carried forward? | GSM canonical only today: cut the model's own solution mid-way; arms = control / inject_false / inject_cued (v2 H6 / RQ2) |

Outputs land in `results/v11/<RUN>/` (`calls.jsonl`, `manifest.json`, `scored.csv`, `analysis/REPORT.md`).

---

## Tests to run (ordered)

Do not skip ahead. Each step gates the next.

| # | Command | Pass criterion | Why |
|---|---------|----------------|-----|
| 0 | `make v11-setup` | `.env` exists; PlanBench present | Deps and key file |
| 1 | `make v11-qa` | `18/18 gates passed` | Bank, prompts, verifiers, resume, P2 parse |
| 2 | `make v11-test` | pytest green | Unit coverage for `rvc11` |
| 3 | `make v11-dry` | `results/v11/_dry/analysis/REPORT.md` exists | Full pipeline on scripted mocks, no API key |
| 4 | Edit `configs/v11/models.yaml` → `make v11-models` | `configs/v11/models.lock.json` written; every real id confirmed | Live runs refuse unlocked ids |
| 5 | `make v11-pilot RUN=pilot` | Report readable; formats and costs look sane | 2 problems/subtype before spending |
| 6 | `make v11-p1 RUN=main` | Resumable; `scored.csv` + P1 tables | Primary surface battery |
| 7 | `make v11-e1 RUN=main` | E1 tables present | Instruction confound contrast |
| 8 | `make v11-e2 RUN=main` | Noise-floor / pass^k tables | Effect-size floor |
| 9 | `make v11-p2 RUN=main` | P2 false-state tables | Plan-continue injection (GSM) |

One model only: append `MODELS=glm,qwen27b`. After any bank or `rvc11/bank.py` change: `make v11-bank && make v11-qa`, and use a **new** `RUN` name.

---

## Next steps

1. **Unblock models.** Replace every `REPLACE-…` id in `configs/v11/models.yaml`, run `make v11-models`, commit the lock file only after a successful catalogue check (lock is gitignored by default; keep the confirmed yaml committed).
2. **Pilot.** `make v11-pilot` on the full roster; read cost, truncation rate, marker rate, and ceiling effects before main.
3. **Main behavioural battery.** P1 → E1 → E2 → P2 in that order on `RUN=main` (or dated run names).
4. **Bank hardening (parallel, non-blocking for pilot).** Regenerate the 200 unverified rewrites with a model that is **not** on the evaluation roster; re-derive GSM W6 gold instead of inheriting legacy answers.
5. **Port remaining P2.** Blocksworld / ALGO / Mystery P2 exist only as v1 scripts (`BW_P2_*`, `ALGO_P2_*`, `MBW_P2_*`); not in `rvc11` yet.
6. **Defer until behavioural numbers exist.** P3 (infini-gram), mechanistic Track T, paper number refresh, site refresh. Absolute v1.1 accuracies are **not** comparable to v1 (new `FINAL ANSWER:` marker + stricter verifiers).

---

## Blockers

| Blocker | Blocks | Resolution |
|---------|--------|------------|
| Placeholder model ids; no `models.lock.json` | Any paid run | Fill `configs/v11/models.yaml` → `make v11-models` |
| `OPENROUTER_API_KEY` (or local vLLM) | Steps 4–9 | Put key in `.env` (see `env.example`) |
| Unknown frontier ceiling / cost | How large to set main | Run pilot first |
| 200 unverified rewrites | Verified-tier-only claims on those rows | Regenerate with non-evaluated rewrite model |
| GSM W6 gold from legacy bank | Strong W6 claims | Re-derive gold at bank-build time |
| P2 only on GSM arithmetic | Cross-family false-state claims | Port BW/ALGO P2 into `rvc11` |
| GPU for mechanism / Track T | Phase-1 architecture claims | Separate from v1.1 behavioural rerun; still Colab-T4 / A100 gated |

---

## Layout (v1.1)

```
rvc11/                 # corrected pipeline (client, bank, verify, experiments, score, analyze)
configs/v11/           # models.yaml, study.yaml
data/v11/              # bank.jsonl + quarantine + summary
results/v11/<RUN>/     # calls, scores, REPORT.md
tests_v11/             # pipeline tests
docs/v11/              # this folder
```

v1 archive (do not delete; do not use for new numbers): `probes/`, `scripts/*_SCR_*`,
`scripts/consolidate/`, `rebuild/`, `audit_2026_08/`, `results/raw/`, `results/derived/`.
Number authority for the **published v1 manuscript** remains `rebuild/NUMBERS.csv` +
[`docs/paper/PAPER_ERRATA.md`](../paper/PAPER_ERRATA.md) until a v1.1 paper refresh.
