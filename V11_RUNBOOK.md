# v1.1 runbook

The corrected rerun of the retrieval-vs-computation study. New code lives in `rvc11/`. The old
pipeline is untouched and stays as an archive. Read `V11_AUDIT.md` for what was wrong and why.

## Run it

```bash
make v11-setup            # deps, PlanBench clone, creates .env
# put your key in .env:   OPENROUTER_API_KEY=...

make v11-qa               # 18 gates, offline, no key. Must print "18/18 gates passed"
make v11-dry              # whole pipeline on scripted models, no key

# edit configs/v11/models.yaml: replace every REPLACE-... id
make v11-models           # checks ids against the live catalogue, writes models.lock.json

make v11-pilot            # 2 problems per subtype, every model. Read the report before spending more
make v11-p1 RUN=main      # surface variants, all usable items, 5 samples
make v11-e1 RUN=main      # instruction x thinking x variant
make v11-e2 RUN=main      # noise floor, pass^k
make v11-p2 RUN=main      # plan, then continue under a false intermediate value
```

Every run command is resumable: run it again and only failed calls are retried.
One model only: `make v11-p1 RUN=main MODELS=glm,qwen27b`.

## What you get

`results/v11/<RUN>/`

| File | What |
|---|---|
| `calls.jsonl` | Raw calls, append-only: prompt, response, reasoning text, finish reason, token counts, provider, request settings, timestamp |
| `manifest.json` | Bank hash, git commit, model settings, study settings for every invocation |
| `scored.csv` | One row per call, derived from `calls.jsonl` by `python -m rvc11 score --run <RUN>` |
| `analysis/REPORT.md` | All tables below, plus one CSV per table |

Tables: call accounting (errors, truncations, marker rate), P1 by subtype and by family, P1 on the
verified tier only, P1 with strict scoring, E1 instruction effects, E2 noise floor, P2 false-state.

## Experiments

| Name | Question | Design |
|---|---|---|
| `p1_surface` | Does accuracy survive a surface change? | canonical + W1..W6, thinking on and off, 5 samples, paired with canonical per problem |
| `e1_instruction` | Was the v1 drop caused by the instruction, not the rewrite? | 3 instruction levels x thinking on/off x {canonical, W3, W4}, 3 samples. This is the v2 "instruction drift" contrast and tests v2 H7 |
| `e2_noise` | How big must an effect be to be real? | 20 verified canonical problems per family x 20 samples. Gives pass^k at k = 1, 4, 8 and the minimum detectable paired difference |
| `p2_plan` + `p2-continue` | Does a wrong intermediate value get corrected or carried forward? | The model's own correct step-by-step solution is cut at the middle step and shown back in three arms: unchanged (control), one value made false, false plus a cue that an error may exist. Tests v2 H6 and feeds v2 RQ2 |

Settings are in `configs/v11/study.yaml`. Change numbers there, not in code.

## Rules the code enforces

- A real run does not start unless all QA gates pass and the model id is in `models.lock.json`.
- Failed API calls are never scored. Truncated wrong answers are excluded and counted.
- Scoring reads the formal spec in the bank, never the prompt text.
- One run directory, one bank. Mixing two bank versions in one run is refused.

## Reading the P1 table

- `diff_vs_canonical` with its interval is the claim. `p_holm` corrects across variants.
- `retention` is blank when paired canonical accuracy is under 0.30.
- `note = canonical_at_ceiling` means the items are too easy for that model: no drop can be read as robustness.
- Compare `p1_by_subtype` with `p1_verified_only`. A result that exists only outside the verified tier is not a result.

## Changing things

- New model: add a block to `configs/v11/models.yaml`, run `make v11-models`.
- Local vLLM server: see the commented `qwen27b_local` block. Use `--allow-unverified`.
- After any change to the legacy banks or to `rvc11/bank.py`: `make v11-bank && make v11-qa`, and use a new `RUN` name.
