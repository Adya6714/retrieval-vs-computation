# Figure–caption–script audit

Compiled figure numbers assume main.tex figures first (teaser, robustness, CCI), then appendix order. Every file in `paper/figures/` is listed.

**Suspected errors (1–5) are confirmed or refuted in the matching figure rows and in the BW section at the end.**

---

## Fig 1 — `teaser.png` (`fig:framework`)

- **Script:** `paper/figures/scripts/legacy/fig5_teaser.py` (writes `results/figures/fig5_teaser.png`, not the paper path). MD5 of `paper/figures/teaser.png` ≠ MD5 of the legacy output. No generator currently writes `paper/figures/teaser.png`.
- **Plotted:** Three-probe framework schematic (hand-placed or copied PNG; 668 KB).
- **Caption:** Multi-probe diagnostic evaluation framework; labels only when probes converge.
- **Verdict: AMBIGUOUS** — caption matches the *intent* of the legacy teaser script, but the shipped PNG is not that script’s output.
- **Suggested caption:** keep, but treat the asset as hand-authored.

---

## Fig 2 — `fig_robustness.pdf` (`fig:dissociation`)

- **Script:** `paper/figures/scripts/gen_figures.py` → `fig_robustness()`
- **Plotted:** (a) GSM canonical accuracy vs W3 retention, 5 models (GPT-4o/Llama n=20). (b) Adversarial ALGO grouped bars, canonical vs W3, 5 models × SP/CC/WIS-adv.
- **Caption:** Accuracy–robustness dissociation; GSM n=44 / 20; SP-adv Fisher p=0.0021; CC reverses; WIS floors.
- **Verdict: CORRECT** (panel content matches; Fisher p is claimed in caption, not drawn as a test annotation).
- **Suggested caption:** none.

---

## Fig 3 — `fig_cci.pdf` (`fig:probe2`)

- **Script:** `paper/figures/scripts/gen_figures.py` → `fig_cci()`
- **Plotted:** (a) GSM mean CCI bars, median ticks, Wilcoxon arrow. Title: **“Plan fidelity (CCI)”**. (b) GSM mean TEP bars + Phase-2A accuracy overlay. Title: **“Trajectory error propagation”**. (c) ALGO Phase-2B stacked `response_type` (compliant / partial-refusal / format-ignored), n=61.
- **Caption:** Probe 2 results; plan fidelity (CCI); trajectory error propagation (TEP); injection-step response.
- **Verdict: CORRECT** — this script uses the Appendix D names, **not** “contradiction-citation” / “target-evidence persistence”.
- **Suggested caption:** none.

---

## Fig 4 — `fig_landscape.pdf` (`fig:landscape`)

- **Script:** `paper/figures/scripts/gen_new_figures.py` → `fig_landscape()`
- **Plotted:** Scatter of canonical accuracy vs W3 retention, 5 models × 3 families, per-family Spearman.
- **Caption:** Landscape; ALGO ρ=+0.90 p=0.04; GSM ρ=+0.15 p=0.80; BW ρ=−0.30 p=0.62.
- **Verdict: CORRECT** (numbers come from the same Spearman the script annotates; they should be regenerated if `master_per_problem_5model.csv` changes).
- **Suggested caption:** none.

---

## Fig 5 — `fig_decay.pdf` (`fig:pervariant`)

- **Script:** `paper/figures/scripts/gen_figures.py` → `fig_decay()`
- **Plotted:** (a) GSM per-variant lines Can–W6, 5 models. (b) Challenging ALGO SP/CC, Claude vs GPT-4o only.
- **Caption:** GSM W3 drop; W6 closer to canonical; ALGO SP vs CC reversal at W3.
- **Verdict: CORRECT**
- **Suggested caption:** none.

---

## Fig 6 — `fig_within_model.pdf` (`fig:within`)  *(suspected error 4)*

- **Script:** `paper/figures/scripts/gen_new_figures.py` → `fig_within_model()`
- **Plotted:** 3×5 small multiples, canonical correct vs W3 kept, Spearman ρ. If `canonical_correct` is constant, title is `ρ=n.a. (constant; n=…)`. **ALGO / o4-mini is that cell** (canonical = 1.0 on all 110 problems, so Spearman is undefined). The cell is still drawn (scatter + n=110), not a blank “n.a.” placeholder (`n.a.` placeholder is only used when n<5).
- **Caption:** “3 families × 5 models. Each cell reports the Spearman ρ …”
- **Verdict: WRONG** (overclaim). The grid is 3×5, but **not every cell reports ρ**. Caption also says Llama and o4-mini on **BW** are the only flat/negative cells; the figure itself flags ALGO/o4-mini as n.a.-constant and ALGO/Llama ρ=−0.04.
- **Suggested caption:** “Within-model per-problem canonical correctness vs W3 correctness (3 families × 5 models). Spearman ρ on the two binary indicators, omitted when either indicator is constant (ALGO/o4-mini: Acc_can=1.00, n=110). Positive ρ in most remaining cells: harder canonical problems also lose more under W3.”

---

## Fig 7 — `fig_corr_matrix.pdf` (`fig:macro`)

- **Script:** `paper/figures/scripts/gen_corr_figure.py` → `fig_cross_probe_corr()`
- **Plotted (PDF text confirms):** five-panel **GSM Spearman correlation heatmaps** (can, W1–W6, CCI, TEP, P2.acc, contam, n-gram) per model. Title: “Cross-probe Spearman correlation matrix per model — GSM”.
- **Caption:** “(a) GSM macro metrics: Gemini Acc_can 0.91 and CCI 0.27 … (b) Proximity predicts VRI …”
- **Verdict: WRONG** — caption describes `gen_figures.py:fig_heatmap()`, which writes **`fig_heatmap.pdf` (not shipped)**. The file in the paper is the correlation matrix.
- **Suggested caption:** “Cross-probe Spearman correlation of GSM Probe-1 variants, Probe-2 CCI/TEP/session-B correctness, and Probe-3 contamination, one panel per model (n=44; GPT-4o/Llama n=20).”
- **Alternatively:** ship `fig_heatmap.pdf` under this `\includegraphics` and keep the current caption.

---

## Fig 8 — `fig_probe2_summary.pdf` (`fig:probe2summary`)  *(suspected error 1)*

- **Script:** `paper/figures/scripts/gen_more_figures.py` → `fig_probe2_summary()`
- **Plotted:** Left: GSM mean CCI and mean TEP bars (`probe2_gsm_metrics.csv`). **Panel title in the script:** `"GSM Probe 2 — contradiction-citation (CCI) and target-evidence persistence (TEP)"`. Right: ALGO P2A normal vs elicited final-correct.
- **Caption:** same wrong expansions: “contradiction-citation index (CCI) and target-evidence persistence (TEP)”.
- **Metric definitions (main §3.2 / Appendix D):** **CCI = Cross-session Consistency Index**; **TEP = Trajectory Error Propagation**. `fig_cci.py` uses those names. This figure does not.
- **Verdict: WRONG** (script and caption agree with each other, and both contradict the paper’s metric section).
- **Suggested caption:** “Probe-2 summary. Left: GSM Cross-session Consistency Index (CCI) and Trajectory Error Propagation (TEP), n=44 (o4-mini 43/44 parseable if that file is used). Right: ALGO Phase-2A final-answer accuracy, unelicited vs algorithm-elicited.”

---

## Fig 9 — `fig_gsm_w5w6.pdf` (`fig:gsmw5w6`)  *(suspected error 2)*

- **Script:** `paper/figures/scripts/gen_more_figures.py` → `fig_gsm_w5w6()`
- **Plotted:** paired bars of `variant_type=="W5"` vs `"W6"` accuracy on the **intersection of problem_ids that have both**, no bank filter. Legend: **“W5 (numeric perturbation)”**, **“W6 (distractor injection)”**. Title repeats that.
- **What those columns actually are (question bank + generator):**
  - **W5** = GSM question reversal (`W5_GSM_SYSTEM`: original answer becomes given; ask for a former given). Example GSM_001: canonical asks cost of an 86-minute call (answer 51); W5 gives cost $51 and asks **how many minutes** (answer 86).
  - **W6** = new instance of the same template (`generate_w6` arithmetic branch). Example GSM_041: 190 students / answer 150 vs 54 students / answer 13.
- **Appendix C:** W5 = direction reversal; W6 = procedural regeneration (new numbers). There is **no** “distractor injection” variant.
- **Caption:** first sentence copies the **wrong** script labels (“numeric perturbation (W5) vs distractor injection (W6)”); later sentences switch to Appendix C (“W6 new numbers”, “W5 direction reversal”). Internally contradictory. Sample-size claim n=40 for GPT-4o/Llama is also wrong: unfiltered intersection is **64**; bank-valid W6 universe is **24** for every model.
- **Verdict: WRONG** — the plotted columns are Appendix-C W5 and W6; the **axis labels are swapped-and-invented**.
- **Suggested caption:** “GSM direction reversal (W5) vs procedural regeneration / new numbers (W6) on problems that have both variants. W6 stays closer to canonical for high-accuracy models; W5 (unknown swapped) drops more. n=24 bank W6 IDs (GSM_041–064) if filtered; unfiltered GPT-4o/Llama intersection is larger because of off-bank rows.”

---

## Fig 10 — `fig_population.pdf` (`fig:population`)

- **Script:** `paper/figures/scripts/gen_figures.py` → `fig_population()`
- **Plotted:** scatter of (canonical acc, W3 retention) cells for 4 models (no o4-mini), ALGO subtype×instance-type + GSM + BW-std/mys; highlights SP-adv and CC-adv Claude↔GPT-4o.
- **Caption:** population-level dissociation; inversion pairs highlighted; pooled Spearman ~0.
- **Verdict: CORRECT** (4-model plot; caption does not claim five).
- **Suggested caption:** none (optional: “o4-mini omitted”).

---

## Fig 11 — `fig_subtype_grid.pdf` (`fig:algo_grid`)  *(suspected error 3)*

- **Script:** `paper/figures/scripts/gen_more_figures.py` → `fig_subtype_grid()`
- **Plotted:** heatmap, 3 ALGO subtypes × 7 variants × 5 models. Row titles: `subtype_labels = {"coin_change": "Coin change (greedy)", "shortest_path": "Shortest path", "wis": "WIS (DP)"}`.
- **Caption:** 3×7×5 grid; “coin change (CC) is the only subtype where renaming retention is non-trivial. **(Note: CC is a dynamic-programming problem;** greedy is not optimal …)”
- **Verdict: WRONG (panel title) / caption is the correction.** Script labels CC as greedy; caption correctly says DP. WIS is the DP-labelled row in the script.
- **Suggested caption:** keep the DP note. **Fix the script label** to `"Coin change (DP)"` (and maybe `"WIS (weighted interval / DP)"`).

---

## Fig 12 — `fig_implaus.pdf` (`fig:plausible`)

- **Script:** `paper/figures/scripts/gen_new_figures.py` → `fig_implaus()`
- **Plotted:** ALGO P2B plausible vs implausible final-correct, 5 models, Wilcoxon p from `implausibility_detection.csv`.
- **Caption:** n=61 paired; no model differentiates.
- **Verdict: CORRECT** (assuming the derived CSV’s n matches; caption does not show per-bar n).
- **Suggested caption:** none.

---

## Fig 13 — `fig_violations.pdf` (`fig:bw_profile`)

- **Script:** `paper/figures/scripts/gen_new_figures.py` → `fig_violations()`
- **Plotted:** four horizontal-bar metrics for **Claude, GPT-4o, Llama only** (`bw_violation_profile.csv`): semantic validity, repetition, partial goals, median first-illegal step. Title n=50 sessions each.
- **Caption:** BW Probe-2 failure profile, n=50, three breakdown modes for those three models.
- **Verdict: CORRECT** for the three-model plot. Gemini/o4-mini are not in the figure (caption does not claim them).
- **Suggested caption:** none.

---

## Fig 14 — `fig_bw_inversion.pdf` (`fig:bw_rename`)  *(suspected error 5 — confirmed)*

- **Script:** `paper/figures/scripts/gen_more_figures.py` → `fig_bw_inversion()`
- **Plotted:** per-model bars of **canonical vs `variant_type=="W5"`** from BW P1 files. Legend: **“canonical blocks (a,b,c,…)”** vs **“renamed blocks (W₅)”**. Title: “renaming blocks flips the sign”.
- **Caption / §4.4:** “canonical vs renamed-blocks (W5)”; “Paired Wilcoxon on W5-rename”.
- **Appendix C:** W5 = **goal and initial states swapped**; W3 = **entity rename**.
- **Generator:** `generate_w5_bw` / `_swap_bw_nl_problem_text` / `swap_pddl_init_goal` = init↔goal. `generate_w3` for `planning_suite` applies entity **and action** mapping.
- **Verdict: WRONG.** The script plots **true W5 (direction reversal)**. It **labels** that column as block-rename. The rename operation is **W3**. Claude +23.9pp / Gemini +18.5pp / Llama −22.0pp is a **reversal** effect, not a rename effect.
- **Suggested caption:** “Blocksworld canonical vs direction-reversal (W5: initial and goal states swapped; block names unchanged). Unstacking the goal tower improves Claude/Gemini, collapses Llama, and leaves GPT-4o/o4-mini unchanged. Entity/action rename is W3, not this panel.”

Full texts for one problem: **[BW variant semantics](#bw-variant-semantics-definitive)** below.

---

## Fig 15 — `fig_mechanistic.pdf` (`fig:mechanistic`)

- **Script:** `paper/figures/scripts/gen_figures.py` → `fig_mechanistic()`
- **Plotted:** Qwen-2.5-7B 2×2: median final rank canonical vs **W6**; rank / logprob / cosine trajectories by layer, three families.
- **Caption:** mechanistic dissociation, canonical vs W6 (new numbers), four metrics; ALGO/BW dissociate, GSM does not.
- **Verdict: CORRECT**
- **Suggested caption:** none.

---

## Fig 16 — `pipeline.png` (`fig:pipeline`)

- **Script:** none under `paper/figures/scripts/`.
- **Plotted:** end-to-end pipeline illustration (hand-authored PNG).
- **Caption:** canonical → W1–W6; Probe 1/2/3; five models.
- **Verdict: AMBIGUOUS** — no generating script to check; caption is a fair description of the intended diagram.
- **Suggested caption:** keep.

---

## Fig 17 — `fig_paradox.pdf` (`fig:invocation`)

- **Script:** `paper/figures/scripts/gen_figures.py` → `fig_paradox()`
- **Plotted:** ALGO Phase-2A final-answer % by `reasoning_type` (algorithm_invocation, backtracking, forward_simulation, local_greedy, unclear); Fisher vs unclear baseline; 4 models.
- **Caption:** step-level reasoning type vs correctness; invocation n=13, 0% correct; Fisher p=0.40.
- **Verdict: CORRECT**
- **Suggested caption:** none.

---

## Unused generator (caption attached to the wrong file)

`paper/figures/scripts/gen_figures.py` → `fig_heatmap()` writes `fig_heatmap.pdf` (GSM Acc_can / R_W3 / CCI / TEP heatmap + proximity→VRI/CCI). That file is **not** in `paper/figures/`. Its caption is currently on **Fig 7** (`fig_corr_matrix.pdf`).

---

## BW variant semantics (definitive)

**Generator:** `scripts/generation/stage2_generate_variants.py`

| Variant | Function | What it does |
|---------|----------|----------------|
| **W3** | `generate_w3` + BW/MBW prompts in `variant_prompts.py` | Bijective **entity and action rename**. Tower structure and plan length preserved; `pick-up/stack/…` become a new verb set; block letters become names. |
| **W5** | `generate_w5_bw` → `swap_pddl_init_goal` + `_swap_bw_nl_problem_text` | **Swap Current state ↔ Goal** (start at the original goal tower; goal is the original init / all-clear-on-table). Block **letters unchanged**. Answer is the inverse unstack sequence. |
| **W6** | `generate_w6` planning branch | New random instance, same block count. |

Appendix C matches the generator. Section 4.4 and Fig 14 do **not**.

### One problem: `BW_001` (from `data/problems/question_bank_bw.csv`)

Wrapping quotes in the CSV are stripped below. Answers are `correct_answer`.

#### Canonical

You are a robot arm. Available actions: pick-up X (X must be clear and on the table, hand must be empty), put-down X (place X on the table), stack X Y (place X on Y; Y must be clear, you must be holding X), unstack X Y (pick up X from Y; X must be clear, hand must be empty). You can hold one block at a time. Current state: Blocks j, f, i, g, d, b, h, and l are clear and on the table. The hand is empty. Goal: Block j is on block f, block f is on block i, block i is on block g, block g is on block d, block d is on block b, block b is on block h, and block h is on block l. Respond with a numbered list of actions only. Each action must be exactly one of: pick-up X / put-down X / stack X Y / unstack X Y. No explanation. No extra text.

**Answer (build the tower):**
```
pick-up h
stack h l
pick-up b
stack b h
pick-up d
stack d b
pick-up g
stack g d
pick-up i
stack i g
pick-up f
stack f i
pick-up j
stack j f
```

#### W3 (entity + action rename — this is the block rename)

You are an HR manager building a reporting chain. Available actions: select X (X must have no direct reports and be unassigned, hands must be free), release X (return X to the unassigned pool), place X under Y (Y must have no one above them, you must be holding X), remove X from Y (X must have no direct reports, hands must be free). You can manage one employee at a time. Current state: James, Fiona, Ian, Grace, Derek, Ben, Hannah, and Leo are all unassigned and available. Your hands are free. Goal: James reports to Fiona, Fiona reports to Ian, Ian reports to Grace, Grace reports to Derek, Derek reports to Ben, Ben reports to Hannah, and Hannah reports to Leo. Respond with a numbered list of actions only. Each action must be exactly one of: select X / release X / place X under Y / remove X from Y. No explanation. No extra text.

**Answer (same tower, renamed):**
```
select Hannah
place Hannah under Leo
select Ben
place Ben under Hannah
select Derek
place Derek under Ben
select Grace
place Grace under Derek
select Ian
place Ian under Grace
select Fiona
place Fiona under Ian
select James
place James under Fiona
```

#### W5 (init/goal swap — not a rename)

You are a robot arm. Available actions: pick-up X (X must be clear and on the table, hand must be empty), put-down X (place X on the table), stack X Y (place X on Y; Y must be clear, you must be holding X), unstack X Y (pick up X from Y; X must be clear, hand must be empty). You can hold one block at a time. Current state: Block j is on block f, block f is on block i, block i is on block g, block g is on block d, block d is on block b, block b is on block h, and block h is on block l. Block l is on the table. The hand is empty. Goal: Blocks j, f, i, g, d, b, h, and l are all clear and on the table. Respond with a numbered list of actions only. Each action must be exactly one of: pick-up X / put-down X / stack X Y / unstack X Y. No explanation. No extra text.

**Answer (disassemble the tower; letters still j, f, i, g, d, b, h, l):**
```
unstack j f
put-down j
unstack f i
put-down f
unstack i g
put-down i
unstack g d
put-down g
unstack d b
put-down d
unstack b h
put-down b
unstack h l
put-down h
```

**One-line statement:** For BW, **W3 renames blocks and actions**; **W5 swaps initial and goal states and leaves names alone.** Fig 14 and §4.4 call W5 a block rename; that is false.

---

## Suspected-error scorecard

| # | Claim | Result |
|---|--------|--------|
| 1 | Fig 8 caption expands CCI as contradiction-citation and TEP as target-evidence persistence | **Confirmed.** In `fig_probe2_summary()` title **and** the caption. Official names are Cross-session Consistency Index / Trajectory Error Propagation (`fig_cci` uses those). |
| 2 | Fig 9 labels W5 numeric perturbation, W6 distractor injection | **Confirmed as a labelling error.** Plotted columns are real W5/W6. Generator+bank: W5=direction reversal, W6=new numbers. Appendix C matches the data, not the legend. |
| 3 | Fig 11 panel “Coin change (greedy)” vs caption “CC is DP” | **Confirmed.** Script: `"Coin change (greedy)"`. Caption is the correct DP note. |
| 4 | Fig 6 “3×5” but ALGO/o4-mini n.a. | **Confirmed as caption overclaim.** Grid is 3×5; ALGO/o4-mini Spearman is `n.a. (constant; n=110)` because Acc_can=1. |
| 5 | BW W5 = rename in §4.4/Fig 14 vs swap in Appendix C | **Confirmed: Appendix C and the generator are right; §4.4 and Fig 14 are wrong.** W3 is the rename. |
