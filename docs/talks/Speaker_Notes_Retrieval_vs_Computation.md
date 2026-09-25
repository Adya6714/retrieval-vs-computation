# SPEAKER NOTES — Retrieval vs Computation (RvC)

**Site (presentation tool):** https://adya6714.github.io/retrieval-vs-computation/  
**Repo:** https://github.com/Adya6714/retrieval-vs-computation  
**Paper (CAISc / behavioural layer):** `paper/main.pdf`  
**Teaching companion:** `BOOK.md` · **Programme spine:** `research-vault/RvC-Vault/THE_PLAN.md`

~15 minutes full walk · ~5 minutes compressed (see Part D).  
These notes track the live site **section by section**. Bold lines in quote blocks are close-to-verbatim things to say. Everything else is why, how, and what to do with your hands.

---

## HOW TO USE THE SITE AS A PRESENTATION TOOL

### Navigation (do this, not free scrolling)

| Control | What it does |
|--------|----------------|
| **Left rail** | Jumps to Motivation → Programme → Strategy → RQs → Methodology → Experiments → F1–F6 → Future → Links |
| **↓ / j** and **↑ / k** | Move **section to section** (deliberate; looks intentional) |
| **Hover a blue number** | Tooltip shows `n` and the CSV filename — offer to open that file if they push |
| **Click cards / claims / phases / RQs / experiments / future methods** | Expand body text |
| **Tabs & chips** | Programme views, bank family/variant, probe tabs, F2 family, F3 model, F5 scope, future tiers |

### Before you share your screen

1. Open: **site**, **GitHub**, **paper PDF**, `results/derived/` (at least `probe1_per_model_variant.csv`, `C7_concordance_filtered.csv`, `P1_failure_patterns.csv`, `injection_recovery_summary.csv` path or paper AUDIT table).
2. Click through **Probe 2** once: Declare → Execute → Inject → Score with **Play stage**.
3. On **Finding 3**, pick Claude, click **Invert direction (W5)** once so you know the bar jumps.
4. Decide lead findings if short: **F3 (direction)** and **F6 (compliance)** — most surprising, easiest to demo.

### Name-clash warning (say the long form)

| Abbreviation on site | Say aloud |
|----------------------|-----------|
| **CC-1…CC-4** | “capability claim one…” |
| **CC** in optimisation | “coin change” |
| **AC-1…AC-3** | “architecture claim…” |
| **CCI** | “plan–execution consistency” (ε = 0.01) |
| **VRI** | “variant robustness index — mean of W1, W2, W4 minus W3” |
| **W3 / W5 / W6** | “rename / direction invert / new numbers” |

### How this site relates to the paper (say early)

Interviewers who saw “Same Score, Different Strategy” / CAISc on your CV need this once:

> “The CAISc paper is the **first paper of a four-paper programme**. It is the **behavioural layer**: it showed that accuracy hides differences in *how* models solve problems. **This site is the whole programme** — the measurement science that paper motivates, plus the gated later phases that ask where the split lives in the network and what training conditions produce it.”

That positions you as someone with an agenda, not a one-off paper.

---

## THE 30-SECOND OPENER (stay on Motivation)

> “When a language model gets a problem right, that correct answer is consistent with two very different processes: **actually executing a procedure**, or **matching a pattern it has seen before**. Accuracy cannot tell them apart. Most perturbation studies report that accuracy drops *on average across a group*. They do not tell you, for a **single instance**, which process produced the answer. My programme builds a **per-instance measurement** for that, then uses it for capability, architecture, and training-data claims.”

Why this works: it names the gap (group-level vs per-instance) in one breath.

---

# 1. MOTIVATION  
**Rail:** Motivation · **URL hash:** `#motivation`

### What is on screen

- Headline: *A measurement science for how language models solve problems.*
- Mission line under it (four-paper / capability + architecture + training laws).
- Three epistemic paragraphs (left border).
- Footer strip: *Four papers · five phases · 110+ vault notes* + GitHub / Paper / Vault / THE_PLAN links.
- Background lattice (atmospheric — not a cost chart).

### What to do

Stay here for the opener. Do **not** click links yet unless they ask for the repo.

### What to say — walk the three lines

1. > “A correct answer is consistent with executing a procedure or with matching a seen pattern. **Accuracy cannot separate them.**”
2. > “Group-level perturbation studies report that accuracy drops. They do not tell you, **for a given instance**, which process produced the answer.”
3. > “So **per-instance diagnosis is the object**. Everything downstream — where the split lives in the network, what training conditions produce it, when it emerges — depends on first having an **instrument** that works at instance granularity.”

### Why “instrument”

> “I use the word **instrument** deliberately. The first job is not to make a claim about models; it is to build a measurement that can be trusted. Claims come **after** the instrument is validated.”

### Point at the meta strip

> “The programme is four papers across five gated phases, kept coherent in a research vault of 110+ linked notes — you will see that force layout two sections down.”

### Likely question — “Why does process matter if the answer is right?”

> “Because the two processes **generalise differently**. A model executing a procedure should stay correct when you change surface details that do not affect the answer. A model matching patterns breaks when the surface moves away from what it saw. On a benchmark they look identical; in deployment they fail completely differently. If you choose a checkpoint, a data mixture, or a quantisation level from benchmark accuracy alone, you cannot see that difference.”

That last sentence is your bridge to systems / Sarvam / training teams.

**→ Press ↓ / j to Programme.**

---

# 2. PROGRAMME  
**Rail:** Programme · `#programme`

### What is on screen

- Status pills (counts of claims / hypotheses / phases by status).
- Tabs: **Claims** | **Hypothesis registry** | **Phase timeline**
- Claim filters: All · Capability · Architecture · Established · In progress · Designed · Planned
- Claim ↔ hypothesis SVG map under Claims
- Hyp filters: All · Untested · Correlational · Pilot · Open · Designed · Planned
- Phase track: click a diamond for do / needs / ships / advances

### What to do (scripted click path)

1. Land on **Claims** (default).
2. Say the three-view framing (below).
3. Click **CC-1** (established) → expand → optionally click linked **H1** / **H3**.
4. Filter **Architecture · AC** → open **AC-1** (in progress).
5. Switch tab → **Hypothesis registry** → open **H1** (gate G1 language).
6. Switch tab → **Phase timeline** → click **P0** (complete) then **P1** (next).

### Framing before any detail

> “Three views of the **same** science. **Claims** are what must be true. **Hypotheses** are what I bet and how I test each one. **Phases** are the gated sequence deciding which bets run next. On the map, capability claims sit above, architecture claims below; lines run to the hypotheses each claim needs.”

> “Each diamond is a **gate**. I only escalate when a pre-set decision criterion fires. That stops the programme from spending later compute — mechanistic probing, training runs — on a behavioural result that has not held up.”

---

### Capability claims (open while speaking)

| ID | Status | One-line | Evidence / refute (speakable) |
|----|--------|----------|-------------------------------|
| **CC-1** Two-dimensional capability space | **established** | Accuracy and surface-invariance are dissociable, with different causal antecedents. | Gemini GSM .909→.958 (W6) vs .523 (W3); pooled CCI×W3 r=.243 while neither tracks accuracy. **Refute:** W1–W6 order is permutation noise, or a universal hard-item set outside obfuscated planning. |
| **CC-2** Absence of source monitoring | **designed** | Models cannot report whether an answer came from memory. | DS-05 on D1-lite. **Refute:** source-report tracks SEEN/UNSEEN above chance after calibration. |
| **CC-3** Developmental order | **planned** | Invariance emerges after accuracy, with measurable lag. | D4 OLMo checkpoints. **Refute:** they move together at every checkpoint. |
| **CC-4** Compositional boundary | **designed** | A depth *k* beyond which retrieval cannot substitute for computation. | DS-09 compositional stress. **Refute:** no such depth on the preregistered schedule. |

### Architecture claims

| ID | Status | One-line | Gate logic |
|----|--------|----------|------------|
| **AC-1** Strategy is architectural | **in_progress** | Occupancy of abstraction→induction→retrieval heads predicts robustness; fragility is pathway bypass. | HP-15; Phase 2 / G2. |
| **AC-2** Installation | **designed** | Invariance can be installed by repairing stage 1 (steering / band-constrained FT) without collateral damage. | Runs **only if** G2 supports H6. |
| **AC-3** Training law | **designed** | Exposure **diversity**, not dose, drives invariance (forces abstraction-head formation). | DS-04 arms; H7 / G3. |

### Hypotheses (registry — know these cold)

| H | Status | Bet | Test | Gate |
|---|--------|-----|------|------|
| **H1** Labels track exposure | untested | Convergence labels recover held-out SEEN/UNSEEN | D1-lite LoRA + corrupted-answer oracle | **G1:** AUC ≥.75 freeze; .60–.75 revise once; <.60 pivot mechanism-first |
| **H2** W3 = persistent canonical binding | correlational | Rename drop is lingering binding, not lost algorithm | D2 can↔W3 residual patching | Patching restores W3 survival on fragile items |
| **H3** Fragility is a dose–response | open | Graded rename → family-consistent transfer curve | D3 / HP-10 ladders | Preregistered form fits or fails OOS |
| **H4** Invariance emerges late | planned | Acc rises before R_W3; lag tracks template diversity | D4 OLMo sweep | Lag wherever Acc is non-floor |
| **H5** Answers fixed before injection | open | High compliance + near-baseline Acc ⇒ answer set early | D2 injection-step patching | Patch at injection locus changes final; later patches do not |
| **H6** Strategy = pathway occupancy | pilot | Head occupancy predicts rename survival | HP-15 | **G2:** supported → AC-2; mixed → task-type boundary |
| **H7** Diversity forces abstraction heads | designed | Varied surfaces (dose fixed) raise R_W3 + head occupancy | DS-04 + AC-3 tracking | **G3:** diversity arm separates from dose-matched control |

### Phases (click each diamond)

| Phase | Status | Ships | Gate in one sentence |
|-------|--------|-------|----------------------|
| **P0** Foundations (wk 1–2) | **complete** | Intrusion table, strategy classes, frozen thresholds, MTMM | G0: validation + `strategy_posteriors.csv` + prereg committed |
| **P1** Calibration + mechanism flagship (wk 3–8) | **next** | **Paper I — the instrument** | **G1** on D1-lite ROC (AUC bands above) |
| **P2** Architecture (wk 9–14) | planned | **Paper II — mechanism** | **G2** H6 verdict |
| **P3** Laws (wk 15–22) | planned | **Paper III — behavioural + training laws** | **G3** forms fit/fail; H7 |
| **P4** Ecology & development (wk 23–30) | planned | **Paper IV — ecology of fragility** | Retention curves + G coefficients recoverable |
| **P5** Year-2 summit | planned | Applied audit suite | Papers I–II out; calibration intact |

### Likely question — “Which phase are you in?”

> “**Phase 0 is complete** — behavioural instrument shipped, thresholds frozen, derived tables regenerable. **Phase 1 is next**: D1-lite calibration. Gate G1 is whether behavioural labels recover known SEEN/UNSEEN exposure on a LoRA fine-tune — AUC at least .75 to freeze thresholds v2. Until that fires, I do not escalate to architecture claims that spend open-weight patching compute.”

**→ ↓ to Strategy catalogue.**

---

# 3. STRATEGY CATALOGUE  
**Rail:** Strategy catalogue · `#strategy`

### What is on screen

- Catalogue of deep strategies (DS / D cards).
- Callout: four-question **generative principle**.
- Live **vault force graph** (hover / drag). Caption: not a cost chart.

### What to do

1. Point at the four-question callout (read it).
2. Hover a few vault nodes; optionally drag one.
3. Name one concrete borrowed method: **item-response theory / mixture-IRT** (executed as DS-01).

### What to say

> “A lot of machine-psychology work borrows **tasks** from cognitive science — questionnaires. I think that borrows the wrong thing. What those sciences built was **inference machinery**: methods for diagnosing hidden processes in minds they could not open. That is the part worth borrowing.”

Four questions:

> “For any method from a science of hidden minds: (1) What hidden property did it infer? (2) What observable signature did it exploit? (3) What is the LLM analogue of that signature? (4) Does it give a **per-instance, theory-grounded** signal my instrument can use?”

Vault:

> “This is a live layout of the research vault — 110+ notes across 12 folders. Clusters run spine → measurement strategies → directions → papers. It is how I keep a multi-paper programme coherent.”

Example if asked:

> “Item-response theory infers a hidden ability from response patterns across items of varying difficulty, **per person**. The LLM analogue is inferring per-instance solving mode from response patterns across **controlled variants**. Mixture-IRT is already executed in this repo as DS-01.”

**→ ↓ to Research questions.**

---

# 4. RESEARCH QUESTIONS  
**Rail:** Research questions · `#research-questions`

### What to do

Expand **all five** cards in order. For each: read the question → what is measured → what is still gated.

### The five (filled from the site)

**Q1 — Which surface properties carry the load, and is the answer a property of the model or of the task?**  
- **Ops:** W1–W6 on 219 × ~6 models; Acc contrasts; Kendall W; shared-hard.  
- **Measured:** Numeric nearly free (Gemini .909→.958 W6) vs rename expensive (.523 W3); formal often costlier than rename (o4-mini .841→.682 W4); direction raises BW (Claude .172→.873); family-level transform order (W .42–.68); Claude across families W=.165 p=.82; shared-hard 0/110 ALGO, 0/20 GSM, 14/64 BW.  
- **Still gated:** Matched init/goal complexity for W5; difficulty-matched WIS for exposure vs hardness.  
- **Findings on site:** F1–F4.

**Q2 — Do behavioural indicators track known training exposure?**  
- **Ops:** D1-lite LoRA SEEN/UNSEEN + corrupted-answer oracle; ROC of labels.  
- **Measured so far:** Infini-gram proximity only — **proxy**, not closed-model membership. Labels **not** yet validated against known exposure.  
- **Gate G1:** AUC bands as in H1.  
- **Finding:** F5 (construct validity path).

**Q3 — Where in the network does the split live, and can invariance be installed?**  
- **Ops:** can↔W3 patching; Commitment Depth; HP-15; then AC-2 if supported.  
- **Measured:** Open-weight pilots solve too few renamed items (1/60, 0/61) for within-model correlation; CD informally predicted rank (r≈+.82) in a pilot.  
- **Open:** Need enough W3 solves before G2.

**Q4 — Dose or diversity of training surfaces?**  
- **Ops:** DS-04 arms (dose fixed, diversity varies) + head-emergence.  
- **Measured:** Not run yet.  
- **Gate G3.**

**Q5 — When does invariance emerge relative to accuracy?**  
- **Ops:** OLMo checkpoints + Dolma counts; Acc_can, R_W3, CD.  
- **Measured:** Not run; variance decomp already shows family-specific facet loadings (ALGO model-heavy; GSM item-heavy).

### How to present

> “Five questions organise everything. For each I track operationalisation, what the behavioural layer already measured, and **what remains gated**. The gated column is the important one — it is where my evidence stops.”

**→ ↓ to Methodology.**

---

# 5. METHODOLOGY  
**Rail:** Methodology · `#method`  
This is where a research interviewer spends the most time. Go slowly. The section has **4 blocks on one long page**: 5a bank · 5b probes · 5c verification · 5d stats. Scroll *within* the section with the trackpad only after you finish each block; do not ↓-skip past probes.

---

## 5a · The bank

### What is on screen

- Cards: **44** Arithmetic · **65** Planning · **110** Optimisation · **W1–W6** Surfaces (219 items total).
- Family tabs + variant chips → live problem text + gold + source CSV link.

### What to do (demo)

1. Leave **Arithmetic** + **Canonical** — read one GSM item aloud briefly.
2. Click **W3** — same gold, renamed surface (say what changed).
3. Switch family → **Planning** → **Canonical**, then **W5** — note gold is **re-derived**.
4. Switch → **Optimisation** — mention coin change ≫ shortest path ≫ WIS prevalence.

### What to say

> “Three families, chosen for **contrasting algorithmic character**. 44 arithmetic from GSM-Symbolic across proximity quartiles. 65 Blocksworld — 50 standard + 15 obfuscated — with **PDDL gold**. 110 optimisation. **219** total.”

Exposure gradient (the design jewel):

> “Optimisation is deliberate: coin change, shortest path, and weighted interval scheduling share DP structure but differ by **orders of magnitude** in public-text prevalence. Coin change is everywhere; WIS is rare. So I have an **exposure gradient inside the bank** while holding algorithmic family roughly constant.”

Surfaces:

> “Each problem has six surfaces, W1–W6. Gold is **fixed** for W1–W4 and **re-derived** for W5/W6, because those change the problem enough to change the correct answer.”

| Surface | What it does | Gold |
|---------|--------------|------|
| **W1** | Lexical paraphrase | Fixed |
| **W2** | Structural reformat / layout | Fixed |
| **W3** | Entity rename | Fixed (bound to new names) |
| **W4** | Formal notation / register | Fixed |
| **W5** | Direction invert (e.g. init↔goal in BW) | Re-derived |
| **W6** | New numbers / regenerated instance | Re-derived |

### Likely questions

**Why these families?**  
> “Arithmetic = short procedural computation. Planning = sequential state with verifiable goals. Optimisation = DP with an exact solver. Each allows **semantic** verification, and together they cover different kinds of solving.”

**Why 219, not 5,000?**  
> “Real limitation — I would not hide it. Controlled variants with semantic verification cost construction effort. 219 supports **directional** claims where effects are large (direction finding). It does not support fine-grained per-model rankings. With 10k items I would do factorial designs over perturbation types with tight intervals.”

---

## 5b · The three probes  
**UI:** Probe studio tabs **Probe 1 | Probe 2 | Probe 3**

### Principle first

> “Three probes, one at a time. **Labels issue only when probes agree.** A single probe is never enough to call an instance retrieval or computation.”

Why: each probe has its own failure mode; agreement is the same logic as multiple controls against every excuse.

---

### Probe 1 — Surface battery  
**Click Probe 1.** Point at fan viz + metrics (Acc_Wv, R_W3, floor Acc_can ≥ .30).

> “One problem → six answer-preserving forms. Which surface axis is the procedure keyed to? If the model executes a procedure, answer-preserving changes should not kill accuracy. If accuracy collapses under a specific change, that is what it was keying on.”

Template control:

> “Same instruction on every variant: solve step by step, give the final answer, then the problem text. Differences are the surface, not the prompt.”

Limitation (say before they do):

> “The battery cannot discharge **tokenisation noise** alone. I bound that with opposite-direction fragility across models — tokenisation noise would not reverse direction across models — and length-matched nonce controls.”

---

### Probe 2 — Plan vs execution ⭐ demo this live  
**Click Probe 2.**

**Layout:** header (what + metrics) → **stage rail** Declare | Execute | Inject | Score → toolbar (brief + Play / Step / Reset) → **prompt | board** split → footer (why isolate / why inject).

#### Live demo script (≈90 seconds)

1. Click **Reset**.
2. Click **Play stage** → watch Session A plan stream onto the board (one-shot declaration).
3. Button becomes **Next: execute →** → click / Play again → Session B prompts rebuild with **history only** (no plan text). Emphasise isolation.
4. **Next: inject →** → Play → highlight: environment reports **DP[2] = 9** (true was 2). Trace diverges (propagated misses).
5. **Next: score →** → CCI meter + pre-inject match / inject / recovered Count.

#### What to say while demoing

> “Session A declares a **full plan in one shot**. Session B is a **hard-isolated** new chat — it never sees the plan transcript; each turn gets the problem plus verified intermediates only. Mid-solve I can **inject a false state**. Then I score plan–execution consistency and recovery.”

Why two sessions:

> “If plan and execution share a transcript, agreement is cheap — the model can copy itself. Isolation makes **CCI** a fingerprint of whether the declared procedure is the one being followed.”

What CCI is (fill — interviewers ask this):

> “**CCI is plan–execution consistency**: positional agreement between declared intermediates and isolated execution. For numeric steps, agreement within **ε = 0.01**; for actions, string match. It is a **fingerprint**, not a proof of internal reasoning. High accuracy does **not** imply high CCI — Gemini GSM mean CCI about **.270**, Claude **.231**, GPT-4o **.108**.”

Injection:

> “Injection separates **compliance** from **correctness**. In the demo, DP[2] is overwritten 2→9. The model can adopt that false value — compliance — and still finish at Count = 2 — correctness. Those come apart in the data.”

---

### Probe 3 — Corpus proximity  
**Click Probe 3.** Point at scanning window viz.

> “Infini-gram n-gram windows over public corpora. This is a **within-family exposure proxy** — not membership inference for closed models. I cannot see what a closed model was trained on.”

**VRI** (they may ask):

> “**VRI**, variant robustness index, is mean accuracy on W1, W2, W4 minus W3 — how much rename hurts relative to other mild surfaces. Some models show proximity–VRI co-variation; o4-mini breaks that pattern. Within ALGO, prevalence CC ≫ SP ≫ WIS co-varies with some VRI patterns — a **partial** control, not isolation from difficulty.”

What would settle exposure:

> “Open-corpus models — OLMo on Dolma — where membership is known by construction, or D1-lite fine-tunes where I control exposure. That is Gate G1.”

---

## 5c · Verification

Point at the generator → verifier flow diagram.

> “Verification is **semantic** where the domain allows: PDDL simulator for planning — the plan must reach the goal; exact optimal solver for optimisation; numeric match for arithmetic. Failures are excluded **with a recorded reason**, never silently scored.”

Why not string match: a valid alternative plan would be marked wrong.

---

## 5d · Statistical treatment + operating rules

> “Optimisation: **10,000-draw percentile cluster bootstrap** over **51 near-duplicate families** covering 110 items. GSM/BW: **Wilson** intervals. Retention floor **Acc_can ≥ 0.30 fixed before analysis**. Kendall W against within-row rank permutation null — 10,000 perms, **seed 42**.”

Defend each choice:

| Choice | Why |
|--------|-----|
| Cluster bootstrap | 110 ALGO items are not independent; resampling items understates uncertainty. Resample **families**. |
| Wilson | Binomial intervals near 0/1; normal approx misbehaves. |
| Floor .30 preregistered | Retention is a ratio; tiny Acc_can makes it unstable — and I could not tune the cutoff post hoc. |
| Kendall W + permutation null | Concordance of rankings without strong distributional assumptions; fixed seed. |

### Operating rules (on site — say them)

1. Raw CSVs are truth; derived tables regenerate by script.  
2. External numbers quarantined until reproduced.  
3. Thresholds, metrics, kill criteria **pre-registered** before runs.  
4. **Gates** decide escalation — no skipping ahead on excitement.  
5. Ship **narrow** papers; breadth lives in the vault.  
6. Reasoning sessions never invent numbers; execution sessions never interpret.  
7. Every session writes back to Programme State.

**→ ↓ to Experiments.**

---

# 6. EXPERIMENTS EXECUTED  
**Rail:** Experiments · `#experiments`

### What to do

Use filters: **All / Probes / Analyses / Mechanistic / Complete / Pilot**.  
Expand 2–3 cards live (recommended: **P1**, **P2B**, **CCI×W3** or **Shared-hard**). Emphasise **Rules out**.

### Card cheat-sheet

| ID | Status | n / scope | Metric | Artefact | Speakable result |
|----|--------|-----------|--------|----------|------------------|
| **P1** | complete | 219 × ~6 × 7 surfaces | Acc_Wv, R_W3 | `probe1_per_model_variant.csv` | Lexical vs numeric axes dissociate; formal & direction distinct |
| **P2A** | complete | GSM 44/model; ALGO varies | CCI ε=.01 | `GSM_P2_metrics.csv` | Gemini CCI .270; Claude .231; GPT-4o .108 — accuracy ≠ consistency |
| **P2B** | complete | 61 ALGO sessions/model | Acceptance; post Acc | `ALGO_P2_phase2_injected.csv` | o4-mini 100% accept; Claude 88.5%; post≈baseline |
| **P3** | complete | per canonical | c̄_T, instance | `*_P3_contamination.csv` | Proximity–VRI for some; o4-mini breaks it |
| **Triangulation** | complete | ALGO 440 rows v3 | convergence_label | `ALGO_P3_triangulation_v3.csv` | Mostly ambiguous; thresholds swing labels hard |
| **Shared-hard** | complete | 110/20/64 five-model | fail_all_five | `P1_failure_patterns.csv` | **0/110 ALGO, 0/20 GSM, 14/64 BW** |
| **Kendall W** | complete | W1–W6 ranks | W, p | `P1_variant_ordering.csv` | Within .42–.68; Claude across .165 p=.82 |
| **Intrusion** | complete | rename-error subsets | intrusion rate | `T2_intrusion_detail.csv` | o4-mini ALGO .619 (13/21); BW 0 |
| **Mixture-IRT** | complete | GSM/ALGO matrices | K by BIC | `strategy_posteriors.csv` | GSM single class; ALGO/BW underpowered |
| **O10 variance** | complete | binary Acc designs | facet σ² | `O10_variance_components.csv` | ALGO model-heavy; GSM item-heavy |
| **Mechanistic pilot** | pilot | Qwen-2.5-7B | gold-token rank | mechanistic raw | CD signal; underpowered for W3 |
| **CCI×W3** | complete | pooled n=272 | r, CI, p | `P2_P1_convergence.csv` | Pooled r=.243 p=.024; per-family null |

> “Each card is a real sweep: models, n, primary metric, derived artefact. Expanding shows design notes and **what the result rules out** — that column is how I talk about results.”

**→ ↓ to Finding 1.**

---

# 7. THE SIX FINDINGS  
Present in order. For each: **claim → number → meaning → likely objection**.  
Depth panels under each chart (Design / Reads as / Rules out) — click/expand if present.

---

## F1 · Numeric vs lexical robustness dissociate  
**Rail:** F1 · `#f1`  
**Chart:** slope can → W6 → W3 · **CSV:** `probe1_per_model_variant.csv` · **n=44** (GPT-4o/Llama often n=20 valid)

### What to do

Let the slope animate; hover Gemini line if useful.

### Say

> “On arithmetic, canonical → **numeric** change (W6) stays flat; canonical → **lexical** rename (W3) falls. Gemini: **.909 / .958 / .523**.”

Meaning:

> “Numeric robustness and lexical robustness are different things. Changing numbers does not hurt — not simple memorisation of specific numeric answers. Changing words hurts — the procedure is keyed to lexical surface more than arithmetic structure.”

If “why does W6 slightly help?”:

> “.909 to .958 I treat as **flat** given the sample. The finding is the **contrast with W3**, not the small numeric gain.”

Other models (if asked): Claude .841/.75/.75; o4-mini nearly flat; GPT-4o .825/.825/.35; Llama .825/.525/.2.

**→ ↓ F2.**

---

## F2 · Formal notation costs more than renaming  
**Rail:** F2 · `#f2`  
**Tabs:** Arithmetic | Planning · **n_GSM=44, n_BW=65** · same CSV

### What to do

Show **Arithmetic** tab first; optionally flip to **Planning**.

### Say

> “W4 keeps entity names intact and only changes **register** into formal notation. **o4-mini alone loses 16 points under W4** — .841 → .682 — while W3 on that cell stays .841. Changing how a problem is written, with no change to what it is about, can cost more than renaming entities.”

**→ ↓ F3.**

---

## F3 · Direction of reasoning in planning ⭐ lead finding  
**Rail:** F3 · `#f3`  
**Demo:** towers + model select + **Invert direction (W5)** · **n=65** · CSV `C7_concordance_filtered.csv` (+ paper BW table)

### What to do (must be live)

1. Select **Claude** (default Acc ~.172 canonical).
2. Click **Invert direction (W5)** — bar jumps; label updates.
3. Optionally flip Gemini / o4-mini.

Family-level numbers (manuscript / site):

| Model | Canonical | W5 |
|-------|-----------|-----|
| Claude | **.172** | **.873** |
| Gemini | **.391** | **.764** |
| o4-mini | **.781** | **.909** |

### Say

> “This is the result I find most striking. Invert the Blocksworld problem — swap start and goal — and accuracy **jumps**. Claude .172 → .873. Genuine planning should not care which direction the problem is posed. A dramatic asymmetry is evidence of matching a familiar solution pattern rather than searching the state space.”

### Objection you must have cold — “Isn’t the inverted problem just easier?”

> “That is the right challenge — and why I use controlled variants of the **same** problem. In Blocksworld, actions are reversible: pick up / put down, stack / unstack. The reverse of a valid plan is a valid plan for the inverted problem, **same length**. Optimal plan length — the standard difficulty measure — is preserved. So it is not structurally easier by that measure, yet accuracy jumps.”

⚠ Before the call: confirm W5 construction preserves optimal plan length on your PDDL gold for the scored set; know any exceptions. Confirm figures are **family-level**, not only the BW_014 demo instance. Confirm these numbers are from **post file-routing-bug-fix** data if that audit touched BW.

Benign explanation (raise yourself):

> “One benign story: inverted direction may resemble how BW solutions appear in training text. That is still compatible with my interpretation — performance depends on similarity to seen solutions. Corpus proximity is how I would pressure-test that.”

**→ ↓ F4.**

---

## F4 · Fragility is a model × item interaction  
**Rail:** F4 · `#f4`  
Three panels: Shared-hard · φ heatmap · Kendall W

### Numbers (filled)

**Shared-hard** (`P1_failure_patterns.csv`, five primary models):  
- Optimisation **0/110**  
- Arithmetic **0/20**  
- Planning **14/64** (all obfuscated)

**Kendall W** (`P1_variant_ordering.csv` / C7 companion):  
- Within family: Opt **W=.419** p=.049; Planning **W=.679** p=.0001; Arith **W=.66** p=.042  
- Across families within Claude: **W=.165** p=.824 (not distinguishable from null)

### Say

> “Fragility is not only a model property or only an item property — it is an **interaction**. Almost no shared-hard items outside obfuscated planning. Models share **family-level** transform ordering, but fail **different** items. You cannot label an item ‘retrieval-solved’ in the abstract; it depends on the model. That is why the instrument is per-instance and per-model.”

**→ ↓ F5.**

---

## F5 · Robustness ≠ accuracy  
**Rail:** F5 · `#f5`  
**Toggle:** Per-family (both null) | **Pooled (significant)** · CSVs `P1_construct_validity.csv`, `P2_P1_convergence.csv`

### What to do

Start on **Per-family**. Then flip to **Pooled** and **immediately** explain Simpson risk.

### Numbers

- Acc vs retention: **ρ=.136**, p=.642  
- Acc vs φ: **ρ=−.426**, p=.146  
- CCI vs W3 retention: Arithmetic r=.156 p=.06 n=128; Optimisation r=.359 p=.085 n=144; **Pooled r=.243** [.027,.414] **p=.024 n=272**

### Say

> “Accuracy does not predict robustness within a family — those correlations are null. A more accurate model is not, in this data, a more robust one.”

**Trap 1 — pooled significant:**

> “I do **not** lean on the pooled CCI×W3 correlation as primary. Pooling across families can create a correlation because families differ in average accuracy and average robustness — Simpson territory. **Within-family nulls** are the relevant test.”

**Trap 2 — absence of evidence:**

> “A null correlation here is **absence of evidence** of a relationship, not proof of independence. I say: accuracy does not predict robustness in this data. I do **not** say they are proven independent.”

**→ ↓ F6.**

---

## F6 · Compliance and correctness are separable ⭐ lead finding  
**Rail:** F6 · `#f6`  
Trace animation + accept-rate cards · **n=61** · `injection_recovery_summary.csv` / paper AUDIT

### Acceptance / post-injection Acc / baseline (speakable)

| Model | Accept | Post-inj Acc | Baseline |
|-------|--------|--------------|----------|
| Claude | **88.5%** | .525 | .541 |
| GPT-4o | **93.4%** | .508 | .557 |
| o4-mini | **100%** | .377 | .409 |
| Llama | 39.3% | .23 | — |
| Gemini | **0%** (treats injection as malformed) | .311 | — |

### Say

> “Sixty-one ALGO injection sessions: models often **adopt** the false intermediate — compliance — and still land near the **uninjected** final accuracy — correctness. If the model were faithfully executing the visible procedure, a corrupted intermediate should poison the answer. Reaching the right answer anyway suggests the final answer is not actually computed from the visible steps. Visible reasoning and answer production are at least partly **decoupled**.”

Systems punchline:

> “Anyone using chain-of-thought as an *explanation* of behaviour should care. If the answer is robust to corrupting the visible reasoning, that reasoning is not a faithful account of how the answer was produced.”

**→ ↓ Future methods.**

---

# 8. FUTURE METHODS  
**Rail:** Future methods · `#next`

### What to do

Filter **Tier 1** (flagship). Expand **D1-lite**, then **D2 patching**. Mention Tier 2 OLMo if they are systems-oriented.

### Tier 1 — flagship (say with kill criteria)

1. **D1-lite calibration (Gate G1)** — Do indicators recover known exposure? LoRA SEEN/UNSEEN on Llama-3.1-8B + corrupted-answer oracle. **Kill / branch:** AUC ≥.75 / .60–.75 / <.60 as in H1.  
2. **D2 causal patching + Commitment Depth** — Is W3 a binding failure? Are answers fixed before injection (H5)?  
3. **Source monitoring (CC-2 / DS-05)** — Can models report memory source? Prediction often d′≈0.

### Tier 2 — geometry & development

4. D3 continuous transfer distance (H3)  
5. D4 developmental checkpoints (H4 / CC-3) — **OLMo intermediate checkpoints**  
6. Matched WIS bank (exposure vs difficulty)  
7. Direction follow-up (branching vs search-direction)

### Tier 3 — fold-ins

8. Exposure diversity vs dose (H7 / AC-3) — Gate G3  
9. Sampling-consistency pilot (kill if |r| with CSS > .8)  
10. HP-15 pathway occupancy — Gate G2 → AC-2 or task-type boundary

> “Every protocol has **predictions** and a **kill criterion**. Writing down what result would kill a hypothesis is what stops a programme from becoming unfalsifiable — and how I decide what **not** to spend compute on.”

**→ ↓ Links (or stop).**

---

# 9. CLOSE / LINKS  
**Rail:** Links · `#footer`

> “Everything here traces to the vault and the repo: THE_PLAN, programme state, and the CSVs behind every number. If any number on the site disagrees with a CSV, **the CSV wins**.”

Offer: GitHub · Paper PDF · BOOK.md · Research vault · THE_PLAN · Issues.

---

# PART A — SELF-CORRECTION STORY (volunteer; do not wait)

> “During camera-ready I found and corrected real issues myself: a **file-routing bug** that affected Blocksworld results; a mechanistic finding that did not survive scrutiny, which I **retracted**; and a temperature-zero determinism claim I could not fully substantiate, which I **withdrew**. Documented in a self-audit appendix.”

Why volunteer: publishing your own bugs is evidence of rigour.

⚠ Be ready: whether **Finding 3 on this site** is from the **corrected** BW data.

Open vulnerability (name it):

> “Convergence labels — retrieval vs computation calls — were never independently validated against ground-truth exposure by a second method. **That is Gate G1 / D1-lite — the next thing I close.**”

---

# PART B — SARVAM / SYSTEMS BRIDGE

> “The practical consequence is for any decision made on benchmark accuracy: choosing a checkpoint, accepting a quantisation, picking a data mixture. If two models reach the same score through different processes, the benchmark cannot see that. Quantisation is sharp: a post-training quantised model can match the original on aggregates while changing *how* it solves problems — visible only under distribution shift. The instrument is meant to detect exactly that hidden difference.”

---

# PART C — TWO PROJECTS, ONE AGENDA (if OCR / VLM also comes up)

> “Both projects are the same research question in two domains. In reasoning: same accuracy, different processes — accuracy cannot separate them. In OCR: same high confidence whether or not the image supported the output — confidence cannot separate them. In both cases an aggregate hides the mechanism; the fix is an instrument that runs **controlled counterfactuals** and diagnoses process directly.”

| | Reasoning (RvC) | OCR instrument |
|--|-----------------|----------------|
| Aggregate | Accuracy | Max-softmax confidence |
| Hides | Retrieval vs computation | Visual grounding vs language prior |
| Counterfactual | Surfaces, state injection | Blank / device / ablation |
| Key control | Direction invert (W5) | Position 0 |
| Honest limit | n=219; closed-model exposure | Synthetic renders; small model |

---

# PART D — COMPRESSED 5-MINUTE PATH

| Time | Section | Do | Say |
|------|---------|----|-----|
| 0:00 | Motivation | Stay | Opener: procedure vs pattern; per-instance object |
| 0:30 | Programme | Click P0 / P1 | Gated phases; currently P1 / G1 next |
| 1:15 | Method 5a | Flip W3 chip | 219; exposure gradient; W1–W6 |
| 2:00 | Probe 2 | Play Declare→Score | Isolation + inject; CCI; compliance≠correctness |
| 3:00 | F3 | Invert W5 on Claude | .172→.873; plan length preserved |
| 4:00 | F6 | Point accept row | o4-mini 100% accept, post≈baseline |
| 4:30 | Close | — | Systems bridge **or** self-audit + G1 next |

Skip: full claim map, all five RQs, F1/F2/F4/F5 detail, future tiers (mention G1 only).

---

# PART E — PRE-CALL CHECKLIST

- [ ] Site loads; rail + ↓/j work  
- [ ] Probe 2 Play stage through Score once  
- [ ] F3 invert demo on Claude  
- [ ] F5: know Simpson story for pooled vs per-family  
- [ ] W5 plan-length preservation verified for scored BW set  
- [ ] F3 numbers = post-bug-fix data  
- [ ] Self-audit wording matches appendix  
- [ ] Tabs open: paper PDF, `probe1_per_model_variant.csv`, injection summary  
- [ ] One sentence: current phase = **P0 complete, P1 next, waiting on G1 (D1-lite AUC)**  
- [ ] Definitions cold: **W1 paraphrase, W2 reformat, CCI, VRI, ε=0.01**

---

# QUICK GLOSSARY (site language)

| Term | Meaning |
|------|---------|
| **Instrument** | Validated per-instance measurement before claims |
| **CCI** | Plan–execution consistency; positional match; ε=0.01 numeric |
| **TEP** | Trace / execution path metrics after corruption (injection analyses) |
| **VRI** | mean(W1,W2,W4) − W3 |
| **R_W3** | Acc_W3 / Acc_can (retention); floor Acc_can≥.30 |
| **Convergence label** | Per-instance retrieval / computation / mixed / ambiguous from probe agreement |
| **Gate G1/G2/G3** | Escalation criteria — see Phase timeline |

---

*Source of truth for numbers: `results/derived/*.csv` and the live site captions. If they disagree, the CSV wins.*
