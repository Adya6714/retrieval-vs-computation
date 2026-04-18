# Shaswat's Task List

## What this project is (in plain language)

We are studying whether LLMs actually reason through problems or just recognize them from training data. We compare models on the same problems dressed up in different ways, and check if their behavior matches what you'd expect from real reasoning.

Your role: **decide what problems we use and why**. You own the final problem set design — which families, which instances, what difficulty distribution. When a reviewer asks "why is coin change in there," your `INSTANCE_SELECTION_CRITERIA.md` should answer it.

---

## Important: the family structure is NOT finalized yet

The current proposal (from the research design doc) is:

- **Family 1 — Planning Suite:** Blocksworld (8) + Logistics (4) + Mystery Blocksworld (3)
- **Family 2 — Arithmetic Reasoning:** GSM-Symbolic standard (8) + GSM-P1/P2 (4) + GSM-NoOp (3)
- **Family 3 — Algorithmic Suite:** Shortest Path (4) + WIS (4) + Coin Change (4) + Knapsack (3)

These are proposals. You can revise sub-types, swap families, or propose alternatives. The constraints that **cannot change**:
- Exactly 3 families
- Exactly 15 problems per family (45 total)
- Each family must support Probe 1 (surface invariance) and Probe 3 (contamination scoring)
- Only the Planning Suite — specifically the Blocksworld subset — supports Probe 2 (plan-vs-execute)

**Open decisions explicitly handed to you:**
1. Is Mystery Blocksworld worth including? It's scientifically strong — it's a built-in retrieval test at the domain level, because a model relying on memorized patterns will fail when predicate names are replaced with nonsense words. Kambhampati's prior work shows large accuracy drops under this obfuscation. But it adds complexity to verification and variant writing. Your call.
2. Is Logistics worth including, or does it add too little beyond Blocksworld?
3. For the Algorithmic Suite: is 4+4+4+3 (SP/WIS/CoinChange/Knapsack) the right split?
4. Exact difficulty distribution per family (how many easy/medium/hard)
5. Exact contamination balance per family — target is 5 high + 5 medium + 5 low per family, but you can adjust within a family if there's a good reason

---

## Background you need to understand the decisions

**What contamination tiers mean:**
- **High:** Classic textbook phrasings from a known benchmark (PlanBench, GSM-Symbolic, standard algorithm textbooks). Model almost certainly saw these during training.
- **Medium:** Adapted versions — same problem type, modified phrasing or parameters. Possibly seen.
- **Low:** Procedurally generated with a fresh random seed, or obfuscated (Mystery Blocksworld). Unlikely seen verbatim.

We need this spread so Probe 3's regression has dynamic range. If everything scores high contamination, we can't measure the effect.

**What variant types apply to which families** (affects how you design instances):

| Variant | Planning Suite | Arithmetic | Algorithmic |
|---------|----------------|------------|-------------|
| W1 — Lexical paraphrase | All | All | All |
| W2 — Structural reformat | All | All | All |
| W3 — Entity rename | All | All | All |
| W4 — Formal notation | BW + Logistics only | Not applicable | All |
| W5 — Procedural regeneration | All | All | All |
| W6 — Reversal | Blocksworld only | Not applicable | SP + Coin Change only |

W4 and W6 are partial — reported separately from the main CSS computation, not pooled.

**What makes an adversarial instance:** For the Algorithmic Suite, adversarial means the instance is specifically constructed so that a common heuristic (greedy, nearest-neighbor, earliest-deadline-first) fails to find the optimal answer. This forces genuine computation. Example: Coin Change with denominations {1, 3, 4} and target 6 — greedy picks 4+1+1=3 coins, but optimal is 3+3=2 coins.

---

## Your tasks, in order

### Task 1: Decide and justify the final family composition
**Deadline:** Before any instance selection begins. Discuss with Adya first.
**Priority:** High — everything blocks on this.

Work through the open decisions. For each sub-type you include: one sentence justifying why it belongs (what it contributes that another sub-type doesn't). For each you cut: one sentence why.

**Deliver:** A message to Adya with your final call. Once Adya signs off, this is locked.

---

### Task 2: Write `team/INSTANCE_SELECTION_CRITERIA.md`
**Deadline:** Before instance selection begins.
**Priority:** High

This is a rules document, not a design document. Adya reviews every instance against it. Paper reviewers may ask for it.

**Must specify, per family:**

**Difficulty tier definitions** with exact parameters. Examples of what this means:
- Blocksworld easy: 3–4 blocks, goal requires ≤ 4 moves, no block needs to be moved more than once
- Blocksworld medium: 5–7 blocks, goal requires 5–8 moves
- Blocksworld hard: 10–12 blocks, goal requires ≥ 9 moves, at least one block must be unstacked and restacked
- GSM easy: 2–3 arithmetic steps
- GSM medium: 4–5 steps with intermediate quantities
- GSM hard (P1/P2): depth-increased, 6+ steps
- Algorithmic easy: standard instances, greedy heuristic works
- Algorithmic hard: adversarial instances, greedy fails

You define the exact thresholds. The point is that two people reading your doc should classify the same instance the same way.

**Contamination tier definitions** — which sources count as high/medium/low, what "adapted" means precisely.

**Answer unambiguity requirements** — e.g., shortest path instances must have a unique optimal path (if two paths tie on cost, exclude the instance). Blocksworld: unique optimal plan, or a bounded set of valid plans that are all accepted by the verifier.

**Source documentation format** — for every instance: source name, instance ID or random seed, URL if applicable. No instance enters the CSV without this.

**Exclusion criteria** — what disqualifies an instance. At minimum: ambiguous answer, trivial solution (solvable by inspection in under 5 seconds), answer depends on unstated tiebreaker rules.

**Review checklist** — a yes/no list checked before any instance enters the CSV.

---

### Task 3: Select instances and build the CSVs
**Deadline:** After Task 2 is approved. Nandini coordinates the work split.
**Priority:** High — main deliverable.

**Sources:**
- Blocksworld: PlanBench repository. Record exact instance ID.
- GSM: GSM-Symbolic dataset (Mirzadeh et al.). Record template name and instance ID.
- Algorithmic: construct manually or from standard benchmarks. Document graph structure fully.
- Low-contamination (procedural): needs a Python script with fixed random seed. **Loop in Adya for the code part.** She has done this before for Blocksworld and can extend it.

**File 1: `data/problems/probe1_instances.csv`**

| Column | What to put |
|--------|-------------|
| `problem_id` | `BW_001`, `GSM_001`, `SP_001`, etc. |
| `problem_family` | `blocksworld`, `logistics`, `mystery_blocksworld`, `gsm`, `shortest_path`, `weighted_interval_scheduling`, `coin_change`, `knapsack` |
| `problem_text` | Full statement, exactly as given to a model |
| `correct_answer` | Machine-readable — see format rules below |
| `difficulty` | `easy`, `medium`, `hard` |
| `source_dataset` | `PlanBench`, `GSM-Symbolic`, `custom_generated`, etc. |
| `source_instance_id` | ID in source, or `generated_seed_N` |
| `contamination_pole` | `high`, `medium`, `low` |
| `verifier_function` | Python function name: `verify_numeric`, `verify_blocksworld_plan`, etc. |
| `notes` | Anything reviewers should know |

**File 2: `data/problems/probe2_instances.csv`** (Blocksworld only, 8 problems — can overlap with probe1)

| Column | What to put |
|--------|-------------|
| `problem_id` | `P2_BW_001`, etc. |
| `problem_text` | Full statement |
| `initial_state` | e.g. `A is on B, B is on table, C is on table` |
| `goal_state` | e.g. `C is on B, B is on A, A is on table` |
| `correct_plan` | One move per line: `move X from Y to Z` |
| `num_blocks` | Integer |
| `difficulty` | `easy`, `medium`, `hard` |
| `source_dataset` | Source |
| `source_instance_id` | ID or seed |
| `contamination_pole` | `high`, `medium`, `low` |

**Answer format rules — follow exactly:**
- GSM / any numeric: just the number. `42` not `$42` not `42 dollars`
- Shortest path (path): comma-separated nodes. `A,C,D,F`
- Shortest path (distance): just the number. `7`
- Blocksworld: one move per line, `move X from Y to Z`
- WIS / Knapsack / Coin Change: optimal value as a plain number

**Before submitting any Blocksworld problem:**
Draw the initial state. Apply each move one by one. Verify you reach the goal state. If you can't, the problem is invalid.
Legal move rules: you can only move a block with nothing on top of it; you can place it on the table or on a block with nothing on top of it.

---

## How to save your work to the repo (Git 101)

1. `git pull` — get latest changes
2. Do your work
3. `git status` — see what changed
4. `git add <filename>` for each changed file
5. `git commit -m "Short description"`
6. `git push`

If you see a merge conflict: **text Adya immediately. Do not try to fix it yourself.**

Edit CSVs in Excel or Google Sheets, export as CSV. Do not open CSVs in a plain text editor.

---

## Finished tasks
*(none yet)*

## Questions log
*(add questions here so nothing gets asked twice)*