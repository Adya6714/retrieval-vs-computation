# How to Contribute the Problem Set — Full Guide

This document tells you everything you need to do to complete Phase 1 of the project.
Read the whole thing before starting. Do not skip sections.

---

## What you have

You are designing problems from scratch using canonical benchmark sources. You do not have a pre-compiled doc. You will pull problems directly from PlanBench (Blocksworld), GSM-Symbolic, and standard shortest-path graph benchmarks. Do not reuse anything from the pilot run.

---

## What you are building

Three things:

1. `data/probe1_instances.csv` — problems for Probe 1
2. `data/probe2_instances.csv` — problems for Probe 2
3. `probes/contamination/verify.py` — script that checks if a model's answer is correct

---

## Step 1 — Source your problems

Blocksworld: pull from PlanBench repository. Pick instances with 3–6 blocks. Record the source instance ID exactly.
GSM: pull from GSM-Symbolic dataset (Mirzadeh et al.). Pick instances across different templates. Record template name and instance ID.
Shortest path: construct small graphs (6–10 nodes) manually or from the algorithmic reasoning benchmark. Document the graph structure in source_instance_id.
Procedurally generated (contamination_pole = low): write a small Python script with a fixed random seed to generate these. Commit the generation script alongside the CSV.

---

## Step 2 — Build `data/probe1_instances.csv`

These are problems for Probe 1 (Surface Invariance). The model will be tested on
whether it gives the same answer when the problem is rephrased. So every problem
here must have one unambiguous correct answer.

**Columns:**

| Column | What to put |
|--------|-------------|
| `problem_id` | Unique ID you make up. Format: `FAMILY_NNN` e.g. `BW_001`, `SP_001`, `GSM_001` |
| `problem_family` | One of: `blocksworld`, `shortest_path`, `gsm` |
| `problem_text` | Full problem statement, exactly as given to a model |
| `correct_answer` | Clean machine-readable answer. See format rules below. |
| `source_dataset` | Where this came from: `PlanBench`, `GSM-Symbolic`, `custom_generated`, etc. |
| `source_instance_id` | ID in the source dataset if it has one. If generated, write `generated_seed_N` |
| `contamination_pole` | `high` if from a known benchmark. `low` if procedurally generated. |

**Correct answer format rules — follow exactly:**

- Math / GSM problems: just the number, no units, no words. `42` not `$42` not `42 dollars`
- Shortest path (asking for the path): node sequence comma-separated. `A,C,D,F`
- Shortest path (asking for the distance): just the number. `7`
- Blocksworld: one move per line, format strictly `move X from Y to Z`. Example:
```
  move A from table to B
  move C from D to table
```

**How many:** 15 problems total. 5 per family. At least 5 must be contamination_pole = low with a committed generation script and documented seed. Zero problems from pilot run or ChatGPT compilation.

---

## Step 3 — Build `data/probe2_instances.csv`

These are Blocksworld problems only, for Probe 2 (Plan-Execution Coupling).

**Columns:**

| Column | What to put |
|--------|-------------|
| `problem_id` | Format: `P2_BW_001` |
| `problem_text` | Full problem statement as given to model |
| `initial_state` | Starting configuration. E.g. `A is on B, B is on table, C is on table` |
| `goal_state` | Goal configuration. E.g. `C is on B, B is on A, A is on table` |
| `correct_plan` | One move per line, format: `move X from Y to Z` |
| `num_blocks` | Integer. |
| `source_dataset` | Where this came from |
| `source_instance_id` | ID in source or `generated_seed_N` |
| `contamination_pole` | `high` or `low` |

**How many:** 10 problems. Range from 3 blocks to 6 blocks. At least 3 generated.

**Validating a Blocksworld problem:**
Draw the initial state on paper. Apply each move in `correct_plan` one by one.
Verify you reach `goal_state`. If you cannot, the problem is invalid — flag it in the PR.

Rules for a legal move:
- You can only move a block that has nothing on top of it
- You can place a block on the table or on top of another block that has nothing on top of it

---

## Step 4 — Write `probes/contamination/verify.py`

This script takes a model's raw text response and a ground truth correct answer,
and returns `True` if the model got it right, `False` if not.

This is needed because models give messy answers like "The answer is **42**." and
you need to extract `42` and compare to ground truth automatically.

**Write the following function:**

```python
def verify_answer(
    model_response: str,
    correct_answer: str,
    problem_family: str,
) -> bool:
    """
    Returns True if model_response contains the correct answer.

    problem_family: one of 'gsm', 'shortest_path', 'blocksworld'
    correct_answer: clean ground truth from the CSV
    model_response: raw text response from the model
    """
```

**Logic per family:**

For `gsm`:
- Extract all numbers from `model_response`
- The model's answer is the last number found
- Compare to `correct_answer` as a number (not string — `42.0 == 42`)
- Return True if they match

For `shortest_path`:
- If `correct_answer` is a number (distance): extract last number from response, compare
- If `correct_answer` is a path like `A,C,D,F`: check if that exact sequence appears
  anywhere in the response (case-insensitive)

For `blocksworld`:
- Split `correct_answer` into individual moves (one per line)
- Check if each move appears in the response (case-insensitive, order matters)
- Return True only if all moves appear in the correct order

**Important:** this function should never crash. If it cannot parse the response,
return `False` and print a warning with the problem_id so it can be reviewed manually.

---

## Step 5 — Test your verify.py

Before submitting the PR, test it manually with these cases:

```python
# GSM — should return True
verify_answer("The total is $42.", "42", "gsm")

# GSM — should return False  
verify_answer("The total is $43.", "42", "gsm")

# Shortest path — should return True
verify_answer("The shortest path is A, C, D, F with distance 7.", "A,C,D,F", "shortest_path")

# Blocksworld — should return True
verify_answer(
    "Step 1: move A from table to B\nStep 2: move C from D to table",
    "move A from table to B\nmove C from D to table",
    "blocksworld"
)
```

Add these as a simple `if __name__ == "__main__"` block at the bottom of the file
so anyone can run `python probes/contamination/verify.py` and see pass/fail.

---

## Step 6 — Submit the PR

1. Fork the repo
2. Add all three files
3. Open a Pull Request titled: `[data+verify] Phase 1 problem set and answer verifier`
4. In the PR description write:
   - How many problems per family in each CSV
   - Which problems are generated and what seeds were used
   - Which test cases in verify.py pass
   - Anything you were unsure about — flag it, do not guess

---

## If you get stuck

| Problem | What to do |
|---------|------------|
| Answer format unclear | Add a note in the PR, I will decide |
| Blocksworld move seems illegal |
| verify.py can't handle some response format | Return False, print a warning, flag in PR |
| Missing problem families in the doc | Flag in PR, do not invent problems |
| Anything else | Flag in PR, do not guess |