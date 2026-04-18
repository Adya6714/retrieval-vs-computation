# Nandini's Task List

## What this project is (in plain language)

We are studying whether LLMs actually reason through problems or just recognize them from training data. We compare models on the same problems dressed up in different ways, and check if their behavior matches what you'd expect from real reasoning.

Your role: **coordinate the work and own the cross-review process**. Once Shaswat finalizes the problem families and criteria, you figure out who picks which instances, how the work splits three ways, what the timeline looks like, and how you make sure nothing enters the CSV without a second pair of eyes.

---

## Important: the family structure is NOT finalized yet

Shaswat is currently working through the final family composition (Task 1 in his doc). Until he finishes that and Adya signs off, instance selection cannot begin. Use this time to read through his task list and this document so you understand the full picture before coordination starts.

The current proposal is three families: Planning Suite (Blocksworld + Logistics + Mystery Blocksworld), Arithmetic Reasoning (GSM-Symbolic variants), Algorithmic Suite (Shortest Path + WIS + Coin Change + Knapsack). These are proposals, not final. Shaswat may revise them.

The constraints that are fixed regardless: 3 families, 15 problems each, 45 total.

---

## Your tasks, in order

### Task 1: Build the work plan once Shaswat finalizes families
**Deadline:** Within 3 days of Shaswat + Adya locking the family structure.
**Priority:** High

Divide the instance selection work across the three of you. The goal is 45 problems total with clear ownership per chunk.

**What you should deliver** (a message to Adya and Shaswat):
- Who picks which sub-types (e.g. Adya takes Blocksworld, Shaswat takes GSM, Nandini takes Algorithmic)
- Estimated timeline per chunk — be realistic, 15 instances with proper source documentation and hand-validation takes time
- How you will track progress (a shared Google Sheet is fine)

**Things to factor in:**
- Adya owns all code. Any instance that needs a generation script (low-contamination procedural instances) goes to Adya after the source+seed decision is made. You don't need to write code.
- Probe 2 needs 8 Blocksworld instances specifically. These can overlap with the 8 Blocksworld instances in Probe 1. Whoever takes Blocksworld handles both CSVs for those instances.
- Each sub-type needs someone who will actually read the source benchmark before selecting — not just copy-paste. Factor that reading time in.

---

### Task 2: Own the cross-review process
**Deadline:** Ongoing, starting when first instances are submitted.
**Priority:** High

Every instance that enters the CSV must be reviewed by at least one person who did not select it. You make this happen.

**How the process works:**
1. Someone (Shaswat, you, or Adya) adds instances to the CSV and opens a pull request on GitHub.
2. You assign a reviewer — cannot be the same person who added those instances.
3. Reviewer checks against `INSTANCE_SELECTION_CRITERIA.md`: is the answer unambiguous? Is the source documented? Is the difficulty tier correctly assigned? Is the correct answer in the right format?
4. If issues found: reviewer comments on the PR, selector fixes before merge.
5. You log what's been reviewed.

**What you should deliver:** A simple review log tracking: problem_id, selector, reviewer, status (pending / approved / needs fix), date. A shared Google Sheet works fine.

---

### Task 3: Variant writing — W1 and W2 for your assigned sub-types
**Deadline:** After `CONTRIBUTING_VARIANTS.md` is written by Adya and instance selection is complete.
**Priority:** Medium — starts after instance selection closes.

**What variants are:**
Each canonical problem gets rewritten in multiple ways (called variants). The variants test whether models give the same answer when the same problem is phrased differently.

- **W1 — Lexical paraphrase:** Same problem, completely rewritten wording. Same numbers, same answer, different sentence structure and vocabulary.
- **W2 — Structural reformat:** Same information, different format. For graph problems: convert prose description to a table or JSON. For word problems: convert to a structured key-value layout.

**Rules:**
- The correct answer must be identical to the canonical problem's answer
- No LLM may be used to generate variants — human-written only
- Another person must be able to independently compute the same answer from your variant
- W1: wording must be genuinely rewritten, not just synonym substitution sentence by sentence

**W2 example for a shortest path problem:**

Original: "There is a network with nodes A, B, C, and G. The edge from A to B costs 3. The edge from A to C costs 1. The edge from B to G costs 4. The edge from C to G costs 6. What is the shortest path from A to G?"

W2 (table):
```
| From | To | Cost |
|------|----|------|
| A    | B  |  3   |
| A    | C  |  1   |
| B    | G  |  4   |
| C    | G  |  6   |

Find the path from A to G with minimum total cost.
```

W2 (JSON):
```json
{
  "nodes": ["A", "B", "C", "G"],
  "edges": [
    {"from": "A", "to": "B", "cost": 3},
    {"from": "A", "to": "C", "cost": 1},
    {"from": "B", "to": "G", "cost": 4},
    {"from": "C", "to": "G", "cost": 6}
  ],
  "query": "Find the path from A to G with minimum total cost."
}
```

**Before submitting any variant, verify:**
- [ ] Answer is unchanged
- [ ] W1: wording is fully rewritten, not surface-level synonym swap
- [ ] W2: format is genuinely different, information is complete
- [ ] No LLM was used
- [ ] Another team member can independently compute the same answer

**Where variants go:** `data/problems/probe1_variants.csv`

| Column | What to put |
|--------|-------------|
| `problem_id` | Same as the canonical problem |
| `variant_id` | e.g. `SP_001_W1`, `SP_001_W2` |
| `variant_type` | `W1`, `W2`, etc. |
| `variant_text` | Full problem statement of the variant |
| `correct_answer` | Same as canonical — must match exactly |
| `written_by` | Your name |
| `reviewed_by` | Reviewer's name (fill after review) |

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
