# BM-02 — How To Run The Lab (the daily loop)

Three roles. You will play all three at first; later you delegate two to AIs.

**1. The Scientist (you + a reasoning model like this chat).** Decides what question matters, interprets results, writes claims. NEVER produces numbers from memory. Output = decisions + HP prompts.
**2. The Engineer (Cursor / a coding AI + the repo).** Runs code, makes tables, plots, statistics. Produces ONLY verified numbers with the file they came from. Follows HP prompts exactly; on a failed check, reports instead of guessing.
**3. The Librarian (you, ~2 hrs/week + the ingestion script).** Keeps the vault current: files new papers, updates Program_State, prunes stale notes.

## The loop for one piece of work
1. OPEN: paste [[00_MOC]] + [[01_Program_State]] + the specific D-note or DS-note into your reasoning session.
2. DECIDE: pick the next task; if it needs numbers, find or write the matching HP prompt in 06_Operating_Manual.
3. EXECUTE: paste the HP prompt into Cursor with repo access. It returns artifacts (CSVs, figures, a report file) at stated paths.
4. CHECK: confirm the HP's "Validate" conditions passed. If not, do not proceed — fix or report.
5. INTERPRET: back in the reasoning session, run the reasoning checklist ([[BM-03_Thinking_Checklists]]) on the result.
6. WRITE BACK: add a dated line to [[01_Program_State]] changelog; update any affected notes; if priorities changed, update [[02_Decision_Memo]].

## The golden guardrails (from the program's own hard lessons)
- Raw data files are the truth. Derived tables must regenerate from them by script, deterministically.
- Any number from an outside document is QUARANTINED until reproduced from our raw data. (This caught a real error once — the A1 GSM discrepancy.)
- The reasoning model never invents statistics. If a claim needs a number, it asks the Engineer for it.
- Pre-register thresholds and decision rules before runs that could tempt you to move the goalposts.

## When to bring in a coding AI vs do it yourself
Do it yourself while learning the zero-cost analyses (HP-13, HP-14) — you learn the most from these. Delegate the GPU-heavy and large-sweep tasks (HP-06, HP-07, HP-09, HP-10) to Cursor once you trust the loop.
