# BM-00 — Start Here (read this first, top to bottom)

You are looking at a research vault: a set of linked notes describing a multi-paper research program about how language models actually solve problems. This manual assumes you know Python basics and nothing else. It tells you what the words mean, what order to do things in, and exactly how to run the first real experiments.

## The whole program in five sentences
Models score high on benchmarks, but a high score can mean "it understood" OR "it memorized something similar." Nobody can currently tell which, per question. This program builds an INSTRUMENT that tells them apart, then CALIBRATES it against cases where we know the truth, then finds LAWS for when models fail. The instrument uses three angles (does the answer survive cosmetic changes; does the model's plan match what it does; how much did it likely see in training) plus deeper measurement (read the wrong answers, look inside the network, fit a statistical model of strategy). The end goal is a measurement science for reasoning, not another benchmark.

## The map (what each folder is)
- **00_MOC** — the master index. Always your first click.
- **01_Program_State** — the single source of truth: what is proven, what is guessed, what is broken. Update it every session.
- **02_Decision_Memo** — which ideas to do first and why.
- **03_Evaluation_Framework** — what the instrument is (the three probes), in detail.
- **04_Directions (D01–D10)** — ten concrete research moves.
- **05_Papers (P01–P34)** — one note per related paper, fixed format.
- **06_Operating_Manual (HP-01–HP-14)** — copy-paste task prompts for a coding AI (Cursor). Each is self-contained.
- **07_Broader_Impact** — why industry cares.
- **08_Lab** — how to build and run the research "brain" (tools, costs, how to pick problems).
- **09_Deep_Strategies (DS-01–DS-12)** — the deeper, more novel/scientific moves.
- **10_Beginners_Manual** — you are here.
- **99_Templates** — blank forms for each note type.

## What to read, in order (about 2 hours)
1. This note, fully.
2. [[BM-01_Glossary_For_Beginners]] — every technical word, plain English.
3. [[00_MOC]] then [[01_Program_State]] sections A and B (the resolved facts).
4. [[02_Decision_Memo]] — just the "Ranked verdicts" section.
5. [[BM-02_How_To_Run_The_Lab]] — how the day-to-day actually works.
6. [[THE_PLAN]] — the master roadmap you will follow phase by phase; [[BM-08_Reading_Curriculum]] tells you what to read when.
Then stop reading and do [[BM-04_Your_First_Week]] (which is Phase 0, Week 1 of THE PLAN).

## The one rule that keeps this from becoming a mess
The vault is the memory. Every work session STARTS by loading the relevant notes and ENDS by writing back what changed (a dated line in [[01_Program_State]]). A coding AI produces the numbers; a reasoning session interprets them; both write back to the vault. If you skip the write-back, the knowledge is lost and you start cold next time. This single habit is what makes a "second brain" actually work.
