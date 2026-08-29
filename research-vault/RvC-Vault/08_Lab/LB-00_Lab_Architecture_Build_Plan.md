# LB-00 — Lab Architecture and Build Plan (the "research brain")

Goal: a system where knowledge accumulates, reasoning sessions never start cold, execution never hallucinates numbers, and any future agent or collaborator can be productive in one read. Five layers. Everything below is buildable by one person; software cost ≈ $0, running cost ≈ a few dollars/month, setup ≈ one week part-time.

## Layer 1 — Storage (the memory)
Tool: **Obsidian** (free) on this vault, version-controlled in a **private GitHub repo** (free). The vault you have is the seed; folder and tag conventions are in [[00_MOC]].
Install these plugins (all free):
- **Dataview** — query notes like a database (e.g., list every paper note where `bears_on` contains D03; list every claim tagged abstract-only). This is what turns notes into a queryable knowledge base.
- **Templater** — binds the 99_Templates forms to hotkeys so every new note keeps the schema. Schema discipline is what makes Dataview queries work.
- **Zotero Integration** (+ **Zotero** app, free) — Zotero holds PDFs and BibTeX metadata; the plugin creates/links paper notes from Zotero entries; Better BibTeX auto-exports a .bib for LaTeX. One canonical bibliography for vault and papers.
- **Smart Connections** or a small local embedding index (see Layer 2) — "what in my vault relates to X" semantic search.
Sync: git push (free) or Obsidian Sync (~$4–5/mo) for phone access.
Rule: the vault is the single source of truth. Chat sessions, Cursor runs, and drafts all write back here or they didn't happen.

## Layer 2 — Ingestion (literature intelligence)
Weekly, ~1–2 hrs of your time + one script.
Sources (all free APIs): **arXiv API** (standing queries live in [[P-00_Ingestion_Queue]]), **Semantic Scholar API** (citation graph, TLDRs, "cited by" alerts on your anchor papers), **OpenReview API** (ICLR/NeurIPS reviews — reviewer weaknesses sections are the highest-value text per token anywhere).
Pipeline (~150 lines Python, runnable by cron or GitHub Action):
1. Pull new candidates matching standing queries + new citations of anchor papers (P17, P18, P20, P25...).
2. Dedupe against 05_Papers.
3. For each candidate: one Claude API call with the [[T_Paper_Note]] schema + abstract (+intro if PDF fetched) → draft note lands in `05_Papers/_inbox/` tagged `abstract-only`.
4. **Human gate:** you read the inbox weekly, promote/reject/edit. Nothing enters 05_Papers without your eyes. This single rule prevents the vault from rotting into AI slop.
Cost: ~20–40 papers/week × ~10K tokens ≈ well under $10/month at mid-tier model rates (verify current pricing).
Semantic index: embed all note bodies + abstracts (any cheap embedding API or a local model) into Chroma/FAISS (free, local). Rebuild weekly. Gives the "connect paper A's probe to paper C's failure mode" queries the pasted advice describes.

## Layer 3 — Reasoning (sessions like this one)
The "scientific reasoning engine" is not software; it is a session protocol plus checklists, enforced every time.
Session protocol: START by pasting [[00_MOC]] + [[01_Program_State]] + the relevant D-note into the model. END by writing a dated changelog entry to [[01_Program_State]] and updating any affected notes. A session that doesn't write back is a leak.
Reasoning checklist (run against every new result or claim): What's the alternative explanation? What hidden assumption? What would falsify this? Does anything in 05_Papers contradict it? Is the effect size worth the claim? Which reviewer attack in [[EF-06_Open_Methodological_Questions]] does it trigger? What human-sciences construct does it map to (see [[LB-02_Problem_Statement_Engine]])?
Division of labor (matches the advice doc, already encoded in [[OM-00_How_To_Use]]): reasoning sessions NEVER produce numbers; they produce HP-style requests. Numbers come only from Layer 4 with verified-raw tags.

## Layer 4 — Execution (Cursor + repo)
Cursor (or any coding agent) operates the retrieval-vs-computation repo using the HP prompts in 06_Operating_Manual. Contract: exact inputs, steps, output paths, validation checks; on check failure, report — never improvise. Every completed HP writes its report file into the repo AND a changelog line into [[01_Program_State]].
Data hygiene rules that already saved you once (the A1 audit): raw CSVs are truth; derived tables regenerate deterministically; external documents' numbers are quarantined until reproduced.

## Layer 5 — Writing (papers as views over the vault)
Each paper = a LaTeX repo + a claim-audit table: every quantitative sentence maps to (CSV path, filter, script). HP-02 is the template for this discipline. Reviewer pre-mortem = walk EF-06 against the draft. Paper-splitting decisions stay deferred until evidence exists (Decision Memo publication-cleavage note).

## Build order (part-time)
Day 1: Obsidian + this vault + git + Dataview/Templater. Day 2: Zotero + import current bibliography + link the 31 existing paper notes. Day 3–4: ingestion script v1 (arXiv + Semantic Scholar → inbox notes). Day 5: embedding index + Smart Connections. Week 2: first weekly sweep; wire OpenReview; add the session-protocol text as a saved prompt. Ongoing: 1–2 hrs/week inbox triage; monthly re-run of standing searches; quarterly prune of stale notes.

## Running costs (the brain itself)
Software $0. Ingestion API <$10/mo. Optional sync $4–5/mo. Your time ~2 hrs/wk. All experiment costs live in [[LB-03_Evaluation_Strategy_Catalog]].
