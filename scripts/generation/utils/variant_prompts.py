"""LLM prompt templates for W1/W2/W3/W5 variant generation (OpenRouter / chat APIs)."""

from __future__ import annotations

# ── W1: Universal paraphrase ──────────────────────────────────────────────

# Used for W1 variants across families. Asks the model to rephrase the prompt
# without changing entities, actions, numbers, or the underlying task.
W1_SYSTEM = """You rewrite planning and reasoning problems in different words.
Your ONLY job is to change sentence structure and phrasing.
You must preserve EXACTLY:
- Every entity name (block names, node numbers, variable names, person names)
- Every action name (pick-up, stack, put-down, unstack for planning problems)
- Every number
- Every spatial, logical, and mathematical relationship
- The goal or question being asked
Output ONLY the rewritten problem text. No explanation, no preamble."""

# User turn: wraps the canonical problem_text for a single paraphrase request.
W1_USER = "Rewrite this problem:\n\n{problem_text}"

# ── W2: GSM structured extraction ────────────────────────────────────────

# Used only for GSM W2 (BW/ALGO W2 are built deterministically). Asks the model
# to turn a word problem into a fixed "Given: / Find:" structured layout.
W2_GSM_SYSTEM = """You extract structured facts from math word problems.
Output ONLY in this exact format:
Given:
- [variable description]: [value]
- [variable description]: [value]
...
Find: [restate the question in one line]
No explanation. No other text."""

# User turn: supplies the GSM canonical problem for structured extraction.
W2_GSM_USER = "Extract the structured facts from this problem:\n\n{problem_text}"

# ── W3: BW/MBW entity and action rename ──────────────────────────────────

# Used for Blocksworld / Mystery Blocksworld W3. Asks the model to pick a domain
# theme and return JSON mappings for blocks and action verbs (not the rewrite itself).
W3_BW_MAPPING_SYSTEM = """You produce domain-rename mappings for planning problems.
Output ONLY a valid JSON object. No explanation. No markdown. No code fences.
Raw JSON only. The JSON must be parseable by json.loads() directly."""

# User turn: lists mapping rules and the BW problem_text; output is applied locally.
W3_BW_MAPPING_USER = """Create a rename mapping for this Blocksworld planning problem.
Choose ONE domain from: hr_org (employees/HR), library (books/library), military (units/commands).

Return ONLY this JSON (no other text):
{{
  "chosen_domain": "hr_org|library|military",
  "entity_mapping": {{"a": "NewName", "b": "NewName", ...}},
  "action_mapping": {{
    "pick-up": "new_action",
    "put-down": "new_action",
    "stack": "new_action",
    "unstack": "new_action"
  }}
}}

Rules:
- Map EVERY block letter that appears in the problem
- All entity_mapping values must be unique (no two blocks get same name)
- All action_mapping values must be unique
- Names should be coherent in the chosen domain (e.g. hr_org: Alice, Bob, Carol)

Problem:
{problem_text}"""

# ── W3: GSM entity rename ─────────────────────────────────────────────────

# Used for GSM W3. Asks the model for a bijective entity rename JSON (names/places only).
W3_GSM_MAPPING_SYSTEM = """You produce entity-rename mappings for math word problems.
Output ONLY a valid JSON object. No explanation. No markdown. No code fences."""

# User turn: GSM problem_text in; pipeline applies entity_mapping via variant_utils.
W3_GSM_MAPPING_USER = """This is a math word problem. Replace the entire domain/scenario 
context while keeping all numbers and mathematical relationships identical.

Replace EVERY domain-specific noun (people types, objects, activities, places) 
with equivalents from a completely different context.

Good domain replacements:
- Students with hobbies → Workers at a factory with job types
- Animals eating food → Vehicles using fuel types  
- People buying items → Machines processing material types

Rules:
- Replace every content noun that identifies what the scenario is about
- Keep ALL numbers exactly as they are
- Keep ALL mathematical relationship words (twice, half, percent, more than, etc.)
- The new domain must make logical sense
- Map EVERY replaced noun — if a noun appears multiple times, map it once

Return ONLY this JSON (no other text):
{{
  "chosen_domain": "brief description of new domain (e.g. factory workers with job types)",
  "entity_mapping": {{
    "original_noun": "replacement_noun",
    "original_activity": "replacement_activity"
  }}
}}

Problem:
{problem_text}"""

# ── W3: SP node rename ───────────────────────────────────────────────────

# Used for shortest-path (SP) W3. Asks the model to map integer node IDs to place names.
W3_SP_MAPPING_SYSTEM = """You produce node-rename mappings for graph problems.
Output ONLY a valid JSON object. No explanation. No markdown. No code fences."""

# User turn: SP problem_text; graph topology and weights stay unchanged.
W3_SP_MAPPING_USER = """This is a shortest-path problem with integer-labeled nodes.
Rename each node to a location name (city, airport, station) from one specific region.

Rules:
- Map every integer node that appears in the problem
- Use SHORT names (1-2 words max)
- All names must be unique
- Pick one coherent geographic region (e.g. all European cities)

Return ONLY this JSON (no other text):
{{
  "entity_mapping": {{"0": "CityName", "1": "CityName", ...}}
}}

Problem:
{problem_text}"""

# ── W3: WIS interval rename ───────────────────────────────────────────────

# Used for weighted interval scheduling (WIS) W3. Asks for task labels per interval.
W3_WIS_MAPPING_SYSTEM = """You produce task-rename mappings for scheduling problems.
Output ONLY a valid JSON object. No explanation. No markdown. No code fences."""

# User turn: WIS problem_text; numeric start/end/weight values are not renamed.
W3_WIS_MAPPING_USER = """This is a weighted interval scheduling problem.
Rename each interval to a named task in one coherent context 
(conference talks, construction jobs, TV shows — pick one).

Rules:
- Map each "Interval N" to a short task name (2-4 words)
- All task names must be unique
- Numeric values (start, end, weight) do NOT change

Return ONLY this JSON (no other text):
{{
  "entity_mapping": {{"Interval 0": "Task Name", "Interval 1": "Task Name", ...}},
  "context": "one sentence describing the scheduling scenario"
}}

Problem:
{problem_text}"""

# ── W3: CC context rename ─────────────────────────────────────────────────

# Used for coin change (CC) W3. Asks for reframing metadata (not denomination math).
W3_CC_MAPPING_SYSTEM = """You reframe coin change problems in a different context.
Output ONLY a valid JSON object. No explanation. No markdown. No code fences."""

# User turn: CC problem_text; denomination list and target amount stay the same.
W3_CC_MAPPING_USER = """This is a coin change problem. Reframe it in a different 
discrete-unit context (postage stamps, trading card packs, food portions).
The denomination values and target value do NOT change.

Return ONLY this JSON (no other text):
{{
  "chosen_context": "context name",
  "unit_name": "what one denomination unit is called",
  "target_description": "what the target value represents"
}}

Problem:
{problem_text}"""

# ── W5: GSM question reversal ─────────────────────────────────────────────

# Used for GSM W5 only. Asks the model to swap which quantity is unknown vs given.
W5_GSM_SYSTEM = """You reverse arithmetic word problems.
The original problem gives some quantities and asks for a final result.
You rewrite it so the original answer becomes a GIVEN VALUE,
and one of the original given values becomes the UNKNOWN.
The reversed problem must be uniquely solvable.
Output ONLY the reversed problem text. No explanation."""

# User turn: passes original text and correct_answer so the old answer is embedded as given.
W5_GSM_USER = """Reverse this problem so it asks for a different unknown.
The original answer is {correct_answer} — make this a given in your reversed version.
Ask for something that was originally given (not the same question in reverse wording).

Original problem:
{problem_text}

Output ONLY the reversed problem text."""

# ── Stage 1 GSM generation (low contamination) ─────────────────────────────

# Used when generating new low-contamination GSM canonical rows (not W1–W6 variants).
# Asks the model to author a novel multi-step word problem plus solution and answer.
GSM_LOW_CONTAM_GENERATION = """You are generating one original arithmetic word problem designed for low contamination.

Requirements:
- Create exactly one problem with uncommon, non-schoolbook context (examples: archival restoration labs, marine sensor maintenance, orbital greenhouse nutrient dosing, antique clock calibration logs, volcanic gas sampling trips).
- Use realistic but non-round values (avoid clean multiples like 10, 50, 100, 1000).
- Ensure the reasoning requires at least 4 arithmetic steps and is internally consistent.
- Do not copy known benchmark phrasings; wording must be novel.
- The final answer must be a single numeric value.

Output format (strict; include all three blocks in this order):
PROBLEM:
<problem statement only>

SOLUTION_STEPS:
<numbered steps showing arithmetic, with equations where appropriate>

ANSWER:
<final numeric answer only>
"""
