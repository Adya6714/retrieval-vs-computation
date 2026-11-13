"""Build the normalised v1.1 bank from the legacy question banks.

For every legacy row this module
  1. strips every answer-format / verbosity instruction out of the problem text,
  2. attaches a formal spec that scoring will use (never the prose),
  3. checks the text against the spec and assigns a tier (verified / unverified),
  4. quarantines rows that cannot be scored soundly, with the reason.

Legacy prose parsers are used here, once, at build time, and their output is cross-checked
(PlanBench PDDL, simulation of gold plans, independent solvers). They are never used for scoring.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from . import algo, arith, planning
from .planning import CANON_VERBS, ARITY, MYSTERY_VERBS, Vocab

ROOT = Path(__file__).resolve().parents[1]
LEGACY = {
    "bw": ROOT / "data/problems/question_bank_bw.csv",
    "gsm": ROOT / "data/problems/question_bank_gsm.csv",
    "algo": ROOT / "data/problems/question_bank_algo.csv",
}
OUT_DIR = ROOT / "data/v11"
PLANBENCH = ROOT / "data/sources/planbench"

VARIANT_CLASS = {
    "canonical": "canonical", "W1": "paraphrase", "W2": "reformat", "W3": "cover_story",
    "W4": "formal_notation", "W5": "reversal", "W6": "regenerated_instance",
}

# First sentence that starts an answer-format or verbosity instruction. Everything from the
# earliest match to the end of the text is removed and logged in ``stripped_tail``.
_INSTRUCTION_START = re.compile(
    r"(?:\*\*)?(?:output\s+format\s*:|respond\s+(?:with|in\s+this|only|using)|reply\s+(?:with|as\b|using|only)"
    r"|provide\s+(?:only|your\s+response|a\s+numbered)|all\s+actions\s+must|return\s+(?:only|a\s+numbered)|report\s+only|give\s+(?:just\s+|only\s+)?the\s+final\s+answer"
    r"|give\s+only|answer\s+with\s+only|format\s*:|every\s+action\s+(?:should|must)\s+be|each\s+(?:action|operation)\s+must\s+"
    r"(?:be|precisely)|no\s+explanation|don'?t\s+add\s+explanat|include\s+no\s+explanation)",
    re.IGNORECASE,
)
_RESIDUAL = re.compile(
    r"no explanation|no (?:extra|other) text|nothing else|exact format|numbered (?:list|sequence)|"
    r"reply with|respond with|output format",
    re.IGNORECASE,
)


def strip_instructions(text: str) -> tuple[str, str]:
    t = str(text).strip()
    if len(t) > 1 and t[0] == '"' and t[-1] == '"':
        t = t[1:-1].strip()
    m = _INSTRUCTION_START.search(t)
    if not m:
        return _tidy(t), ""
    cut = m.start()
    # back up over an orphaned lead-in such as "**Question:** ... " that belongs to the task
    head, tail = t[:cut], t[cut:]
    return _tidy(head), tail.strip()


def _tidy(t: str) -> str:
    t = re.sub(r"[ \t]+\n", "\n", t).strip()
    t = re.sub(r"(?:\n\s*(?:---+|\*\*\*+)\s*)+$", "", t).strip()
    t = re.sub(r"^(?:---+\s*\n)+", "", t).strip()
    t = re.sub(r"\n\s*(?:CONSTRAINTS|OUTPUT FORMAT|TASK)\s*:\s*-?\s*$", "", t).strip()
    return t.rstrip("`*_ \n")


def _read(path: Path) -> list[dict]:
    csv.field_size_limit(10**9)
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _facts_json(facts) -> list[list[str]]:
    return sorted([list(f) for f in facts])


# ------------------------------------------------------------------ planning
def _legacy_parse(text: str, mystery: bool):
    """(init, goal, handempty_stated) in canonical BW facts, or None. Build-time only."""
    from probes.contamination import verify as L  # legacy prose parsers

    if mystery:
        parsed = L._parse_mystery_state(text)
        if parsed and L._mystery_state_usable(parsed):
            st, gl = planning.mystery_to_bw(parsed[0]), planning.mystery_to_bw(parsed[1])
            stated = bool(re.search(r"harmony|goodwill", text.split("oal")[0], re.IGNORECASE))
            return st, gl, stated
    parsed = L._parse_blocksworld_state(text)
    if parsed and L._bw_state_usable(parsed):
        cur = L._extract_current_and_goal(text)
        stated = bool(cur and re.search(r"hand|gripper|handempty", cur[0], re.IGNORECASE))
        return parsed[0], parsed[1], stated
    return None


def _normalise_init(state) -> set:
    on = {(f[1], f[2]) for f in state if f[0] == "on"}
    blocks = {f[1] for f in state if f[0] in ("ontable", "clear")}
    return planning.close_state(on, blocks)


def _normalise_goal(goal) -> set:
    return {f for f in goal if f[0] in ("on", "ontable", "clear")}


def _pddl_spec(source: str):
    m = re.search(r"(plan-bench/instances/\S+\.pddl)", source)
    if not m:
        return None
    rel = m.group(1)
    cands = [PLANBENCH / rel, PLANBENCH / rel.replace("blocksworld/mystery/", "mystery_blocksworld/")]
    path = next((p for p in cands if p.exists()), None)
    if path is None:
        return None
    t = path.read_text().lower()
    grab = lambda s: {tuple(x.split()) for x in re.findall(r"\(([a-z][a-z0-9 _-]*)\)", s)}
    init = grab(t[t.index("(:init"): t.index("(:goal")]) - {("init",)}
    goal = grab(t[t.index("(:goal"):]) - {("and",)}
    return planning.mystery_to_bw(init), planning.mystery_to_bw(goal)


def _base_vocab(mystery: bool) -> dict:
    return {v: (MYSTERY_VERBS[v] if mystery else v) + (" X Y" if ARITY[v] == 2 else " X") for v in CANON_VERBS}


def _schema_templates(tail_and_text: str):
    m = re.search(r"(?:exactly one of|precisely(?: be one of these)?)\s*:\s*([^.\n]+)", tail_and_text, re.IGNORECASE)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split("/")]
    if len(parts) != 4 or not all(re.search(r"\bX\b", p) for p in parts):
        return None
    return dict(zip(CANON_VERBS, parts))


def _notes_json(notes: str) -> dict:
    i = str(notes).find("{")
    if i < 0:
        return {}
    try:
        return json.JSONDecoder().raw_decode(str(notes)[i:])[0]
    except json.JSONDecodeError:
        return {}


def _surface_name(text: str, name: str) -> str:
    m = re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE)
    return m.group(0) if m else name


def _parse_legacy_plan(gold: str, vocab: Vocab):
    plan, status, _ = planning.parse_plan(gold, vocab)
    return plan if status == "parsed" else None


def _planning_items(rows: list[dict]) -> list[dict]:
    by_pid: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_pid.setdefault(r["problem_id"].strip(), {})[r["variant_type"].strip()] = r
    items = []
    for pid, variants in by_pid.items():
        canon = variants.get("canonical")
        mystery = (canon or next(iter(variants.values())))["problem_subtype"].strip().startswith("mystery")
        canon_spec, canon_src = None, ""
        if canon:
            parsed = _legacy_parse(canon["problem_text"], mystery)
            pddl = _pddl_spec(canon["source"])
            if pddl and parsed:
                agree = _normalise_init(pddl[0]) == _normalise_init(parsed[0]) and _normalise_goal(pddl[1]) == _normalise_goal(parsed[1])
                canon_spec, canon_src = (pddl[0], pddl[1]), "pddl+text_agree" if agree else "pddl_text_DISAGREE"
            elif pddl:
                canon_spec, canon_src = (pddl[0], pddl[1]), "pddl_only"
            elif parsed:
                canon_spec, canon_src = (parsed[0], parsed[1]), "text_only"
        for vt, r in sorted(variants.items()):
            text = r["problem_text"]
            task_text, tail = strip_instructions(text)
            flags, status, reason, tier = [], "ok", "", "unverified"
            vocab_t, entities = _base_vocab(mystery), {}
            own = _legacy_parse(text, mystery)
            spec = None
            if vt in ("canonical", "W1", "W2", "W4"):
                spec, src = canon_spec, canon_src if vt == "canonical" else "inherited_from_canonical"
                if spec is None:
                    status, reason = "quarantined", "no_canonical_spec"
                elif vt == "canonical":
                    tier = "verified" if canon_src in ("pddl+text_agree",) else "unverified"
                    if canon_src == "pddl_text_DISAGREE":
                        status, reason = "quarantined", "pddl_and_text_disagree"
                elif own and _normalise_init(own[0]) == _normalise_init(spec[0]) and _normalise_goal(own[1]) == _normalise_goal(spec[1]):
                    tier = "verified"
                else:
                    flags.append("rewrite_not_machine_checked")
            elif vt == "W3":
                src = "inherited_from_canonical+rename"
                nj = _notes_json(r["notes"])
                sch = _schema_templates(text)
                if nj.get("action_mapping"):
                    am = {str(k).lower(): str(v) for k, v in nj["action_mapping"].items()}
                    inv = {v: k for k, v in MYSTERY_VERBS.items()}
                    for k, v in am.items():
                        ck = inv.get(k, k)
                        if ck in ARITY:
                            vocab_t[ck] = v + (" X Y" if ARITY[ck] == 2 else " X")
                    if sch and [t.split()[0].lower() for t in sch.values()] == [vocab_t[v].split()[0].lower() for v in CANON_VERBS]:
                        vocab_t = sch
                elif sch:
                    vocab_t = sch
                else:
                    status, reason = "quarantined", "w3_action_vocabulary_unresolved"
                if canon_spec is None:
                    status, reason = "quarantined", "no_canonical_spec"
                else:
                    spec = canon_spec
                    objs = {x for f in list(spec[0]) + list(spec[1]) for x in f[1:]}
                    if nj.get("entity_mapping"):
                        entities = {str(k).lower(): str(v) for k, v in nj["entity_mapping"].items()}
                    elif own:
                        names = {x for f in list(own[0]) + list(own[1]) for x in f[1:]}
                        for o in objs:
                            hit = [n for n in names if n[0] == o[0]]
                            if len(hit) == 1:
                                entities[o] = _surface_name(text, hit[0])
                    if own and (set(entities) != objs or len(set(v.lower() for v in entities.values())) != len(objs)):
                        # fallback: pair objects and names by order of first mention; accepted only
                        # if the renamed facts then equal the canonical spec (checked just below)
                        ctext = canon["problem_text"].lower()
                        o_ord = sorted(objs, key=lambda o: (m.start() if (m := re.search(rf"\b{re.escape(o)}\b", ctext[ctext.find("current"):])) else 10**9))
                        n_ord = sorted(names, key=lambda n: (m.start() if (m := re.search(rf"\b{re.escape(n)}\b", text.lower()[text.lower().find("current"):])) else 10**9))
                        entities = {o: _surface_name(text, n) for o, n in zip(o_ord, n_ord)} if len(o_ord) == len(n_ord) else {}
                        inferred_by_order = True
                    else:
                        inferred_by_order = False
                    if set(entities) != objs or len(set(v.lower() for v in entities.values())) != len(objs):
                        status, reason = "quarantined", "w3_entity_mapping_unresolved"
                    elif own:
                        back = {v.lower(): k for k, v in entities.items()}
                        ren = lambda fs: {(f[0],) + tuple(back.get(x, x) for x in f[1:]) for f in fs}
                        if _normalise_init(ren(own[0])) == _normalise_init(spec[0]) and _normalise_goal(ren(own[1])) == _normalise_goal(spec[1]):
                            tier = "verified"
                        elif inferred_by_order:
                            status, reason = "quarantined", "w3_entity_mapping_unresolved"
                        else:
                            flags.append("rewrite_not_machine_checked")
                    else:
                        flags.append("rewrite_not_machine_checked")
                kind = "cover_story" if not re.search(r"robot arm|\bblocks?\b", task_text, re.IGNORECASE) else "rename_only"
                flags.append(f"w3_kind:{kind}")
            else:  # W5, W6: their own instance
                src = "parsed_from_own_text"
                if own is None:
                    status, reason = "quarantined", "own_instance_text_not_parseable"
                else:
                    spec = (own[0], own[1])
                    if not own[2]:
                        flags.append("init_hand_state_not_stated")
                    if canon and " ".join(text.split()).strip('"') == " ".join(canon["problem_text"].split()).strip('"'):
                        status, reason = "quarantined", f"{vt.lower()}_text_identical_to_canonical"
                    tier = "verified"
                    if vt == "W5" and not any(f[0] == "on" for f in own[1]):
                        flags.append("w5_goal_is_unstack_everything")
                    if vt == "W5" and canon_spec and _normalise_init(own[0]) != planning.close_state(
                        {(f[1], f[2]) for f in canon_spec[1] if f[0] == "on"},
                        {x for f in canon_spec[0] for x in f[1:]},
                    ):
                        flags.append("w5_not_a_reversal_of_canonical")
            item_spec, gold_block = None, ""
            if spec is not None and status == "ok":
                init, goal = _normalise_init(spec[0]), _normalise_goal(spec[1])
                phys, why = planning.state_is_physical(init)
                vocab = Vocab(vocab_t, entities)
                if not phys:
                    status, reason = "quarantined", f"init_not_physical:{why}"
                elif not goal:
                    status, reason = "quarantined", "empty_goal"
                else:
                    try:
                        ref = planning.reference_plan(init, goal)
                    except ValueError as exc:
                        status, reason, ref = "quarantined", f"unsolvable:{exc}", None
                    if ref is not None:
                        legacy_plan = _parse_legacy_plan(r["correct_answer"], vocab)
                        if legacy_plan and planning.simulate(init, goal, legacy_plan)[0]:
                            gold_plan = legacy_plan
                        else:
                            gold_plan = ref
                            flags.append("legacy_gold_invalid_replaced_by_reference_plan")
                        gold_block = vocab.render_plan(gold_plan)
                        item_spec = {
                            "kind": "plan", "init": _facts_json(init), "goal": _facts_json(goal),
                            "templates": vocab_t, "entities": entities, "spec_source": src,
                            "n_blocks": len({x for f in init for x in f[1:]}), "gold_len": len(gold_plan),
                            "ref_len": len(ref),
                        }
            if status != "ok":
                tier = "quarantined"
            items.append(_item(r, pid, vt, "planning", "mystery_blocksworld" if mystery else "blocksworld",
                               task_text, tail, item_spec, gold_block, status, reason, tier, flags))
    return items


# ------------------------------------------------------------------ algorithmic
_LABEL = r"(?:[A-Z][A-Za-z]*\s+)*[A-Z][A-Za-z]*\s+[A-Za-z0-9]+"


def _node_id(label: str):
    tok = label.strip().split()[-1]
    if re.fullmatch(r"\d+", tok):
        return int(tok)
    if re.fullmatch(r"[A-Z]", tok):
        return ord(tok) - 65
    return None


def _sp_text_edges(text: str):
    """Edges as written in the prompt: list of (label_u, label_v, weight). Handles every bank format."""
    t = text.replace("→", "->").replace("**", "")
    lab = rf"(?:{_LABEL}|\d+)"
    pats = [
        rf"({lab})\s*(?:\bto\b|\band\b|->)\s*({lab})\s*:\s*(-?\d+)",
        rf"\(\s*({lab})\s*(?:->|,)\s*({lab})\s*,\s*(?:weight\s*=\s*)?(-?\d+)\s*\)",
        rf"\|\s*({lab})\s*\|\s*({lab})\s*\|\s*(-?\d+)(?:\s*[A-Za-z]+)?\s*\|",
    ]
    best: list = []
    for pat in pats:
        found = [(m.group(1).strip(), m.group(2).strip(), int(m.group(3))) for m in re.finditer(pat, t)]
        if len(found) > len(best):
            best = found
    return best


def _pairs_in(text: str) -> set:
    """Unordered id pairs written as a-b, {a,b}, (a,b) or two-integer table rows."""
    t = text.replace("–", "-")
    out = set()
    for pat in (r"(?<![\w.])(\d+)\s*-\s*(\d+)(?![\w.])", r"\{\s*(\d+)\s*,\s*(\d+)\s*\}", r"\(\s*(\d+)\s*,\s*(\d+)\s*\)",
                r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|"):
        for m in re.finditer(pat, t):
            a, c = int(m.group(1)), int(m.group(2))
            if a != c:
                out.add((min(a, c), max(a, c)))
    return out


def _algo_items(rows: list[dict]) -> list[dict]:
    items = []
    canon_text = {r["problem_id"].strip(): " ".join(r["problem_text"].split()) for r in rows if r["variant_type"].strip() == "canonical"}
    canon_rows = {r["problem_id"].strip(): r for r in rows if r["variant_type"].strip() == "canonical"}
    for r in rows:
        pid, vt = r["problem_id"].strip(), r["variant_type"].strip()
        sub = r["problem_subtype"].strip().lower()
        task_text, tail = strip_instructions(r["problem_text"])
        flags, status, reason, tier, spec, gold_block = [], "ok", "", "unverified", None, ""
        try:
            p = json.loads(r["difficulty_params"])
        except json.JSONDecodeError:
            p, status, reason = {}, "quarantined", "difficulty_params_not_json"
        nums = arith.numbers_in(task_text)
        if status == "ok" and vt != "canonical" and " ".join(r["problem_text"].split()) == canon_text.get(pid):
            status, reason = "quarantined", f"{vt.lower()}_text_identical_to_canonical"
        if status == "ok" and sub == "shortest_path":
            graph = p["graph"] if isinstance(p["graph"], list) else json.loads(p["graph"])
            edges = [[int(e["u"]), int(e["v"]), int(e["w"])] for e in graph]
            labels: dict[int, set] = {}
            text_edges = _sp_text_edges(task_text)
            mapped = []
            for lu, lv, w in text_edges:
                u, v = _node_id(lu) if not lu.isdigit() else int(lu), _node_id(lv) if not lv.isdigit() else int(lv)
                mapped.append((u, v, w))
                for idx, lab in ((u, lu), (v, lv)):
                    if idx is not None:
                        labels.setdefault(idx, set()).add(lab)
            if any(u is None or v is None for u, v, _ in mapped) and len(text_edges) == len(edges) and [w for *_, w in text_edges] == [e[2] for e in edges]:
                # named nodes (cities, people): align prompt edges to spec edges by position
                fwd: dict[str, int] = {}
                consistent = True
                for (lu, lv, _w), (u, v, _w2) in zip(text_edges, edges):
                    for lab_, idx in ((lu, u), (lv, v)):
                        consistent &= fwd.setdefault(lab_, idx) == idx
                if consistent and len(set(fwd.values())) == len(fwd):
                    mapped = [tuple(e) for e in edges]
                    labels = {}
                    for lab_, idx in fwd.items():
                        labels.setdefault(idx, set()).add(lab_)
                        labels[idx].add(re.sub(r"^Node\s+", "", lab_))
            directed = bool(p.get("directed", True))
            canon_e = lambda es: sorted(((u, v, w) if directed or str(u) <= str(v) else (v, u, w) for u, v, w in es), key=str)
            if canon_e(mapped) == canon_e(tuple(e) for e in edges):
                tier = "verified"
            elif not all(any(abs(abs(w) - n) < 1e-9 for n in nums) for _, _, w in edges):
                status, reason = "quarantined", "text_does_not_contain_spec_edge_weights"
            else:
                flags.append("graph_in_text_not_machine_checked")
            node_ids = {int(p["source"]), int(p["target"])} | {x for e in edges for x in e[:2]}
            for mp in (p.get("node_mapping") or {}).items():
                if str(mp[1]).lower() in task_text.lower():
                    labels.setdefault(int(mp[0]), set()).add(str(mp[1]))
            lab_out = {str(i): sorted(labels.get(i, set()) | ({str(i)} if not labels.get(i) or any(ch.isdigit() for l in labels[i] for ch in l) else set())) for i in sorted(node_ids)}
            spec = {"kind": "sp", "directed": bool(p.get("directed", True)), "source": int(p["source"]),
                    "target": int(p["target"]), "edges": edges, "labels": lab_out}
            try:
                algo.sp_optimum(spec)
            except ValueError as exc:
                status, reason = "quarantined", f"unsolvable:{exc}"
        elif status == "ok" and sub == "wis":
            iv = p["intervals"] if isinstance(p["intervals"], list) else json.loads(p["intervals"])
            iv = sorted(iv, key=lambda d: int(d["id"]))
            ids = [int(d["id"]) for d in iv]
            overlap = [[a["id"], b["id"]] for i, a in enumerate(iv) for b in iv[i + 1:]
                       if not (int(a["end"]) <= int(b["start"]) or int(b["end"]) <= int(a["start"]))]
            interval_style = bool(re.search(r"start\s*=", canon_rows[pid]["problem_text"])) if pid in canon_rows else bool(overlap)
            canon_row = canon_rows.get(pid)
            canon_task = strip_instructions(canon_row["problem_text"])[0] if canon_row else ""
            valid = lambda ps: {q for q in ps if q[0] in ids and q[1] in ids}
            # drop "0-9"-style range mentions ("plots numbered 0-9") from the pair list
            stated = lambda txt: valid(_pairs_in(re.sub(r"\(\s*\d+\s*[-–]\s*\d+\s*\)|\bnumbered\s+\d+\s*(?:-|to|through)\s*\d+", " ", txt)))
            if interval_style:
                conflicts = overlap
            else:
                conflicts = [list(q) for q in sorted(stated(canon_task))]
            weights = {str(int(d["id"])): int(d["weight"]) for d in iv}
            seq = []
            for d in iv:
                seq += ([float(d["start"]), float(d["end"])] if interval_style else []) + [float(d["weight"])]
            it = iter(nums)
            in_order = all(any(abs(x - y) < 1e-9 for y in it) for x in seq)
            m_gold = re.search(r"Total\s*:\s*(\d+)", canon_row["correct_answer"]) if canon_row else None
            if not in_order:
                status, reason = "quarantined", "text_does_not_contain_spec_values_in_order"
            elif not interval_style and not conflicts:
                status, reason = "quarantined", "conflict_pairs_not_found_in_canonical_text"
            elif interval_style or vt == "canonical" or {tuple(c) for c in conflicts} == stated(task_text):
                tier = "verified"
            else:
                flags.append("conflict_structure_in_text_not_machine_checked")
            labels = {i: {str(i)} for i in ids}
            named = re.findall(r"^\s*(?:[-*]\s*)?([A-Z][^:\n|]{1,40}?):\s*start\s*=", task_text, flags=re.MULTILINE)
            if interval_style and len(named) == len(ids):
                for i, nm in zip(ids, named):
                    nm = nm.strip()
                    labels[i] = {nm} if not re.search(r"\d", nm) else {nm, str(i)}
            elif interval_style and status == "ok":
                flags.append("item_names_not_recovered")
            spec = {"kind": "wis", "weights": weights, "conflicts": conflicts,
                    "labels": {str(i): sorted(labels[i]) for i in ids},
                    "structure": "intervals" if interval_style else "conflict_list"}
            if status == "ok" and m_gold and abs(algo.wis_optimum(spec) - int(m_gold.group(1))) > 1e-9:
                # two independent sources (planner-era gold, our solver on the parsed structure) disagree
                status, reason = "quarantined", "solver_optimum_differs_from_legacy_gold"
        elif status == "ok" and sub == "coin_change":
            spec = {"kind": "cc", "denominations": [int(d) for d in p["denominations"]], "target": int(p["target"])}
            need = [float(spec["target"])] + [float(d) for d in spec["denominations"]]
            if all(any(abs(x - n) < 1e-9 for n in nums) for x in need):
                tier = "verified"
            else:
                status, reason = "quarantined", "text_does_not_contain_spec_values"
            try:
                best = algo.cc_optimum(spec)
                m = re.search(r"(?:Count|Total)\s*:\s*(\d+)", r["correct_answer"])
                if m and int(m.group(1)) != best:
                    flags.append("legacy_gold_count_differs_from_solver")
            except ValueError as exc:
                status, reason = "quarantined", f"unsolvable:{exc}"
        elif status == "ok":
            status, reason = "quarantined", f"unknown_subtype:{sub}"
        if status == "ok":
            gold_block = algo.render_gold(spec)
        else:
            tier = "quarantined"
        if p.get("instance_type"):
            flags.append(f"instance_type:{p['instance_type']}")
        items.append(_item(r, pid, vt, "algorithmic", sub, task_text, tail, spec if status == "ok" else None,
                           gold_block, status, reason, tier, flags))
    return items


# ------------------------------------------------------------------ arithmetic
def _gsm_items(rows: list[dict]) -> list[dict]:
    items = []
    canon_nums = {}
    for r in rows:
        if r["variant_type"].strip() == "canonical":
            canon_nums[r["problem_id"].strip()] = Counter(round(n, 6) for n in arith.numbers_in(strip_instructions(r["problem_text"])[0]))
    for r in rows:
        pid, vt = r["problem_id"].strip(), r["variant_type"].strip()
        task_text, tail = strip_instructions(r["problem_text"])
        flags, status, reason, tier, spec = [], "ok", "", "unverified", None
        try:
            gold = float(str(r["correct_answer"]).replace(",", "").strip())
            spec = {"kind": "numeric", "answer": gold}
        except ValueError:
            status, reason = "quarantined", "gold_not_numeric"
        if status == "ok":
            here = Counter(round(n, 6) for n in arith.numbers_in(task_text))
            base = canon_nums.get(pid, Counter())
            new = sorted(set(here) - set(base))
            if vt == "canonical":
                tier = "verified"
            elif vt in ("W1", "W3"):
                if set(here) == set(base):
                    tier = "verified"
                else:
                    flags.append("numbers_differ_from_canonical")
            elif vt in ("W2", "W4"):
                # A reformat must not add quantities. New numbers are derived intermediate values.
                if round(gold, 6) in new:
                    status, reason = "quarantined", "variant_text_states_the_answer"
                elif new:
                    flags.append("adds_derived_quantities:" + ",".join(f"{x:g}" for x in new[:6]))
                elif set(here) == set(base):
                    tier = "verified"
                if re.search(r"rounded up|rounded down|remaining \d|= sum of|sum of both", task_text, re.IGNORECASE):
                    flags.append("states_solution_step_in_words")
            elif vt == "W5":
                flags.append("different_question_inverse_problem")
            elif vt == "W6":
                flags.append("regenerated_numbers_gold_not_rederived_here")
        if status != "ok":
            tier = "quarantined"
        gold_block = f"{spec['answer']:g}" if spec else ""
        items.append(_item(r, pid, vt, "arithmetic", "gsm", task_text, tail, spec if status == "ok" else None,
                           gold_block, status, reason, tier, flags))
    return items


def _item(r, pid, vt, family, subtype, task_text, tail, spec, gold_block, status, reason, tier, flags) -> dict:
    if status == "ok" and _RESIDUAL.search(task_text):
        status, reason, tier, spec = "quarantined", "instruction_text_survived_stripping", "quarantined", None
    if status == "ok" and len(task_text) < 40:
        status, reason, tier, spec = "quarantined", "task_text_too_short_after_stripping", "quarantined", None
    return {
        "item_id": f"{pid}::{vt}", "problem_id": pid, "variant": vt, "variant_class": VARIANT_CLASS.get(vt, vt),
        "family": family, "subtype": subtype, "task_text": task_text, "stripped_tail": tail,
        "spec": spec, "gold_block": gold_block, "status": status, "quarantine_reason": reason,
        "tier": tier, "flags": flags,
        "meta": {"contamination_pole": r.get("contamination_pole", "").strip(), "difficulty": r.get("difficulty", "").strip(),
                 "source": r.get("source", "").strip()[:200], "legacy_gold": r.get("correct_answer", "")},
        "text_sha": hashlib.sha256(task_text.encode()).hexdigest()[:16],
    }


def build() -> dict:
    items = _planning_items(_read(LEGACY["bw"])) + _gsm_items(_read(LEGACY["gsm"])) + _algo_items(_read(LEGACY["algo"]))
    # a variant is only usable when its canonical partner is usable (paired analysis)
    ok_canon = {i["problem_id"] for i in items if i["variant"] == "canonical" and i["status"] == "ok"}
    for i in items:
        if i["status"] == "ok" and i["problem_id"] not in ok_canon:
            i.update(status="quarantined", quarantine_reason="canonical_partner_quarantined", tier="quarantined", spec=None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "bank.jsonl").open("w", encoding="utf-8") as fh:
        for i in items:
            fh.write(json.dumps(i, ensure_ascii=False, sort_keys=True) + "\n")
    with (OUT_DIR / "quarantine.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "subtype", "variant", "reason"])
        for i in items:
            if i["status"] != "ok":
                w.writerow([i["item_id"], i["subtype"], i["variant"], i["quarantine_reason"]])
    with (OUT_DIR / "stripped_instructions.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "removed_text"])
        for i in items:
            if i["stripped_tail"]:
                w.writerow([i["item_id"], " ".join(i["stripped_tail"].split())])
    digest = hashlib.sha256((OUT_DIR / "bank.jsonl").read_bytes()).hexdigest()
    summary = {
        "n_items": len(items), "n_ok": sum(i["status"] == "ok" for i in items),
        "bank_sha256": digest,
        "by_subtype_variant_tier": {f"{k[0]}|{k[1]}|{k[2]}": n for k, n in sorted(Counter((i["subtype"], i["variant"], i["tier"]) for i in items).items())},
        "quarantine_reasons": dict(Counter(re.sub(r":.*", "", i["quarantine_reason"]) for i in items if i["status"] != "ok").most_common()),
        "flags": dict(Counter(re.sub(r":.*", "", f) for i in items if i["status"] == "ok" for f in i["flags"]).most_common()),
    }
    (OUT_DIR / "bank_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load(include_quarantined: bool = False) -> list[dict]:
    path = OUT_DIR / "bank.jsonl"
    if not path.exists():
        raise FileNotFoundError("data/v11/bank.jsonl missing. Run: python -m rvc11 build-bank")
    items = [json.loads(line) for line in path.open(encoding="utf-8")]
    return items if include_quarantined else [i for i in items if i["status"] == "ok"]
