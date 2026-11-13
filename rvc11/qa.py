"""QA gates. A real run is blocked until every gate passes (``python -m rvc11 qa``)."""

from __future__ import annotations

import shutil
from collections import defaultdict

from . import analyze, bank as bank_mod, client, config, experiments, prompts, runner, scoring, verify


def _gate(results: list, name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def run_all(verbose: bool = True) -> bool:
    res: list = []
    items = bank_mod.load()
    by_id = {i["item_id"]: i for i in items}

    # 1. gold in, gold out
    bad = [i["item_id"] for i in items if not verify.score(verify.gold_response(i), i).correct_strict]
    _gate(res, "gold answer passes the verifier for every usable item", not bad, f"{len(items)} items, failures: {bad[:5]}")
    bad = [i["item_id"] for i in items if not verify.score("Some reasoning.\n" + i["gold_block"], i).correct]
    _gate(res, "gold answer without the marker passes lenient scoring", not bad, f"failures: {bad[:5]}")

    # 2. attacks on the verifier
    bad = [i["item_id"] for i in items if verify.score("FINAL ANSWER:\n" + client._wrong_block(i), i).correct]
    _gate(res, "a well-formed wrong answer is rejected for every item", not bad, f"accepted: {bad[:5]}")
    bad = [i["item_id"] for i in items if any(verify.score(t, i).correct for t in ("", "FINAL ANSWER:", "I cannot solve this."))]
    _gate(res, "empty and refusal responses are rejected", not bad, f"accepted: {bad[:5]}")
    by_sub = defaultdict(list)
    for i in items:
        by_sub[i["subtype"]].append(i)
    cross = tot = 0
    for sub, group in by_sub.items():
        for a, b in zip(group, group[7:] + group[:7]):
            if a["problem_id"] != b["problem_id"]:
                tot += 1
                cross += verify.score(verify.gold_response(b), a).correct
    _gate(res, "another problem's gold answer is rarely accepted", cross / max(tot, 1) < 0.03, f"{cross}/{tot}")

    # 3. prompts differ only as intended
    bad = []
    for ins in prompts.INSTRUCTIONS:
        seen = {}
        for i in items:
            sc = prompts.scaffold(i, ins)
            if seen.setdefault(i["subtype"], sc) != sc:
                bad.append(i["item_id"])
    _gate(res, "prompt scaffold is identical across all variants of a subtype", not bad, f"differs: {bad[:5]}")
    bad = [i["item_id"] for i in items for ins in ("show_work", "answer_only")
           if not prompts.build_prompt(i, ins).startswith(prompts.build_prompt(i, "none"))]
    _gate(res, "instruction levels differ only by the appended instruction sentence", not bad)
    bad = [i["item_id"] for i in items if bank_mod._RESIDUAL.search(i["task_text"]) or bank_mod._INSTRUCTION_START.search(i["task_text"])]
    _gate(res, "no answer-format or verbosity instruction is left inside any task text", not bad, f"left: {bad[:5]}")
    bad = [i["item_id"] for i in items if i["variant"] != "canonical" and i["task_text"] == by_id.get(i["problem_id"] + "::canonical", {}).get("task_text")]
    _gate(res, "no variant is a copy of its canonical", not bad, f"copies: {bad[:5]}")
    bad = [i["item_id"] for i in items if i["problem_id"] + "::canonical" not in by_id]
    _gate(res, "every usable variant has a usable canonical partner", not bad)

    # 4. offline end-to-end with scripted models
    models, study = config.load_models(), config.load_study()
    d = runner.run_dir("_qa_selftest")
    shutil.rmtree(d)
    d = runner.run_dir("_qa_selftest")
    mocks = ["mock_oracle", "mock_wrong", "mock_empty", "mock_unmarked", "mock_truncated", "mock_error"]
    plan = experiments.build_plan("p1_surface", study, models, items, mocks, limit=3)
    runner.write_manifest(d, models, study, "p1_surface")
    counts = runner.execute(plan, models, d, by_id, workers=8, progress_every=10**9)
    rows = scoring.score_run(d, by_id)
    acc = defaultdict(list)
    for r in rows:
        acc[r["model"]].append(r)
    mean = lambda m, k: sum(bool(r[k]) for r in acc[m] if r["usable"]) / max(1, sum(bool(r["usable"]) for r in acc[m]))
    _gate(res, "scripted always-right model scores 100%", mean("mock_oracle", "correct") == 1.0)
    _gate(res, "scripted always-wrong model scores 0%", mean("mock_wrong", "correct") == 0.0)
    _gate(res, "scripted empty model scores 0%", mean("mock_empty", "correct") == 0.0)
    _gate(res, "unmarked right answers: lenient 100%, strict 0%", mean("mock_unmarked", "correct") == 1.0 and mean("mock_unmarked", "correct_strict") == 0.0)
    _gate(res, "failed API calls are never scored", all(not r["usable"] and r["exclude_reason"] == "api_error" for r in acc["mock_error"]) and len(acc["mock_error"]) > 0)
    again = runner.execute(plan, models, d, by_id, workers=4, progress_every=10**9)
    _gate(res, "resume repeats only failed calls", again["ok"] == 0 and again["error"] == counts["error"], str(again))
    try:
        analyze.run(d)
        _gate(res, "analysis runs on the self-test output", True)
    except Exception as exc:  # noqa: BLE001
        _gate(res, "analysis runs on the self-test output", False, repr(exc))

    # 5. P2 parsing
    item = next(i for i in items if i["subtype"] == "gsm" and i["variant"] == "canonical")
    steps = experiments.p2_parse_steps("Step 1: cost of first part = 18\nStep 2: remaining minutes = 66\n**Step 3:** total = $1,051.50\nFINAL ANSWER: 51")
    ok = [s[2] for s in steps] == [18.0, 66.0, 1051.5]
    pr = experiments.p2_continue_prompt(item, steps, 2, "inject_false")
    _gate(res, "P2 step parsing and false-value injection", ok and "= 99" in pr and "= 66" not in pr and "= 18" in pr)
    shutil.rmtree(d)
    passed = all(ok for _, ok, _ in res)
    print(f"\n{sum(ok for _, ok, _ in res)}/{len(res)} gates passed" + ("" if passed else "  -> DO NOT RUN"))
    return passed
