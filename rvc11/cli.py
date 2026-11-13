"""Command line. ``python -m rvc11 <command>``; see V11_RUNBOOK.md."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def _models_arg(models: dict, value: str) -> list[str]:
    if value == "all":
        return [k for k, m in models.items() if m.get("role") != "mock"]
    keys = [k.strip() for k in value.split(",") if k.strip()]
    missing = [k for k in keys if k not in models]
    if missing:
        sys.exit(f"unknown model key(s) {missing}. Known: {', '.join(models)}")
    return keys


def _guard_real(models: dict, keys: list[str], allow_unverified: bool) -> None:
    from . import config

    real = [k for k in keys if models[k].get("role") != "mock"]
    if not real:
        return
    locked = config.locked_ids()
    bad = [k for k in real if models[k]["id"] not in locked]
    if bad and not allow_unverified:
        sys.exit(f"model id not verified for: {bad}. Run `python -m rvc11 models-check` first "
                 "(or pass --allow-unverified for a non-OpenRouter endpoint).")


def cmd_build_bank(_a) -> None:
    from . import bank

    s = bank.build()
    print(f"items: {s['n_items']}  usable: {s['n_ok']}  bank sha256: {s['bank_sha256'][:16]}")
    print("quarantined:", json.dumps(s["quarantine_reasons"], indent=1))
    print("wrote data/v11/bank.jsonl, quarantine.csv, stripped_instructions.csv, bank_summary.json")


def cmd_qa(_a) -> None:
    from . import qa

    sys.exit(0 if qa.run_all() else 1)


def cmd_models_check(a) -> None:
    import os

    import requests

    from . import config

    _env()
    models = config.load_models()
    verified, report = [], []
    catalogues: dict = {}
    for key, m in models.items():
        if m.get("role") == "mock":
            continue
        base = m["base_url"].rstrip("/")
        if base not in catalogues:
            headers = {"Authorization": f"Bearer {os.environ.get(m.get('api_key_env') or '', '')}"} if m.get("api_key_env") else {}
            try:
                data = requests.get(f"{base}/models", headers=headers, timeout=60).json().get("data", [])
                catalogues[base] = {d["id"]: d for d in data}
            except Exception as exc:  # noqa: BLE001
                print(f"cannot read {base}/models: {exc}")
                catalogues[base] = {}
        cat = catalogues[base]
        if m["id"] in cat:
            sp = cat[m["id"]].get("supported_parameters") or []
            verified.append(m["id"])
            note = "" if not sp or "reasoning" in sp else "  WARNING: catalogue does not list 'reasoning'; set thinking.off: null or adjust"
            report.append(f"  OK       {key:14s} {m['id']}{note}")
        else:
            fam = m["id"].split("/")[0]
            near = sorted((i for i in cat if i.startswith(fam + "/")), key=lambda i: -(cat[i].get("created") or 0))[:6]
            report.append(f"  MISSING  {key:14s} {m['id']}\n           newest ids from '{fam}': {near or 'none found'}")
    print("\n".join(report))
    config.LOCK.write_text(json.dumps({"verified_ids": verified}, indent=2))
    print(f"\n{len(verified)} id(s) verified -> {config.LOCK.relative_to(ROOT)}. Edit configs/v11/models.yaml for the rest and re-run.")


def _plan(a):
    from . import bank, config, experiments

    models, study = config.load_models(), config.load_study()
    keys = _models_arg(models, a.models)
    items = bank.load()
    return models, study, keys, items, experiments.build_plan(a.experiment, study, models, items, keys, limit=a.limit)


def cmd_plan(a) -> None:
    models, _study, keys, _items, plan = _plan(a)
    by = Counter((c["model"], c["thinking"], c["instruction"]) for c in plan)
    for k, n in sorted(by.items()):
        print(f"  {k[0]:14s} thinking={k[1]:3s} instruction={k[2]:11s} calls={n}")
    chars = sum(len(c["prompt"]) for c in plan)
    print(f"total calls: {len(plan)}   prompt tokens (approx): {chars // 4:,}")
    print("Output tokens depend on the model and on thinking. Run with --limit 2 first and read "
          "median_completion_tokens in analysis/exclusions.csv to cost the full run.")


def cmd_run(a) -> None:
    from . import analyze, qa, runner, scoring

    _env()
    models, study, keys, items, plan = _plan(a)
    _guard_real(models, keys, a.allow_unverified)
    real = any(models[k].get("role") != "mock" for k in keys)
    if real and not a.skip_qa and not qa.run_all():
        sys.exit("QA gates failed. Fix them before spending API budget.")
    d = runner.run_dir(a.run)
    by_id = {i["item_id"]: i for i in items}
    runner.write_manifest(d, {k: models[k] for k in keys}, study, a.experiment)
    print(f"{a.experiment}: {len(plan)} calls planned -> {d.relative_to(ROOT)}")
    counts = runner.execute(plan, models, d, by_id, workers=a.workers, max_calls=a.max_calls)
    print(counts)
    scoring.score_run(d, by_id)
    print("report:", analyze.run(d).relative_to(ROOT))
    if counts["error"]:
        print(f"{counts['error']} call(s) failed. Re-run the same command to retry only those.")


def cmd_p2_continue(a) -> None:
    from . import analyze, bank, config, experiments, runner, scoring

    _env()
    models = config.load_models()
    items = bank.load()
    by_id = {i["item_id"]: i for i in items}
    d = runner.run_dir(a.run)
    scored = {r["call_id"]: r for r in scoring.score_run(d, by_id)}
    plan_calls = [c for c in runner.read_calls(d / "calls.jsonl").values() if c["experiment"] == "p2_plan"]
    if not plan_calls:
        sys.exit("no p2_plan calls in this run. First: python -m rvc11 run p2_plan --run <same name> ...")
    keys = sorted({c["model"] for c in plan_calls})
    _guard_real(models, keys, a.allow_unverified)
    plan = experiments.build_p2_continue(plan_calls, scored, by_id)
    print(f"p2_continue: {len(plan_calls)} plan sessions, {len(plan) // 3} eligible, {len(plan)} calls")
    print(runner.execute(plan, models, d, by_id, workers=a.workers, max_calls=a.max_calls))
    scoring.score_run(d, by_id)
    print("report:", analyze.run(d).relative_to(ROOT))


def cmd_score(a) -> None:
    from . import analyze, bank, runner, scoring

    d = runner.run_dir(a.run)
    rows = scoring.score_run(d, {i["item_id"]: i for i in bank.load()})
    print(f"scored {len(rows)} calls -> {(d / 'scored.csv').relative_to(ROOT)}")
    print("report:", analyze.run(d).relative_to(ROOT))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="rvc11", description="Corrected rerun pipeline (v1.1).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-bank", help="rebuild data/v11 from the legacy banks").set_defaults(fn=cmd_build_bank)
    sub.add_parser("qa", help="run every QA gate offline").set_defaults(fn=cmd_qa)
    sub.add_parser("models-check", help="verify model ids against the live catalogue").set_defaults(fn=cmd_models_check)
    for name, fn, hlp in (("plan", cmd_plan, "count the calls an experiment needs"), ("run", cmd_run, "run an experiment (resumable)")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("experiment", choices=["p1_surface", "e1_instruction", "e2_noise", "p2_plan"])
        s.add_argument("--models", default="all", help="comma-separated keys from models.yaml, or 'all'")
        s.add_argument("--limit", type=int, default=None, help="problems per subtype (pilot runs)")
        if name == "run":
            s.add_argument("--run", required=True, help="run name; results go to results/v11/<name>/")
            s.add_argument("--workers", type=int, default=8)
            s.add_argument("--max-calls", type=int, default=None)
            s.add_argument("--allow-unverified", action="store_true")
            s.add_argument("--skip-qa", action="store_true")
        s.set_defaults(fn=fn)
    s = sub.add_parser("p2-continue", help="stage B of P2, built from stage-A results in the same run")
    s.add_argument("--run", required=True)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--max-calls", type=int, default=None)
    s.add_argument("--allow-unverified", action="store_true")
    s.set_defaults(fn=cmd_p2_continue)
    s = sub.add_parser("score", help="re-derive scores and the report from raw calls")
    s.add_argument("--run", required=True)
    s.set_defaults(fn=cmd_score)
    a = p.parse_args(argv)
    a.fn(a)
