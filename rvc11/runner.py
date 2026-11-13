"""Run call plans. Append-only JSONL, resumable, concurrent. Stores raw calls only.

Scores are never written here: ``python -m rvc11 score`` derives them from the raw log, so there is
exactly one scoring path and a verifier fix never needs a re-run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import __version__, bank as bank_mod
from .client import make_client

ROOT = Path(__file__).resolve().parents[1]


def run_dir(name: str) -> Path:
    d = ROOT / "results/v11" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_calls(path: Path) -> dict:
    """Latest record per call_id; an ok record is never replaced by a later error."""
    latest: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                old = latest.get(rec["call_id"])
                if old is None or old.get("status") != "ok" or rec.get("status") == "ok":
                    latest[rec["call_id"]] = rec
    return latest


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def write_manifest(d: Path, models: dict, study: dict, experiment: str) -> None:
    bank_path = bank_mod.OUT_DIR / "bank.jsonl"
    entry = {
        "experiment": experiment, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "rvc11_version": __version__, "git_commit": _git_commit(), "python": platform.python_version(),
        "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        "models": {k: {kk: vv for kk, vv in v.items() if "key_env" not in kk} for k, v in models.items()},
        "study": study["experiments"].get(experiment, {}),
    }
    path = d / "manifest.json"
    hist = json.loads(path.read_text()) if path.exists() else []
    if hist and hist[0]["bank_sha256"] != entry["bank_sha256"]:
        raise RuntimeError(
            f"{d.name} was started with a different bank (sha {hist[0]['bank_sha256'][:12]}). "
            "Use a new --run name; results from two banks must not be mixed.")
    hist.append(entry)
    path.write_text(json.dumps(hist, indent=2, default=str))


def execute(plan: list[dict], models: dict, d: Path, bank_by_id: dict, workers: int = 8, max_calls: int | None = None,
            progress_every: int = 50) -> dict:
    path = d / "calls.jsonl"
    done = {cid for cid, rec in read_calls(path).items() if rec.get("status") == "ok"}
    todo = [c for c in plan if c["call_id"] not in done]
    if max_calls is not None:
        todo = todo[:max_calls]
    clients = {mk: make_client(models[mk]) for mk in sorted({c["model"] for c in todo})}
    lock = threading.Lock()
    counts = {"planned": len(plan), "already_done": len(plan) - len([c for c in plan if c["call_id"] not in done]), "ok": 0, "error": 0}

    def one(spec: dict) -> dict:
        res = clients[spec["model"]].complete(spec["prompt"], spec["thinking"], spec["sample"], item=bank_by_id[spec["item_id"]], arm=spec.get("arm", "main"))
        rec = {**spec, **res, "model_id": models[spec["model"]]["id"],
               "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        with lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[res["status"]] += 1
            n = counts["ok"] + counts["error"]
            if n % progress_every == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} calls  ok={counts['ok']} error={counts['error']}", flush=True)
        return rec

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for fut in as_completed([pool.submit(one, c) for c in todo]):
                fut.result()
    return counts
