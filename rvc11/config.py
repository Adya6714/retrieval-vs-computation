"""Config loading. configs/v11/models.yaml and configs/v11/study.yaml are the only knobs."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs/v11"
LOCK = CFG_DIR / "models.lock.json"


def _onoff(x):
    """YAML 1.1 reads bare on/off as booleans. Normalise them back to the strings "on"/"off"."""
    if isinstance(x, bool):
        return "on" if x else "off"
    if isinstance(x, list):
        return [_onoff(v) for v in x]
    if isinstance(x, dict):
        return {_onoff(k) if isinstance(k, bool) else k: v for k, v in x.items()}
    return x


def load_models(path: Path | None = None) -> dict:
    raw = yaml.safe_load((path or CFG_DIR / "models.yaml").read_text())
    out = {}
    for key, m in raw["models"].items():
        cfg = {**raw.get("defaults", {}), **m, "key": key}
        cfg["thinking"] = _onoff(cfg.get("thinking") or {"on": {}})
        out[key] = cfg
    return out


def load_study(path: Path | None = None) -> dict:
    study = yaml.safe_load((path or CFG_DIR / "study.yaml").read_text())
    for exp in study["experiments"].values():
        exp["thinking"] = _onoff(exp["thinking"])
    return study


def locked_ids() -> set:
    if not LOCK.exists():
        return set()
    return set(json.loads(LOCK.read_text()).get("verified_ids", []))
