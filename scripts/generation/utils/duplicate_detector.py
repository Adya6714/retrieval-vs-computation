from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

BANK_PATHS = [
    REPO_ROOT / "data/problems/question_bank_algo.csv",
    REPO_ROOT / "data/problems/question_bank_gsm.csv",
    REPO_ROOT / "data/problems/question_bank_bw.csv",
]

STAGING_GLOB = "*_canonical.csv"
TEMPLATE_ID_RE = re.compile(r"template_id=(\w+)")
PDDL_FILENAME_RE = re.compile(r"filename=([^|]+)")


class DuplicateDetector:
    """Detect duplicate canonical instances across banks and staging files."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or REPO_ROOT
        self._keys: dict[tuple[Any, ...], str] = {}
        self._load_existing_rows()

    def _load_existing_rows(self) -> None:
        paths = list(BANK_PATHS)
        staging_dir = self.repo_root / "data/staging"
        if staging_dir.exists():
            paths.extend(sorted(staging_dir.glob(STAGING_GLOB)))

        for path in paths:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, dtype=str).fillna("")
            except Exception:
                continue
            for _, row in df.iterrows():
                row_dict = {col: str(row[col]) for col in df.columns}
                self._register_row(row_dict, origin=str(path.relative_to(self.repo_root)))

    def _register_row(self, row_dict: dict[str, Any], origin: str) -> None:
        key = self._make_key(row_dict)
        if key is None:
            return
        if key in self._keys:
            return
        pid = str(row_dict.get("problem_id", "")).strip()
        subtype = str(row_dict.get("problem_subtype", "")).strip()
        if pid:
            self._keys[key] = f"duplicate {subtype} matches existing {pid} ({origin})"
        else:
            self._keys[key] = f"duplicate {subtype} matches existing row ({origin})"

    def register(self, row_dict: dict[str, Any]) -> None:
        """Record an accepted row so later checks catch duplicates in the same run."""
        self._register_row(row_dict, origin="current_run")

    def is_duplicate(self, row_dict: dict[str, Any]) -> tuple[bool, str]:
        key = self._make_key(row_dict)
        if key is None:
            return False, ""
        reason = self._keys.get(key)
        if reason:
            return True, reason
        return False, ""

    def _make_key(self, row_dict: dict[str, Any]) -> tuple[Any, ...] | None:
        subtype = str(row_dict.get("problem_subtype", "")).strip().lower()
        if subtype == "coin_change":
            return self._key_coin_change(row_dict)
        if subtype == "shortest_path":
            return self._key_shortest_path(row_dict)
        if subtype in {"wis", "wis_independent_set"}:
            return self._key_wis(row_dict)
        if subtype == "gsm_symbolic":
            return self._key_gsm_symbolic(row_dict)
        if subtype == "blocksworld":
            return self._key_blocksworld(row_dict)
        return None

    def _parse_difficulty_params(self, row_dict: dict[str, Any]) -> dict[str, Any] | None:
        raw = str(row_dict.get("difficulty_params", "")).strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _key_coin_change(self, row_dict: dict[str, Any]) -> tuple[Any, ...] | None:
        params = self._parse_difficulty_params(row_dict)
        if params is None:
            return None
        denoms = params.get("denominations")
        target = params.get("target")
        if not isinstance(denoms, list) or target is None:
            return None
        try:
            denom_set = frozenset(int(x) for x in denoms)
            target_val = int(target)
        except (TypeError, ValueError):
            return None
        return ("coin_change", denom_set, target_val)

    def _key_shortest_path(self, row_dict: dict[str, Any]) -> tuple[Any, ...] | None:
        params = self._parse_difficulty_params(row_dict)
        if params is None:
            return None
        graph = params.get("graph")
        source = params.get("source")
        target = params.get("target")
        if not isinstance(graph, list) or source is None or target is None:
            return None
        try:
            edge_set = frozenset(
                (int(e["u"]), int(e["v"]), int(e["w"]))
                for e in graph
                if isinstance(e, dict) and {"u", "v", "w"}.issubset(e)
            )
            source_val = int(source)
            target_val = int(target)
        except (TypeError, ValueError, KeyError):
            return None
        if not edge_set:
            return None
        return ("shortest_path", edge_set, source_val, target_val)

    def _key_wis(self, row_dict: dict[str, Any]) -> tuple[Any, ...] | None:
        params = self._parse_difficulty_params(row_dict)
        if params is None:
            return None
        intervals = params.get("intervals")
        if not isinstance(intervals, list):
            return None
        triples: list[tuple[int, int, int]] = []
        for iv in intervals:
            if not isinstance(iv, dict):
                continue
            try:
                triples.append((int(iv["start"]), int(iv["end"]), int(iv["weight"])))
            except (TypeError, ValueError, KeyError):
                return None
        if not triples:
            return None
        subtype = str(row_dict.get("problem_subtype", "")).strip().lower()
        return (subtype, frozenset(triples))

    def _key_gsm_symbolic(self, row_dict: dict[str, Any]) -> tuple[Any, ...] | None:
        source = str(row_dict.get("source", "")).strip()
        m = TEMPLATE_ID_RE.search(source)
        if not m:
            return None
        return ("gsm_symbolic", m.group(1))

    def _key_blocksworld(self, row_dict: dict[str, Any]) -> tuple[Any, ...] | None:
        source = str(row_dict.get("source", "")).strip()
        m = PDDL_FILENAME_RE.search(source)
        if m:
            return ("blocksworld", m.group(1).strip())
        path_m = re.search(r"path=([^|]+)", source)
        if path_m:
            return ("blocksworld", Path(path_m.group(1).strip()).name)
        return None
