#!/usr/bin/env python3
"""Generate the behavioural model roster table for README.md from configs/models.yaml.

Writes (or refreshes) the markdown table between the markers:

    <!-- MODELS:START -->
    ...
    <!-- MODELS:END -->

Usage:
    PYTHONPATH=. python scripts/consolidate/make_model_table.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_YAML = REPO / "configs" / "models.yaml"
DEFAULT_README = REPO / "README.md"
START = "<!-- MODELS:START -->"
END = "<!-- MODELS:END -->"


def _load_roster(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for entry in data.get("roster") or []:
        rows.append(
            {
                "display_name": str(entry.get("display_name") or entry.get("name") or ""),
                "openrouter_id": str(entry.get("openrouter_id") or ""),
                "via": str(entry.get("via") or ""),
                "behavioral": bool(entry.get("behavioral", False)),
                "mechanistic": bool(entry.get("mechanistic", False)),
                "priority": str(entry.get("priority") or ""),
                "scientific_role": str(entry.get("scientific_role") or ""),
            }
        )
    for entry in data.get("open_models_local") or []:
        if str(entry.get("priority") or "") == "testing":
            continue
        rows.append(
            {
                "display_name": str(entry.get("display_name") or entry.get("name") or ""),
                "openrouter_id": str(entry.get("hf_id") or ""),
                "via": "local HF",
                "behavioral": bool(entry.get("behavioral", False)),
                "mechanistic": bool(entry.get("mechanistic", False)),
                "priority": str(entry.get("priority") or ""),
                "scientific_role": str(entry.get("notes") or entry.get("scientific_role") or ""),
            }
        )
    return rows


def render_table(rows: list[dict]) -> str:
    lines = [
        "| Model | ID | Via | Behavioural | Mechanistic | Priority | Role |",
        "|---|---|---|:---:|:---:|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {display_name} | `{openrouter_id}` | {via} | {b} | {m} | {priority} | {scientific_role} |".format(
                display_name=r["display_name"],
                openrouter_id=r["openrouter_id"],
                via=r["via"],
                b="yes" if r["behavioral"] else "no",
                m="yes" if r["mechanistic"] else "no",
                priority=r["priority"],
                scientific_role=r["scientific_role"].replace("|", "/"),
            )
        )
    return "\n".join(lines) + "\n"


def update_readme(readme: Path, table: str) -> None:
    text = readme.read_text(encoding="utf-8")
    block = f"{START}\n{table}{END}"
    if START in text and END in text:
        text = re.sub(
            re.escape(START) + r".*?" + re.escape(END),
            block,
            text,
            count=1,
            flags=re.DOTALL,
        )
    else:
        raise SystemExit(
            f"{readme} is missing {START} / {END} markers; add them before running."
        )
    readme.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the table only; do not edit README.md",
    )
    args = parser.parse_args()
    table = render_table(_load_roster(args.yaml))
    if args.stdout:
        print(table, end="")
        return
    update_readme(args.readme, table)
    print(f"Updated model table in {args.readme}")


if __name__ == "__main__":
    main()
