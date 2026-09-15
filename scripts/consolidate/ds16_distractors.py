#!/usr/bin/env python3
"""DS-16 recognition distractors — deterministic, scripted, documented.

Used by ``colab/ds16_recognition_recall.ipynb``. Do not hand-write options.

## Design

Recognition presents the gold answer plus **k=4** distractors. Each option is
scored by teacher-forced ``mean_logprob`` under the same Probe-1 prompt as O5.
Recognition is correct iff gold ranks first by ``mean_logprob``.

Distractors are family-specific and **deterministic** given
``(family, problem_id, variant, gold, problem_text)`` via
``hashlib.sha256`` → ``random.Random(seed)``. Re-running yields identical sets.

### BW — valid-looking plans that fail the goal
Mutations of the gold action list (must stay in
``{pick-up, put-down, stack, unstack}`` syntax):
  1. Drop the last action.
  2. Swap two action lines (seeded indices).
  3. Replace one block argument with another block from the gold plan.
  4. Duplicate the first action at the end (extra useless step).
If a mutation equals gold or another distractor, fall back to a seeded
block-rename of the first argument.

### GSM — arithmetic slips
Parse the numeric gold (optional ``####``). Emit, in order, until 4 unique:
  gold+1, gold-1, gold+10, gold-10, 2*gold, gold//10 (if |gold|≥10),
  digit-transposition of the decimal representation, gold+2.
Format matches gold (``#### N`` if gold used that tag, else bare number).

### ALGO — wrong-but-related algorithm outputs
Branch on ``problem_id`` prefix / gold shape:
  - **SP** (``Path:``): drop an intermediate node; reverse path; change Cost by
    ±1 / ±5; swap two path nodes.
  - **CC** (``Count:`` / ``Coins:``, or W3 ``Total:`` / ``Scoops:``): Count±1;
    permute coin multiset; singleton wrong coin; empty-ish alternative count.
    Surface labels follow the gold form.
  - **WIS** (``Selected:`` / ``Total:``): drop one selected index; add an
    off-by-one index; Total±1 / ±5; swap two selected ids.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Iterable

K_DEFAULT = 4

_BW_ACTION = re.compile(
    r"^(pick-up|put-down|stack|unstack)\s+(\S+)(?:\s+(\S+))?$",
    re.I,
)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def distractor_seed(
    family: str,
    problem_id: str,
    variant: str,
    gold: str,
) -> int:
    key = f"DS16|{family}|{problem_id}|{variant}|{gold.strip()}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _norm_opt(s: str) -> str:
    """Normalize for uniqueness (collapse whitespace; keep newlines as \\n)."""
    return " ".join(str(s).split())


def _uniq(options: Iterable[str], gold: str, k: int) -> list[str]:
    out: list[str] = []
    seen = {_norm_opt(gold)}
    for o in options:
        s = str(o).strip()
        key = _norm_opt(s)
        if not s or key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= k:
            break
    return out


def _bw_blocks(plan: str) -> list[str]:
    blocks: list[str] = []
    for line in plan.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        # strip numbering
        line = re.sub(r"^\d+[\).\:\-]\s*", "", line)
        m = _BW_ACTION.match(line)
        if not m:
            continue
        blocks.append(m.group(2))
        if m.group(3):
            blocks.append(m.group(3))
    # unique preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    return uniq


def _bw_lines(plan: str) -> list[str]:
    lines: list[str] = []
    for line in plan.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\).\:\-]\s*", "", line)
        if _BW_ACTION.match(line):
            lines.append(line)
    return lines


def distractors_bw(gold: str, rng: random.Random, k: int = K_DEFAULT) -> list[str]:
    lines = _bw_lines(gold)
    blocks = _bw_blocks(gold)
    if len(lines) < 1:
        return [f"pick-up z{i}" for i in range(k)]

    cands: list[str] = []

    # 1. Drop last action
    if len(lines) >= 2:
        cands.append("\n".join(lines[:-1]))
    else:
        cands.append(lines[0] + "\nput-down " + (blocks[0] if blocks else "a"))

    # 2. Swap two lines
    if len(lines) >= 2:
        i, j = sorted(rng.sample(range(len(lines)), 2))
        swapped = lines.copy()
        swapped[i], swapped[j] = swapped[j], swapped[i]
        cands.append("\n".join(swapped))
    else:
        cands.append("put-down " + (blocks[0] if blocks else "a"))

    # 3. Replace one block arg with another
    if blocks:
        src = lines[rng.randrange(len(lines))]
        m = _BW_ACTION.match(src)
        alt_blocks = [b for b in blocks if b != (m.group(2) if m else "")]
        if m and alt_blocks:
            new_b = rng.choice(alt_blocks)
            if m.group(1).lower() in {"stack", "unstack"} and m.group(3):
                # change second arg
                mut = f"{m.group(1)} {m.group(2)} {new_b}"
            else:
                mut = f"{m.group(1)} {new_b}" + (
                    f" {m.group(3)}" if m.group(3) else ""
                )
            mut_lines = lines.copy()
            mut_lines[lines.index(src)] = mut
            cands.append("\n".join(mut_lines))
        else:
            cands.append("\n".join(lines + [f"pick-up {blocks[0]}"]))
    else:
        cands.append("\n".join(lines + ["pick-up a"]))

    # 4. Duplicate first action at end
    cands.append("\n".join(lines + [lines[0]]))

    # Fallbacks
    fb_i = 0
    while len(_uniq(cands, gold, k)) < k:
        b = blocks[fb_i % len(blocks)] if blocks else "a"
        cands.append(f"pick-up {b}\nput-down {b}")
        fb_i += 1
        if fb_i > 20:
            break
    return _uniq(cands, gold, k)


def _parse_gsm_number(gold: str) -> tuple[float, bool, bool]:
    """Return (value, used_hash_tag, is_int_like)."""
    s = gold.strip()
    used_hash = bool(re.match(r"^####\s*", s))
    s2 = re.sub(r"^####\s*", "", s).replace(",", "").strip()
    # last number if free text
    nums = _NUM.findall(s2)
    if not nums:
        return 0.0, used_hash, True
    raw = nums[-1]
    val = float(raw)
    is_int = "." not in raw
    return val, used_hash, is_int


def _fmt_gsm(val: float, used_hash: bool, is_int: bool) -> str:
    if is_int and float(val).is_integer():
        body = str(int(val))
    else:
        body = str(val)
        if body.endswith(".0") and is_int:
            body = body[:-2]
    return f"#### {body}" if used_hash else body


def distractors_gsm(gold: str, rng: random.Random, k: int = K_DEFAULT) -> list[str]:
    val, used_hash, is_int = _parse_gsm_number(gold)
    cands: list[str] = []

    def add(v: float) -> None:
        cands.append(_fmt_gsm(v, used_hash, is_int))

    add(val + 1)
    add(val - 1)
    add(val + 10)
    add(val - 10)
    add(2 * val)
    if abs(val) >= 10:
        add(val / 10 if not is_int else float(int(val) // 10))
    # digit transposition
    digits = [c for c in str(int(abs(val))) if c.isdigit()]
    if len(digits) >= 2:
        i = rng.randrange(len(digits) - 1)
        digits[i], digits[i + 1] = digits[i + 1], digits[i]
        transposed = int("".join(digits))
        if val < 0:
            transposed = -transposed
        add(float(transposed))
    add(val + 2)
    add(val - 2)
    add(0.0)
    return _uniq(cands, gold, k)


def _sp_distractors(gold: str, rng: random.Random) -> list[str]:
    cands: list[str] = []
    path_m = re.search(r"Path:\s*(.+?)(?:,\s*Cost:|$)", gold, re.I | re.S)
    cost_m = re.search(r"Cost:\s*(-?\d+)", gold, re.I)
    path = path_m.group(1).strip() if path_m else ""
    cost = int(cost_m.group(1)) if cost_m else 0
    nodes = [n.strip() for n in re.split(r"\s*→\s*|\s*->\s*", path) if n.strip()]
    if len(nodes) >= 3:
        drop = nodes.copy()
        drop.pop(rng.randrange(1, len(drop) - 1))
        cands.append(f"Path: {' → '.join(drop)}, Cost: {cost}")
        rev = nodes[::-1]
        cands.append(f"Path: {' → '.join(rev)}, Cost: {cost}")
        swap = nodes.copy()
        i, j = sorted(rng.sample(range(len(swap)), 2))
        swap[i], swap[j] = swap[j], swap[i]
        cands.append(f"Path: {' → '.join(swap)}, Cost: {cost}")
    if nodes:
        cands.append(f"Path: {' → '.join(nodes)}, Cost: {cost + 1}")
        cands.append(f"Path: {' → '.join(nodes)}, Cost: {max(0, cost - 1)}")
        cands.append(f"Path: {' → '.join(nodes)}, Cost: {cost + 5}")
    return cands


def _cc_distractors(gold: str, rng: random.Random) -> list[str]:
    """Coin-change / scoop-change: Count±1, permute multiset, singleton, empty.

    Preserves W3 surface form ``Total:`` / ``Scoops:`` when present.
    """
    cands: list[str] = []
    scoops = bool(re.search(r"Scoops:\s*\[", gold, re.I)) or (
        bool(re.search(r"Total:\s*-?\d+", gold, re.I))
        and not bool(re.search(r"Count:\s*-?\d+", gold, re.I))
    )
    count_lab, list_lab = ("Total", "Scoops") if scoops else ("Count", "Coins")
    count_m = re.search(rf"{count_lab}:\s*(-?\d+)", gold, re.I)
    if count_m is None:
        count_m = re.search(r"(?:Count|Total):\s*(-?\d+)", gold, re.I)
    coins_m = re.search(rf"{list_lab}:\s*\[([^\]]*)\]", gold, re.I)
    if coins_m is None:
        coins_m = re.search(r"(?:Coins|Scoops):\s*\[([^\]]*)\]", gold, re.I)
    count = int(count_m.group(1)) if count_m else 0
    coins_raw = coins_m.group(1) if coins_m else ""
    coins = [c.strip() for c in coins_raw.split(",") if c.strip()]

    def fmt(cnt: int, body: str) -> str:
        return f"{count_lab}: {cnt}\n{list_lab}: [{body}]"

    cands.append(fmt(count + 1, coins_raw))
    cands.append(fmt(max(0, count - 1), coins_raw))
    if len(coins) >= 2:
        perm = coins.copy()
        i, j = sorted(rng.sample(range(len(perm)), 2))
        perm[i], perm[j] = perm[j], perm[i]
        cands.append(fmt(count, ", ".join(perm)))
        cands.append(fmt(count, coins[0]))
    else:
        cands.append(fmt(count + 2, coins_raw))
        cands.append(fmt(count, "1"))
    cands.append(fmt(0, ""))
    return cands


def _wis_distractors(gold: str, rng: random.Random) -> list[str]:
    cands: list[str] = []
    sel_m = re.search(r"Selected:\s*\{([^}]*)\}", gold, re.I)
    tot_m = re.search(r"Total:\s*(-?\d+)", gold, re.I)
    total = int(tot_m.group(1)) if tot_m else 0
    ids = []
    if sel_m:
        ids = [x.strip() for x in sel_m.group(1).split(",") if x.strip()]
    if len(ids) >= 2:
        drop = ids.copy()
        drop.pop(rng.randrange(len(drop)))
        cands.append(f"Selected: {{{', '.join(drop)}}}, Total: {total}")
        swap = ids.copy()
        i, j = sorted(rng.sample(range(len(swap)), 2))
        swap[i], swap[j] = swap[j], swap[i]
        cands.append(f"Selected: {{{', '.join(swap)}}}, Total: {total}")
        # add off-by-one numeric id if possible; else a labeled extra
        try:
            nums = [int(x) for x in ids]
            extra = max(nums) + 1
            cands.append(
                f"Selected: {{{', '.join(ids + [str(extra)])}}}, Total: {total}"
            )
        except ValueError:
            cands.append(
                f"Selected: {{{', '.join(ids + ['Item Z'])}}}, Total: {total}"
            )
    elif ids:
        cands.append(f"Selected: {{}}, Total: {total}")
        cands.append(f"Selected: {{{ids[0]}}}, Total: {total + 1}")
    cands.append(f"Selected: {{{', '.join(ids) if ids else '0'}}}, Total: {total + 1}")
    cands.append(f"Selected: {{{', '.join(ids) if ids else '0'}}}, Total: {max(0, total - 1)}")
    cands.append(f"Selected: {{{', '.join(ids) if ids else '0'}}}, Total: {total + 5}")
    return cands


def distractors_algo(
    problem_id: str,
    gold: str,
    rng: random.Random,
    k: int = K_DEFAULT,
) -> list[str]:
    pid = problem_id.upper()
    g = gold.strip()
    gl = g.lower()
    if pid.startswith("SP") or gl.startswith("path:"):
        cands = _sp_distractors(g, rng)
    elif pid.startswith("CC") or "coins:" in gl or "scoops:" in gl or (
        gl.startswith("count:") or gl.startswith("total:")
    ):
        cands = _cc_distractors(g, rng)
    elif pid.startswith("WIS") or "selected:" in gl:
        cands = _wis_distractors(g, rng)
    else:
        cands = [g + " ", g + "\n", "INVALID", "0", "None"]
    # pad
    i = 0
    while len(_uniq(cands, gold, k)) < k:
        cands.append(f"{g} #alt{i}")
        i += 1
        if i > 10:
            break
    return _uniq(cands, gold, k)


def make_distractors(
    family: str,
    problem_id: str,
    variant: str,
    gold: str,
    problem_text: str = "",
    k: int = K_DEFAULT,
) -> list[str]:
    """Return exactly k distractors (or fewer only if impossible)."""
    _ = problem_text  # reserved for future state-aware BW failures
    seed = distractor_seed(family, problem_id, variant, gold)
    rng = random.Random(seed)
    fam = family.upper()
    if fam == "BW":
        out = distractors_bw(gold, rng, k=k)
    elif fam == "GSM":
        out = distractors_gsm(gold, rng, k=k)
    elif fam == "ALGO":
        out = distractors_algo(problem_id, gold, rng, k=k)
    else:
        raise ValueError(f"unknown family {family}")
    if len(out) < k:
        # last-resort padding (still deterministic)
        while len(out) < k:
            out.append(f"__pad_{seed}_{len(out)}__")
    return out[:k]


def recognition_options(
    family: str,
    problem_id: str,
    variant: str,
    gold: str,
    problem_text: str = "",
    k: int = K_DEFAULT,
) -> list[dict]:
    """Gold + k distractors with stable option ids; gold always included.

    Option order is shuffled with the same seed so position is not a cue that
    is constant across items, but is reproducible.
    """
    distractors = make_distractors(
        family, problem_id, variant, gold, problem_text=problem_text, k=k
    )
    opts = [{"option_id": "gold", "is_gold": True, "text": gold.strip()}]
    for i, d in enumerate(distractors):
        opts.append({"option_id": f"d{i}", "is_gold": False, "text": d})
    rng = random.Random(distractor_seed(family, problem_id, variant, gold) ^ 0xABCDE)
    order = list(range(len(opts)))
    rng.shuffle(order)
    return [opts[i] for i in order]
