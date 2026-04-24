"""Strict algorithmic verifiers with parse-status tracking.

Each verifier returns ``(is_correct, reason)`` and updates ``LAST_VERIFY_META`` with:
- parse_status: parsed_clean | parsed_with_normalization | parse_failed
- auxiliary flags (e.g., correct_alternative_path, path_provided)
"""

from __future__ import annotations

import json
import re
from typing import Any


LAST_VERIFY_META: dict[str, Any] = {}


def _set_meta(**kwargs: Any) -> None:
    LAST_VERIFY_META.clear()
    LAST_VERIFY_META.update(kwargs)


def _parse_params(difficulty_params: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(difficulty_params, dict):
        return difficulty_params
    if not isinstance(difficulty_params, str):
        raise ValueError("difficulty_params must be JSON string or dict.")
    try:
        parsed = json.loads(difficulty_params)
    except json.JSONDecodeError as exc:
        raise ValueError(f"difficulty_params must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("difficulty_params JSON must decode to an object.")
    return parsed


def _parse_cc_ground_truth_count(ground_truth: str) -> int:
    m = re.search(r"(?:Count|Total)\s*:\s*(\d+)", str(ground_truth), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unable to parse ground-truth count from: {ground_truth!r}")
    return int(m.group(1))


def _extract_int_list_candidates(text: str) -> list[list[int]]:
    candidates: list[list[int]] = []
    for m in re.finditer(r"\[([^\]]+)\]", text):
        nums = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
        if nums:
            candidates.append(nums)
    for m in re.finditer(r"(?:coins?|scoops?)\s*[:\-]?\s*([0-9,\s]+)", text, flags=re.IGNORECASE):
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if nums:
            candidates.append(nums)
    return candidates


def _extract_claimed_count(text: str, target: int) -> int | None:
    for pat in (
        r"(?:Count|Total)\s*:\s*(\d+)",
        r"(\d+)\s*coins?",
        r"minimum\s+is\s+(\d+)",
        r"just\s+(\d+)",
    ):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 <= val <= max(target, 1000000):
                return val
    nums = [int(x) for x in re.findall(r"\b\d+\b", text)]
    if len(nums) == 1 and nums[0] <= max(target, 1000000):
        return nums[0]
    return None


def verify_coinchange(
    model_answer: str, ground_truth: str, difficulty_params: str | dict[str, Any]
) -> tuple[bool, str]:
    try:
        params = _parse_params(difficulty_params)
        denoms = [int(x) for x in params["denominations"]]
        target = int(params["target"])
        gt_count = _parse_cc_ground_truth_count(ground_truth)
    except Exception as exc:
        _set_meta(parse_status="parse_failed")
        return False, f"parse_failed: {exc}"

    text = re.sub(r"\s+", " ", str(model_answer or "").lower().replace("\n", " ")).strip()
    claimed_count = _extract_claimed_count(text, target)
    list_candidates = _extract_int_list_candidates(text)

    if claimed_count is None and not list_candidates:
        _set_meta(parse_status="parse_failed")
        return False, "parse_failed: unable to extract count or coin list"

    parse_status = "parsed_clean"
    if "[" not in text or "Count" not in text:
        parse_status = "parsed_with_normalization"

    chosen_list: list[int] | None = None
    if list_candidates:
        for coins in list_candidates:
            if all(c in denoms for c in coins):
                if sum(coins) == target and (claimed_count is None or len(coins) == claimed_count):
                    chosen_list = coins
                    if claimed_count is None:
                        claimed_count = len(coins)
                    break
        if chosen_list is None:
            _set_meta(parse_status=parse_status)
            return False, "invalid_coin_list: list does not match denominations/target/count"

    if claimed_count is None:
        _set_meta(parse_status="parse_failed")
        return False, "parse_failed: no valid count extracted"

    if claimed_count != gt_count:
        _set_meta(parse_status=parse_status)
        return False, f"wrong_count: predicted={claimed_count}, expected={gt_count}"

    _set_meta(parse_status=parse_status, coin_list_provided=chosen_list is not None)
    return True, "correct"


def verify_coinchange_scoops(
    model_answer: str, ground_truth: str, difficulty_params: str | dict[str, Any]
) -> tuple[bool, str]:
    try:
        params = _parse_params(difficulty_params)
        denoms = [int(x) for x in params["denominations"]]
        target = int(params["target"])
        gt_count = _parse_cc_ground_truth_count(ground_truth)
        scoop_map = params.get("scoop_names", {})
        if scoop_map is not None and not isinstance(scoop_map, dict):
            raise ValueError("scoop_names must be a mapping when provided.")
    except Exception as exc:
        _set_meta(parse_status="parse_failed")
        return False, f"parse_failed: {exc}"

    raw_text = str(model_answer or "")
    normalized = raw_text.lower().replace("\n", " ")
    # Remove unit annotations like "1g" / "1 g".
    normalized = re.sub(r"(\d)\s*g\b", r"\1", normalized)
    normalized = re.sub(r"\bg\b", "", normalized)
    # Normalize punctuation/spaces.
    normalized = normalized.replace(";", ",")
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    used_mapping = False
    for name, value in (scoop_map or {}).items():
        if not isinstance(name, str):
            continue
        key = name.lower()
        if re.search(rf"\b{re.escape(key)}\b", normalized):
            used_mapping = True
            normalized = re.sub(
                rf"\b{re.escape(key)}\b", str(int(value)), normalized
            )

    # Step 2: extract count.
    claimed_count: int | None = None
    m = re.search(r"total\s*:\s*(\d+)", normalized)
    if m:
        claimed_count = int(m.group(1))
    if claimed_count is None:
        m = re.search(r"(\d+)\s*scoops?\b", normalized)
        if m:
            claimed_count = int(m.group(1))
    all_numbers = [int(x) for x in re.findall(r"\d+", normalized)]
    if claimed_count is None:
        for n in all_numbers:
            if 0 <= n <= target:
                claimed_count = n
                break

    # Step 3: extract candidate coins from all integers, dropping first count occurrence.
    coins: list[int] = []
    if all_numbers and claimed_count is not None:
        removed = False
        for n in all_numbers:
            if not removed and n == claimed_count:
                removed = True
                continue
            coins.append(n)
    elif all_numbers:
        coins = list(all_numbers)

    if claimed_count is None and not coins:
        _set_meta(parse_status="parse_failed")
        return False, "parse_failed: unable to extract count or scoops"

    parse_status = "parsed_clean"
    if (
        "\n" in raw_text
        or "g" in raw_text.lower()
        or "[" not in raw_text
        or "total:" not in raw_text.lower()
        or used_mapping
    ):
        parse_status = "parsed_with_normalization"

    coin_list_provided = len(coins) > 0
    if coin_list_provided:
        # Edge case: explanation numbers after scoops; only use first N after count.
        if claimed_count is not None and claimed_count >= 0 and len(coins) > claimed_count:
            coins = coins[:claimed_count]
        if claimed_count is None:
            claimed_count = len(coins)
        if any(c not in denoms for c in coins):
            _set_meta(parse_status=parse_status, coin_list_provided=True)
            return False, "invalid_coin_list: contains value outside denomination set"
        if sum(coins) != target:
            _set_meta(parse_status=parse_status, coin_list_provided=True)
            return False, f"invalid_coin_list: sum={sum(coins)} target={target}"
        if len(coins) != claimed_count:
            _set_meta(parse_status=parse_status, coin_list_provided=True)
            return (
                False,
                f"invalid_coin_list: len={len(coins)} does not match count={claimed_count}",
            )

    if claimed_count is None:
        _set_meta(parse_status="parse_failed")
        return False, "parse_failed: no valid count extracted"
    if claimed_count != gt_count:
        _set_meta(parse_status=parse_status, coin_list_provided=coin_list_provided)
        return False, f"wrong_count: predicted={claimed_count}, expected={gt_count}"

    _set_meta(parse_status=parse_status, coin_list_provided=coin_list_provided)
    return True, "correct"


def _parse_sp_ground_truth(ground_truth: str) -> tuple[list[int], int]:
    m = re.search(r"Path:\s*(.+?)\s*,\s*Cost:\s*(-?\d+)", str(ground_truth))
    if not m:
        raise ValueError(f"Unable to parse SP ground truth: {ground_truth!r}")
    nodes = [int(x) for x in re.findall(r"\d+", m.group(1))]
    return nodes, int(m.group(2))


def _parse_sp_model_answer(text: str) -> tuple[list[str] | None, int | None, str]:
    status = "parsed_clean"
    cost_match = re.search(r"cost\s*[:=]\s*(-?\d+)", text, flags=re.IGNORECASE)
    cost = int(cost_match.group(1)) if cost_match else None

    path_tokens: list[str] | None = None
    if "path" in text.lower():
        m = re.search(r"path\s*:\s*([^,\n]+)", text, flags=re.IGNORECASE)
        if m:
            raw = re.sub(r"\(\s*cost\s*[:=].*?\)\s*$", "", m.group(1), flags=re.IGNORECASE).strip()
            path_tokens = [t.strip() for t in re.split(r"\s*→\s*|\s*->\s*|,\s*", raw) if t.strip()]
    if path_tokens is None:
        m = re.search(r"shortest path is\s+(.+?)\s+with cost", text, flags=re.IGNORECASE)
        if m:
            path_tokens = [t.strip() for t in re.split(r"\s*→\s*|\s*->\s*|,\s*", m.group(1)) if t.strip()]
            status = "parsed_with_normalization"
    return path_tokens, cost, status


def verify_sp(
    model_answer: str, ground_truth: str, difficulty_params: str | dict[str, Any]
) -> tuple[bool, str]:
    try:
        params = _parse_params(difficulty_params)
        edges = params["graph"]
        directed = bool(params.get("directed", True))
        source = int(params["source"])
        target = int(params["target"])
        _, gt_cost = _parse_sp_ground_truth(ground_truth)
    except Exception as exc:
        _set_meta(parse_status="parse_failed")
        return False, f"parse_failed: {exc}"

    text_raw = str(model_answer or "")
    text = text_raw.lower().replace("\n", " ").replace("→", "->")
    text = re.sub(r"\s+", " ", text).strip()

    parse_status = "parsed_clean"

    # STEP 2: robust cost extraction.
    claimed_cost: int | None = None
    for pat in (r"cost\s*[:=]\s*(-?\d+)", r"total\s+cost\s*[:=]?\s*(-?\d+)"):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            claimed_cost = int(m.group(1))
            break
    all_ints = [int(x) for x in re.findall(r"-?\d+", text)]
    if claimed_cost is None and all_ints:
        claimed_cost = all_ints[-1]
        parse_status = "parsed_with_normalization"
    if claimed_cost is None:
        _set_meta(parse_status="parse_failed")
        return False, "parse_failed: cost not found"

    label_to_id = {str(v): int(k) for k, v in params.get("node_mapping", {}).items()}
    edge_cost: dict[tuple[int, int], int] = {}
    for e in edges:
        u, v, w = int(e["u"]), int(e["v"]), int(e["w"])
        edge_cost[(u, v)] = w
        if not directed:
            edge_cost[(v, u)] = w

    # STEP 3/4: extract ALL arrow sequences and pick best candidate by claimed cost.
    arrow_chunks = re.findall(
        r"([a-z0-9][a-z0-9 ]*(?:\s*->\s*[a-z0-9][a-z0-9 ]*)+)",
        text,
        flags=re.IGNORECASE,
    )

    # STEP 4/5: W3 labels and numeric support.
    def _token_to_node(token: str) -> int:
        tok = token.strip().lower()
        # Numeric path token support (must not fail).
        if re.fullmatch(r"-?\d+", tok):
            return int(tok)

        if label_to_id:
            # Direct mapping by full label.
            for k, v in label_to_id.items():
                if str(k).lower() == tok:
                    parse_state[0] = "parsed_with_normalization"
                    return v

            # Support "A" and "hub A" via normalized letter matching.
            tok_no_hub = re.sub(r"^\s*hub\s+", "", tok).strip()
            if re.fullmatch(r"[a-z]", tok_no_hub):
                for k, v in label_to_id.items():
                    label_norm = str(k).lower().strip()
                    label_letter = re.sub(r"^\s*hub\s+", "", label_norm).strip()
                    if label_letter == tok_no_hub:
                        parse_state[0] = "parsed_with_normalization"
                        return v
            for label, idx in label_to_id.items():
                m = re.search(r"\b([a-z])\b$", str(label).lower())
                if not m:
                    continue
                letter = m.group(1)
                if tok == letter or tok == f"hub {letter}":
                    parse_state[0] = "parsed_with_normalization"
                    return idx
        raise ValueError(f"unmapped path token: {token!r}")

    def _path_cost(nodes: list[int]) -> int | None:
        total = 0
        for a, b in zip(nodes[:-1], nodes[1:]):
            if (a, b) not in edge_cost:
                return None
            total += edge_cost[(a, b)]
        return total

    path_tokens: list[str] | None = None
    chosen_nodes: list[int] | None = None
    parse_state = [parse_status]

    if arrow_chunks:
        candidates: list[tuple[list[str], list[int], int]] = []
        for chunk in arrow_chunks:
            toks = [t.strip() for t in re.split(r"\s*->\s*", chunk) if t.strip()]
            if len(toks) < 2:
                continue
            try:
                nodes = [_token_to_node(t) for t in toks]
            except Exception:
                continue
            c = _path_cost(nodes)
            if c is None:
                continue
            candidates.append((toks, nodes, c))
        if candidates:
            # Priority: any candidate path whose computed cost equals claimed cost.
            match = next((x for x in candidates if x[2] == claimed_cost), None)
            if match is None:
                # Fallback: last valid-looking arrow sequence in text.
                match = candidates[-1]
                parse_state[0] = "parsed_with_normalization"
            path_tokens, chosen_nodes, _ = match
            if len(candidates) > 1:
                parse_state[0] = "parsed_with_normalization"

    # STEP 6 fallback: no usable arrow sequence, try numeric extraction.
    if chosen_nodes is None:
        nums = [int(x) for x in re.findall(r"\d+", text)]
        if claimed_cost in nums:
            removed = False
            filtered: list[int] = []
            for n in nums:
                if not removed and n == claimed_cost:
                    removed = True
                    continue
                filtered.append(n)
            nums = filtered
        if len(nums) >= 2:
            chosen_nodes = nums
            path_tokens = [str(n) for n in nums]
            parse_state[0] = "parsed_with_normalization"

    if chosen_nodes is None:
        # cost-only answer accepted if it matches ground truth cost.
        ok = claimed_cost == gt_cost
        _set_meta(
            parse_status=parse_state[0] if parse_state[0] else "parsed_with_normalization",
            path_provided=False,
            correct_alternative_path=False,
        )
        if ok:
            return True, "correct_cost_only"
        return False, f"wrong_cost: predicted={claimed_cost}, expected={gt_cost}"

    nodes = chosen_nodes
    parse_status = parse_state[0]

    if len(nodes) < 2:
        _set_meta(parse_status="parse_failed")
        return False, "parse_failed: path too short"
    if nodes[0] != source or nodes[-1] != target:
        _set_meta(parse_status=parse_status, path_provided=True)
        return False, "path_endpoints_invalid"

    computed = 0
    for a, b in zip(nodes[:-1], nodes[1:]):
        if (a, b) not in edge_cost:
            _set_meta(parse_status=parse_status, path_provided=True)
            return False, f"path_invalid_edge: ({a},{b}) missing"
        computed += edge_cost[(a, b)]

    if computed != claimed_cost:
        _set_meta(parse_status=parse_status, path_provided=True)
        return False, f"path_cost_mismatch: path_sum={computed}, claimed={claimed_cost}"
    if claimed_cost != gt_cost:
        _set_meta(parse_status=parse_status, path_provided=True)
        return False, f"wrong_cost: predicted={claimed_cost}, expected={gt_cost}"

    gt_path, _ = _parse_sp_ground_truth(ground_truth)
    alt = nodes != gt_path
    _set_meta(
        parse_status=parse_status,
        path_provided=True,
        correct_alternative_path=alt,
    )
    return True, "correct"


def _parse_wis_ground_truth_total(ground_truth: str) -> int:
    m = re.search(r"Total:\s*(\d+)", str(ground_truth))
    if not m:
        raise ValueError(f"Unable to parse WIS ground truth total: {ground_truth!r}")
    return int(m.group(1))


def _parse_wis_model_answer(text: str) -> tuple[list[str] | None, int | None, str]:
    status = "parsed_clean"
    total_match = re.search(r"Total:\s*(\d+)", text, flags=re.IGNORECASE)
    total = int(total_match.group(1)) if total_match else None

    selected: list[str] | None = None
    set_match = re.search(r"(?:Selected|Rooms|Servers|Compounds|Spaces)\s*:\s*\{([^}]*)\}", text, flags=re.IGNORECASE)
    if set_match:
        selected = [x.strip() for x in set_match.group(1).split(",") if x.strip()]
    else:
        m = re.search(r"intervals?\s+([0-9,\s]+)\s+with total", text, flags=re.IGNORECASE)
        if m:
            selected = [x.strip() for x in m.group(1).split(",") if x.strip()]
            status = "parsed_with_normalization"
    return selected, total, status


def verify_wis(
    model_answer: str, ground_truth: str, difficulty_params: str | dict[str, Any]
) -> tuple[bool, str]:
    def _interval_triplet(interval_obj: Any) -> tuple[int, int, int]:
        if isinstance(interval_obj, dict):
            return (
                int(interval_obj["start"]),
                int(interval_obj["end"]),
                int(interval_obj["weight"]),
            )
        if isinstance(interval_obj, (list, tuple)) and len(interval_obj) >= 3:
            return (int(interval_obj[0]), int(interval_obj[1]), int(interval_obj[2]))
        raise ValueError(f"Invalid interval entry: {interval_obj!r}")

    try:
        params = _parse_params(difficulty_params)
        intervals = params["intervals"]
        gt_total = _parse_wis_ground_truth_total(ground_truth)
    except Exception as exc:
        _set_meta(parse_status="parse_failed")
        return False, f"parse_failed: {exc}"

    text = str(model_answer or "")
    selected_tokens, claimed_total, parse_status = _parse_wis_model_answer(text)
    if claimed_total is None:
        _set_meta(parse_status="parse_failed")
        return False, "parse_failed: total not found"
    if claimed_total != gt_total:
        _set_meta(parse_status=parse_status)
        return False, f"wrong_total: predicted={claimed_total}, expected={gt_total}"

    label_to_idx = {str(v): int(k) for k, v in params.get("item_mapping", {}).items()}
    selected: list[int] = []
    if selected_tokens is not None:
        try:
            if label_to_idx:
                selected = [label_to_idx[t] for t in selected_tokens]
                parse_status = "parsed_with_normalization"
            else:
                selected = [int(t) for t in selected_tokens]
        except Exception:
            _set_meta(parse_status="parse_failed")
            return False, "parse_failed: selected set parsing failed"

        for idx in selected:
            if idx < 0 or idx >= len(intervals):
                _set_meta(parse_status=parse_status)
                return False, f"selected_index_out_of_range: {idx}"

        # Independence (non-overlap)
        for i in range(len(selected)):
            si, ei, _wi = _interval_triplet(intervals[selected[i]])
            for j in range(i + 1, len(selected)):
                sj, ej, _wj = _interval_triplet(intervals[selected[j]])
                if not (ei <= sj or ej <= si):
                    _set_meta(parse_status=parse_status)
                    return False, "selected_intervals_overlap"

        computed = sum(_interval_triplet(intervals[idx])[2] for idx in selected)
        if computed != claimed_total:
            _set_meta(parse_status=parse_status)
            return False, f"selected_weight_mismatch: computed={computed}, claimed={claimed_total}"

    alt = False
    m_gt = re.search(r"\{([^}]*)\}", str(ground_truth))
    if selected_tokens is not None and m_gt:
        gt_set = {int(x.strip()) for x in m_gt.group(1).split(",") if x.strip()}
        alt = set(selected) != gt_set

    _set_meta(
        parse_status=parse_status,
        correct_alternative_set=alt,
        set_provided=selected_tokens is not None,
    )
    return True, "correct"


def verify_algo(
    problem_id: str,
    model_answer: str,
    ground_truth: str,
    problem_subtype: str,
    variant_type: str,
    difficulty_params: str | dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    subtype = str(problem_subtype).strip().lower()
    variant = str(variant_type).strip()

    if subtype == "coin_change":
        if variant == "W3":
            verified, reason = verify_coinchange_scoops(
                model_answer, ground_truth, difficulty_params
            )
        else:
            verified, reason = verify_coinchange(
                model_answer, ground_truth, difficulty_params
            )
    elif subtype == "shortest_path":
        verified, reason = verify_sp(model_answer, ground_truth, difficulty_params)
    elif subtype == "wis":
        verified, reason = verify_wis(model_answer, ground_truth, difficulty_params)
    else:
        raise ValueError(f"Unknown subtype: {problem_subtype}")

    metadata = dict(LAST_VERIFY_META)
    # Backward/forward compatibility aliases for downstream scripts.
    if "correct_alternative_path" in metadata and "alternative_path" not in metadata:
        metadata["alternative_path"] = bool(metadata["correct_alternative_path"])
    if "correct_alternative_set" in metadata and "alternative_set" not in metadata:
        metadata["alternative_set"] = bool(metadata["correct_alternative_set"])
    return verified, reason, metadata
