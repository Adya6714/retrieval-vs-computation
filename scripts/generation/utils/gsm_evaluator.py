from __future__ import annotations

import re

_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
_LHS_EXPR = rf"{_NUMBER}\s*[\+\-\*/]\s*{_NUMBER}"
_EXPR_RE = re.compile(rf"(?P<lhs>{_LHS_EXPR})(?:\s*=\s*(?P<rhs>{_NUMBER}))?")


def _parse_number(text: str) -> float:
    return float(text.replace(",", "").strip())


def _safe_eval_numeric_expr(expr: str) -> float:
    clean = expr.replace(",", "").strip()
    if not re.fullmatch(r"[\d\.\+\-\*\/\s]+", clean):
        raise ValueError("unsafe_expression")
    return float(eval(clean, {"__builtins__": {}}, {}))


def verify_arithmetic_chain(solution_steps: str, stated_answer: str | int) -> tuple[bool, str]:
    """Validate arithmetic consistency in solution steps.

    Rules:
    1) Find arithmetic expressions of form "a op b" or "a op b = c".
    2) Evaluate lhs safely.
    3) If rhs exists, compare lhs and rhs within 0.01.
    4) Last evaluated value must match stated_answer within 0.01.
    5) If none found, return (False, "no_arithmetic_found").
    """
    lines = [ln.strip() for ln in str(solution_steps or "").splitlines() if ln.strip()]
    stated_match = re.search(_NUMBER, str(stated_answer))
    if not stated_match:
        return False, "invalid_stated_answer"

    try:
        stated_val = _parse_number(stated_match.group(0))
    except ValueError:
        return False, "invalid_stated_answer"

    found_any = False
    last_value: float | None = None

    for line in lines:
        matches = list(_EXPR_RE.finditer(line))
        if not matches:
            continue
        for match in matches:
            found_any = True
            lhs_text = match.group("lhs")
            rhs_text = match.group("rhs")
            try:
                lhs_val = _safe_eval_numeric_expr(lhs_text)
            except Exception:
                return False, "unsafe_or_invalid_expression"

            if rhs_text is not None:
                try:
                    rhs_val = _parse_number(rhs_text)
                except ValueError:
                    return False, "invalid_explicit_result"
                if abs(lhs_val - rhs_val) > 0.01:
                    return False, "expression_result_mismatch"
                last_value = rhs_val
            else:
                last_value = lhs_val

    if not found_any:
        return False, "no_arithmetic_found"
    if last_value is None:
        return False, "no_terminal_value"
    if abs(last_value - stated_val) > 0.01:
        return False, "final_answer_mismatch"
    return True, ""
