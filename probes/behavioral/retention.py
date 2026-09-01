"""W3 retention R_W3 = Acc_W3 / Acc_canonical.

Undefined when canonical accuracy is below MIN_CANONICAL_FOR_RETENTION.
The floor is a single pre-specified constant, applied uniformly across
families and models. Do not tune it to change which cells survive.
"""

from __future__ import annotations

MIN_CANONICAL_FOR_RETENTION = 0.30
REASON_CANONICAL_BELOW_FLOOR = "canonical_below_floor"
REASON_MISSING = "missing_canonical_or_w3"
REASON_OK = "in_range"


def retention_ratio(
    acc_w3: float | None,
    acc_canonical: float | None,
    *,
    floor: float = MIN_CANONICAL_FOR_RETENTION,
) -> tuple[float | None, str]:
    """Return (R_W3 or None, reason_code)."""
    if acc_w3 is None or acc_canonical is None:
        return None, REASON_MISSING
    try:
        a_w3 = float(acc_w3)
        a_can = float(acc_canonical)
    except (TypeError, ValueError):
        return None, REASON_MISSING
    if a_can != a_can or a_w3 != a_w3:  # NaN
        return None, REASON_MISSING
    if a_can < floor:
        return None, REASON_CANONICAL_BELOW_FLOOR
    if a_can == 0:
        return None, REASON_CANONICAL_BELOW_FLOOR
    return a_w3 / a_can, REASON_OK
