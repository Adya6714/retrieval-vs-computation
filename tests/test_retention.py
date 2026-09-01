"""W3 retention floor is a single pre-specified constant."""

from probes.behavioral.retention import (
    MIN_CANONICAL_FOR_RETENTION,
    REASON_CANONICAL_BELOW_FLOOR,
    REASON_OK,
    retention_ratio,
)


def test_floor_is_documented_constant():
    assert MIN_CANONICAL_FOR_RETENTION == 0.30


def test_below_floor_is_null():
    ret, reason = retention_ratio(0.14, 0.02)
    assert ret is None
    assert reason == REASON_CANONICAL_BELOW_FLOOR


def test_at_floor_is_defined():
    ret, reason = retention_ratio(0.15, 0.30)
    assert reason == REASON_OK
    assert abs(ret - 0.5) < 1e-12


def test_uniform_not_per_family():
    llama_bw, r1 = retention_ratio(0.14, 0.02)
    gpt_bw, r2 = retention_ratio(0.14, 0.08)
    claude_wis, r3 = retention_ratio(0.04, 0.2333)
    assert llama_bw is gpt_bw is claude_wis is None
    assert r1 == r2 == r3 == REASON_CANONICAL_BELOW_FLOOR
