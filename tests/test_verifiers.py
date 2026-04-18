"""Tests for the answer verification logic."""

import pytest

from probes.contamination.verify import verify_answer


@pytest.mark.parametrize(
    "model_answer, correct_answer, family, expected",
    [
        ("The answer is $42.", "42", "gsm", True),
        ("The total is 43.", "42", "gsm", False),
        ("Result: 100.0", "100", "coin_change", True),
        ("no numbers here", "5", "gsm", False),
        ("", "5", "gsm", False),
    ],
)
def test_numeric_families(model_answer, correct_answer, family, expected):
    """Test correct parsing and verification of numeric answers."""
    assert verify_answer("dummy_id", model_answer, correct_answer, family) is expected


@pytest.mark.parametrize(
    "model_answer, correct_answer, expected",
    [
        ("move A to B\nmove B to C", "move A to B\nmove B to C", True),
        ("move B to C\nmove A to B", "move A to B\nmove B to C", False),
        ("move A to B", "move A to B\nmove B to C", False),
        ("MOVE a TO b\nMove B to c", "move A to B\nmove B to C", True),
    ],
)
def test_plan_families(model_answer, correct_answer, expected):
    """Test sequence matching for plan families like blocksworld."""
    assert verify_answer("dummy_id", model_answer, correct_answer, "blocksworld") is expected


def test_cross_family_unrecognized():
    """Test that an unrecognized problem family raises a ValueError."""
    with pytest.raises(ValueError):
        verify_answer("dummy_id", "42", "42", "unknown_family")


def test_never_crash_none():
    """Test that providing None as the model answer does not raise an exception."""
    assert verify_answer("dummy_id", None, "42", "gsm") is False


def test_never_crash_very_long_string():
    """Test that a very long string does not cause regex timeouts or crashes."""
    long_string = "x" * 10000
    assert verify_answer("dummy_id", long_string, "42", "gsm") is False
