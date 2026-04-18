"""Tests for the output parsing utilities."""

import pytest

from probes.common.parsers import extract_numeric, extract_path, extract_plan


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The answer is 42.", 42.0),
        ("Total: $1,234", 1234.0),
        ("Step 1 gives 10, step 2 gives 20, answer is 30", 30.0),
        ("no numbers here", None),
        ("", None),
        (None, None),
        ("3.14", 3.14),
        ("-7", -7.0),
    ],
)
def test_extract_numeric(text, expected):
    """Test extracting the last numeric value from a string."""
    assert extract_numeric(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The path is A, B, C, D", "A,B,C,D"),
        ("A -> B -> C", "A,B,C"),
        ("no path here", None),
        ("", None),
        (None, None),
        ("a,b,c", "A,B,C"),
    ],
)
def test_extract_path(text, expected):
    """Test extracting and normalizing a node path from a string."""
    assert extract_path(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Step 1: move A from table to B\nStep 2: move C from D to table",
            ["move a from table to b", "move c from d to table"],
        ),
        ("move X from Y to Z", ["move x from y to z"]),
        ("no moves here", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_plan(text, expected):
    """Test extracting and normalizing Blocksworld moves from a string."""
    assert extract_plan(text) == expected
