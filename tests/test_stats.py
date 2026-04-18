"""Tests for common statistical functions."""

import math
import numpy as np
import pytest

from probes.common.stats import bootstrap_ci, wilcoxon_test, effect_size_r


# Lock seed directly at module execution root for robust reproducibility
np.random.seed(42)


def test_bootstrap_empty_list():
    """Test that an empty list properly falls back to nan boundaries without crashing."""
    lower, upper = bootstrap_ci([])
    assert math.isnan(lower)
    assert math.isnan(upper)


def test_bootstrap_single_value():
    """Test that CI brackets cleanly encompass single isolated values."""
    lower, upper = bootstrap_ci([5.0])
    assert lower <= 5.0 <= upper


def test_bootstrap_mixed_values():
    """Test that repeating fractional metrics return correctly bounded limits."""
    vals = [0.0, 1.0] * 50
    lower, upper = bootstrap_ci(vals)
    assert 0.0 <= lower <= 1.0
    assert 0.0 <= upper <= 1.0


def test_bootstrap_ci_width_positive():
    """Test that variable metrics yield functional, strictly positive confidence widths."""
    vals = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
    lower, upper = bootstrap_ci(vals)
    assert upper > lower


def test_bootstrap_identical_values():
    """Test that homogeneous metrics yield a collapsed boundary interval directly on that value."""
    vals = [3.0] * 100
    lower, upper = bootstrap_ci(vals)
    assert math.isclose(lower, 3.0, abs_tol=1e-9)
    assert math.isclose(upper, 3.0, abs_tol=1e-9)


def test_wilcoxon_mismatched_lengths():
    """Test that unaligned evaluation mappings generate ValueErrors correctly."""
    with pytest.raises(ValueError, match="same length"):
        wilcoxon_test([1.0], [1.0, 2.0])


def test_wilcoxon_identical_lists():
    """Test that identical sequences bypass zero-difference throws and safely negate significance."""
    res = wilcoxon_test([1.0] * 20, [1.0] * 20)
    assert res["p_value"] > 0.05
    assert res["significant"] is False


def test_wilcoxon_different_lists():
    """Test that widely disconnected matrices properly yield significance flags."""
    res = wilcoxon_test([0.0] * 20, [10.0] * 20)
    assert res["significant"] is True


def test_wilcoxon_warning_key_small_sample():
    """Test that lists falling below the minimum sampling bounds explicitly attach warning records."""
    res = wilcoxon_test([0.0] * 5, [1.0] * 5)
    assert "warning" in res


def test_wilcoxon_no_warning_key_large_sample():
    """Test that statistically viable sequence arrays execute free from length warnings."""
    res = wilcoxon_test([0.0] * 15, [1.0] * 15)
    assert "warning" not in res


def test_effect_size_r_bounds():
    """Test explicit constraints and bound mapping around ratio limits."""
    res = effect_size_r(100.0, 10000)
    assert 0.0 <= res <= 1.0
    assert isinstance(res, float)


def test_effect_size_r_zero():
    """Test that zero-metric values perfectly return 0.0."""
    res = effect_size_r(0.0, 100)
    assert res == 0.0
    assert isinstance(res, float)
