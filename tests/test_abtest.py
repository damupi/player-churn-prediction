import numpy as np
import pytest
from scipy.stats import chi2_contingency

from churn.abtest import sample_size_per_group, two_proportion_ztest


def test_identical_groups_give_no_difference():
    r = two_proportion_ztest(100, 1000, 100, 1000)
    assert r.diff == 0
    assert r.p_value == pytest.approx(1.0)
    assert r.ci_low < 0 < r.ci_high


def test_matches_chi_square_without_continuity_correction():
    # For two groups, the z-test p-value equals the chi-square test p-value.
    a, n_a, b, n_b = 220, 1000, 180, 1000
    r = two_proportion_ztest(a, n_a, b, n_b)
    table = [[a, n_a - a], [b, n_b - b]]
    _, p_chi2, _, _ = chi2_contingency(table, correction=False)
    assert r.p_value == pytest.approx(p_chi2, rel=1e-9)


def test_large_effect_is_significant():
    r = two_proportion_ztest(300, 1000, 200, 1000)
    assert r.p_value < 0.001
    assert r.ci_high < 0  # B churns less, the whole interval is below 0


def test_sample_size_textbook_value():
    # 10% -> 12%, alpha 0.05, power 0.8: about 3,841 per group (standard calculators agree).
    assert sample_size_per_group(0.10, 0.02) == pytest.approx(3841, abs=5)


def test_smaller_effects_need_more_players():
    assert sample_size_per_group(0.22, -0.01) > sample_size_per_group(0.22, -0.02)


def test_aa_simulation_false_positive_rate_near_alpha():
    # Two groups with the same true rate: about 5% of tests should still "find" a difference.
    rng = np.random.default_rng(0)
    p, n, runs = 0.22, 2000, 2000
    hits = sum(
        two_proportion_ztest(rng.binomial(n, p), n, rng.binomial(n, p), n).p_value < 0.05
        for _ in range(runs)
    )
    assert 0.035 < hits / runs < 0.065


def test_invalid_inputs():
    with pytest.raises(ValueError):
        sample_size_per_group(0.22, 0)
    with pytest.raises(ValueError):
        two_proportion_ztest(1, 0, 1, 10)
