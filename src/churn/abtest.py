"""
Reading an A/B test on a rate (churn, retention, conversion).

Plain formulas on purpose, so every number can be checked by hand.
"""

from dataclasses import dataclass
from math import ceil, sqrt

from scipy.stats import norm


@dataclass
class ProportionTest:
    rate_a: float
    rate_b: float
    diff: float  # rate_b - rate_a
    ci_low: float
    ci_high: float
    z: float
    p_value: float


def two_proportion_ztest(
    successes_a: int, n_a: int, successes_b: int, n_b: int, alpha: float = 0.05
) -> ProportionTest:
    """
    Is the difference between two rates bigger than chance would explain?

    - z and p-value use the *pooled* rate: under "no difference" both groups share one rate.
    - the confidence interval uses each group's own rate (unpooled), because it
      describes the difference we actually observed.
    """
    if min(n_a, n_b) <= 0:
        raise ValueError("both groups need at least one observation")
    p_a, p_b = successes_a / n_a, successes_b / n_b
    diff = p_b - p_a

    pooled = (successes_a + successes_b) / (n_a + n_b)
    se_pooled = sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = diff / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z)))

    se_diff = sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = norm.ppf(1 - alpha / 2)
    return ProportionTest(
        rate_a=p_a,
        rate_b=p_b,
        diff=diff,
        ci_low=diff - z_crit * se_diff,
        ci_high=diff + z_crit * se_diff,
        z=z,
        p_value=p_value,
    )


def sample_size_per_group(
    baseline_rate: float, min_detectable_effect: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """
    Players needed in EACH group to detect an absolute change of `min_detectable_effect`
    (e.g. churn 0.22 -> 0.20 is an effect of 0.02), two-sided test.

    alpha: chance of a false alarm (declaring a difference that isn't there).
    power: chance of catching the effect if it is really there.
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be between 0 and 1")
    if min_detectable_effect == 0:
        raise ValueError("min_detectable_effect must not be 0")
    p1 = baseline_rate
    p2 = baseline_rate + min_detectable_effect
    if not 0 < p2 < 1:
        raise ValueError("baseline_rate + min_detectable_effect must be between 0 and 1")

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    p_bar = (p1 + p2) / 2
    numerator = (
        z_alpha * sqrt(2 * p_bar * (1 - p_bar)) + z_beta * sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return ceil(numerator / min_detectable_effect**2)
