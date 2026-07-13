"""Statistical tests for paired D-STSGCN experiment comparisons.

Requirements: numpy and scipy must be importable.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


def _clean_paired_arrays(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired arrays and remove non-finite pairs.

    Args:
        a: Metric values for model A.
        b: Metric values for model B.

    Returns:
        Tuple of finite paired arrays with matching one-dimensional shapes.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape for a paired test.")
    finite = np.isfinite(a) & np.isfinite(b)
    return a[finite], b[finite]


def wilcoxon_paired_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Run a paired Wilcoxon signed-rank test with median-difference CI.

    Args:
        a: One-dimensional metric values for model A.
        b: One-dimensional metric values for model B, paired with ``a``.

    Returns:
        Dictionary with statistic, p_value, effect_size_r, n_pairs,
        median_diff, ci_lower_95, and ci_upper_95. The median difference and
        bootstrap CI are computed from ``a - b``.
    """
    clean_a, clean_b = _clean_paired_arrays(a, b)
    diff = clean_a - clean_b
    n_pairs = int(diff.size)
    if n_pairs == 0:
        return {
            "statistic": math.nan,
            "p_value": math.nan,
            "effect_size_r": math.nan,
            "n_pairs": 0.0,
            "median_diff": math.nan,
            "ci_lower_95": math.nan,
            "ci_upper_95": math.nan,
        }

    median_diff = float(np.median(diff))
    if np.allclose(diff, 0.0, atol=1e-12, rtol=0.0):
        statistic = 0.0
        p_value = 1.0
    else:
        result = stats.wilcoxon(clean_a, clean_b, alternative="two-sided", method="auto")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    denominator = n_pairs * (n_pairs + 1) / 2.0
    effect_size_r = (
        float(1.0 - 2.0 * statistic / denominator)
        if denominator > 0 and math.isfinite(statistic)
        else math.nan
    )

    rng = np.random.default_rng(12345)
    medians = []
    for _ in range(1000):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        medians.append(float(np.median(diff[idx])))
    ci_lower_95, ci_upper_95 = np.percentile(np.asarray(medians), [2.5, 97.5])

    return {
        "statistic": statistic,
        "p_value": p_value,
        "effect_size_r": effect_size_r,
        "n_pairs": float(n_pairs),
        "median_diff": median_diff,
        "ci_lower_95": float(ci_lower_95),
        "ci_upper_95": float(ci_upper_95),
    }


def holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni multiple-comparison correction.

    Args:
        p_values: Raw p-values in the original comparison order.

    Returns:
        Adjusted p-values in the same order as the input. NaN p-values are
        preserved as NaN and excluded from the correction count.
    """
    adjusted = [math.nan] * len(p_values)
    valid = [
        (idx, float(p_value))
        for idx, p_value in enumerate(p_values)
        if p_value is not None and math.isfinite(float(p_value))
    ]
    m = len(valid)
    if m == 0:
        return adjusted

    sorted_valid = sorted(valid, key=lambda item: item[1])
    previous = 0.0
    for rank, (idx, p_value) in enumerate(sorted_valid):
        corrected = min((m - rank) * p_value, 1.0)
        corrected = max(corrected, previous)
        adjusted[idx] = corrected
        previous = corrected
    return adjusted


def format_test_result(result: dict[str, Any], alpha: float = 0.05) -> str:
    """Format a Wilcoxon test result for reports.

    Args:
        result: Result dictionary from ``wilcoxon_paired_test``. If present,
            ``p_value_adj`` is included and used for the significance marker.
        alpha: Significance threshold for the single-star marker.

    Returns:
        Compact formatted string with W, p, adjusted p, r, median delta, CI,
        and significance stars.
    """
    p_value = float(result.get("p_value", math.nan))
    p_value_adj = result.get("p_value_adj")
    p_for_stars = float(p_value_adj) if p_value_adj is not None else p_value
    if not math.isfinite(p_for_stars) or p_for_stars >= alpha:
        stars = ""
    elif p_for_stars < 0.001:
        stars = "***"
    elif p_for_stars < 0.01:
        stars = "**"
    else:
        stars = "*"

    adj_text = ""
    if p_value_adj is not None and math.isfinite(float(p_value_adj)):
        adj_text = f"(adj p={float(p_value_adj):.3g})"

    return (
        f"W={float(result.get('statistic', math.nan)):.3g}, "
        f"p={p_value:.3g}{adj_text}, "
        f"r={float(result.get('effect_size_r', math.nan)):.3g}, "
        f"median_delta={float(result.get('median_diff', math.nan)):.3g}"
        f"[CI: {float(result.get('ci_lower_95', math.nan)):.3g}, "
        f"{float(result.get('ci_upper_95', math.nan)):.3g}]{stars}"
    )


def moving_block_bootstrap_difference(
    a: np.ndarray,
    b: np.ndarray,
    block_length: int = 12,
    repetitions: int = 2000,
    statistic: str = "mean",
    seed: int = 12345,
) -> dict[str, float]:
    """Bootstrap a paired time-series difference using contiguous blocks."""
    clean_a, clean_b = _clean_paired_arrays(a, b)
    diff = clean_a - clean_b
    n = diff.size
    if n == 0:
        return {"estimate": math.nan, "ci_lower_95": math.nan, "ci_upper_95": math.nan, "n": 0.0}
    block_length = max(1, min(int(block_length), n))
    repetitions = max(int(repetitions), 1)
    reducer = np.median if statistic == "median" else np.mean
    rng = np.random.default_rng(seed)
    starts = np.arange(max(n - block_length + 1, 1))
    samples = np.empty(repetitions, dtype=float)
    blocks_needed = int(np.ceil(n / block_length))
    for rep in range(repetitions):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sampled = np.concatenate([diff[start : start + block_length] for start in chosen])[:n]
        samples[rep] = float(reducer(sampled))
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return {
        "estimate": float(reducer(diff)),
        "ci_lower_95": float(lower),
        "ci_upper_95": float(upper),
        "n": float(n),
        "block_length": float(block_length),
        "repetitions": float(repetitions),
    }


def diebold_mariano_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    hac_lag: int = 11,
) -> dict[str, float | str]:
    """Run a two-sided Diebold-Mariano test with Newey-West HAC variance."""
    clean_a, clean_b = _clean_paired_arrays(loss_a, loss_b)
    differential = clean_a - clean_b
    n = differential.size
    if n < 3:
        return {"statistic": math.nan, "p_value": math.nan, "n": float(n), "warning": "insufficient samples"}
    centered = differential - differential.mean()
    lag = max(0, min(int(hac_lag), n - 2))
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    for k in range(1, lag + 1):
        covariance = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    variance_mean = long_run_variance / n
    if not np.isfinite(variance_mean) or variance_mean <= 1e-15:
        return {"statistic": math.nan, "p_value": math.nan, "n": float(n), "warning": "non-positive HAC variance"}
    statistic_value = float(differential.mean() / np.sqrt(variance_mean))
    p_value = float(2.0 * stats.norm.sf(abs(statistic_value)))
    return {
        "statistic": statistic_value,
        "p_value": p_value,
        "mean_loss_difference": float(differential.mean()),
        "n": float(n),
        "hac_lag": float(lag),
        "warning": "",
    }