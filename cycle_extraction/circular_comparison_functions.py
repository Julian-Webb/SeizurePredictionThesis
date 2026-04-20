import logging
from typing import Sequence

import numpy as np
from scipy.stats import rankdata, chi2


def get_safe_ci(boot_angles, obs_mean, ci_level=95):
    centered = wrap_angle(boot_angles - obs_mean)
    lower_p = (100 - ci_level) / 2.0
    l_c, u_c = np.percentile(centered, [lower_p, 100 - lower_p])
    return wrap_angle(l_c + obs_mean), wrap_angle(u_c + obs_mean)


def wrap_angle(angle):
    """Bounds an angle between -pi and pi."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def circular_mean(angles):
    """Calculates the circular mean."""
    S = np.sum(np.sin(angles))
    C = np.sum(np.cos(angles))
    return np.arctan2(S, C)


def circular_comparison(phases1: np.ndarray, phases2: np.ndarray, n_iters: int = 10000, ci_level: int = 95,
                        seed: int = 42):
    """
    Bootstrap and permutation

    Parameters
    ----------
    phases1, phases2
        The phases of the two events in radians
    n_iters
    ci_level
        Confidence interval level
    seed
        The seed for numpy.random

    Returns
    -------
    dict[str, np.ndarray | float]
        Dictionary containing comparison outputs:
        - ``observed_diff``: Circular mean difference (group 1 - group 2) in radians.
        - ``ci_lower`` / ``ci_upper``: Bootstrap confidence interval bounds for ``observed_diff``.
        - ``p_value``: Two-sided permutation p-value for the null of equal mean phase.
        - ``boot_diffs``: Bootstrap distribution of circular mean differences.
        - ``perm_diffs``: Permutation null distribution of circular mean differences.
        - ``obs_mean1`` / ``obs_mean2``: Observed circular means for each input group.
        - ``lower1`` / ``upper1``: Bootstrap confidence interval bounds for ``obs_mean1``.
        - ``lower2`` / ``upper2``: Bootstrap confidence interval bounds for ``obs_mean2``.
    """
    np.random.seed(seed)
    ph1 = np.asarray(phases1)
    ph2 = np.asarray(phases2)
    n1, n2 = len(ph1), len(ph2)

    # Calculate true observed means and difference
    obs_mean1 = circular_mean(ph1)
    obs_mean2 = circular_mean(ph2)
    obs_diff = wrap_angle(obs_mean1 - obs_mean2)

    # =========================================================
    #  THE BOOTSTRAP (Calculates the physical 95% CI)
    # =========================================================
    idxs1 = np.random.randint(0, n1, size=(n_iters, n1))
    idxs2 = np.random.randint(0, n2, size=(n_iters, n2))

    S1, C1 = np.sum(np.sin(ph1[idxs1]), axis=1), np.sum(np.cos(ph1[idxs1]), axis=1)
    S2, C2 = np.sum(np.sin(ph2[idxs2]), axis=1), np.sum(np.cos(ph2[idxs2]), axis=1)

    boot_means1 = np.arctan2(S1, C1)
    boot_means2 = np.arctan2(S2, C2)
    boot_diffs = wrap_angle(boot_means1 - boot_means2)

    # Calculate CIs for individual groups (for polar plot)
    lower1, upper1 = get_safe_ci(boot_means1, obs_mean1, ci_level=ci_level)
    lower2, upper2 = get_safe_ci(boot_means2, obs_mean2, ci_level=ci_level)

    # Calculate CI for the difference (for histogram/reporting)
    ci_lower, ci_upper = get_safe_ci(boot_diffs, obs_diff, ci_level=ci_level)

    # Permutation for p_val
    pool = np.concatenate([ph1, ph2])
    n_total = len(pool)

    shuffled_idxs = np.random.rand(n_iters, n_total).argsort(axis=1)
    shuffled_pool = pool[shuffled_idxs]

    pseudo1 = shuffled_pool[:, :n1]
    pseudo2 = shuffled_pool[:, n1:]

    S_p1, C_p1 = np.sum(np.sin(pseudo1), axis=1), np.sum(np.cos(pseudo1), axis=1)
    S_p2, C_p2 = np.sum(np.sin(pseudo2), axis=1), np.sum(np.cos(pseudo2), axis=1)

    perm_diffs = wrap_angle(np.arctan2(S_p1, C_p1) - np.arctan2(S_p2, C_p2))
    p_value = (np.sum(np.abs(perm_diffs) >= np.abs(obs_diff)) + 1) / (n_iters + 1)

    return {
        "observed_diff": obs_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "boot_diffs": boot_diffs,
        "perm_diffs": perm_diffs,
        "obs_mean1": obs_mean1,
        "lower1": lower1,
        "upper1": upper1,
        "obs_mean2": obs_mean2,
        "lower2": lower2,
        "upper2": upper2
    }


def watson_wheeler_test(samples: Sequence[np.ndarray]):
    """
    Performs the (Mardia-)Watson-Wheeler test for equal circular distributions.

    Parameters:
    samples: 1D numpy arrays of circular data (in radians).
               e.g., watson_wheeler_test(group1, group2, group3)

    Returns:
    W : float, the test statistic
    p_value : float, the p-value
    """
    pooled = np.concatenate(samples)
    n_total = len(pooled)
    k = len(samples)

    # The Chi-square approximation is generally valid for N >= 15
    if n_total < 15:
        logging.warning("Warning: Chi-square approximation may be inaccurate for N < 15.")

    # 1. Rank the pooled data
    ranks = rankdata(pooled)

    # 2. Convert ranks to uniform scores (angles in radians)
    uniform_scores = 2 * np.pi * ranks / n_total

    # 3. Calculate the test statistic W
    W = 0
    current_idx = 0
    for sample in samples:
        n_j = len(sample)
        if n_j == 0:
            continue

        # Isolate the uniform scores for the current group
        group_scores = uniform_scores[current_idx: current_idx + n_j]
        current_idx += n_j

        # Calculate vector sums for the group
        C_j = np.sum(np.cos(group_scores))
        S_j = np.sum(np.sin(group_scores))

        # Add to the overall statistic
        W += (C_j ** 2 + S_j ** 2) / n_j

    W *= 2

    # 4. Calculate p-value based on Degrees of Freedom
    df = 2 * (k - 1)
    p_value = 1 - chi2.cdf(W, df)

    return W, p_value
