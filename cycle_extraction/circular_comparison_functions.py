import logging
from typing import Sequence

import numpy as np
from numpy import pi, arctan2, sin, cos
from scipy.stats import rankdata, chi2


def get_safe_ci(boot_angles, obs_mean, ci_level=95):
    centered = wrap_angle(boot_angles - obs_mean)
    lower_p = (100 - ci_level) / 2.0
    l_c, u_c = np.percentile(centered, [lower_p, 100 - lower_p])
    return wrap_angle(l_c + obs_mean), wrap_angle(u_c + obs_mean)


def wrap_angle(angle):
    """Bounds an angle between -pi and pi."""
    return (angle + pi) % (2 * pi) - pi


def circular_mean(angles):
    """Calculates the circular mean."""
    S = np.sum(sin(angles))
    C = np.sum(cos(angles))
    return arctan2(S, C)


def _circular_var_and_T(sin1, cos1, n1, sin2, cos2, n2, diff):
    eps = 1e-8  # buffer to prevent zero div.
    R1 = np.sqrt(sin1 ** 2 + cos1 ** 2) / n1
    R2 = np.sqrt(sin2 ** 2 + cos2 ** 2) / n2
    var1 = (1.0 - R1) / (n1 * (R1 ** 2 + eps))
    var2 = (1.0 - R2) / (n2 * (R2 ** 2 + eps))
    T = np.abs(diff) / np.sqrt(var1 + var2)
    return var1, var2, T


def _sum_sin_cos(x):
    """Returns (sum_sin, sum_cos) for input array x."""
    return np.sum(sin(x), axis=-1), np.sum(cos(x), axis=-1)


def circular_comparison(
        phases1: np.ndarray,
        phases2: np.ndarray,
        n_iters: int = 5000,
        ci_level: int = 95,
        seed: int = 42,
):
    """
    Bootstrap and permutation

    Parameters
    ----------
    phases1, phases2
        The phases of the two events in radians
    n_iters
    ci_level
        Confidence interval level between 0 and 100
    seed
        The seed for the random number generator

    Returns
    -------
    dict[str, np.ndarray | float]
        Dictionary containing comparison outputs:
        - `obs_mean1` / `obs_mean2`: Observed circular means for each input group.
        - `observed_diff`: Circular mean difference (group 1 - group 2) in radians.
        - `lower1` / `upper1`: Bootstrap confidence interval bounds for `obs_mean1`.
        - `lower2` / `upper2`: Bootstrap confidence interval bounds for `obs_mean2`.
        - `ci_lower` / `ci_upper`: Bootstrap confidence interval bounds for `observed_diff`.
        - `p_value`: Two-sided permutation p-value for the null of equal mean phase.
        - `boot_diffs`: Bootstrap distribution of circular mean differences.
        - `perm_diffs`: Permutation null distribution of circular mean differences.
    """
    rng = np.random.default_rng(seed)
    ph1, ph2 = np.asarray(phases1), np.asarray(phases2)
    n1, n2 = len(ph1), len(ph2)

    # Calculate true observed means and difference
    obs_mean1 = circular_mean(ph1)
    obs_mean2 = circular_mean(ph2)
    obs_diff = wrap_angle(obs_mean1 - obs_mean2)

    # =========================================================
    #  THE BOOTSTRAP (Calculates the physical 95% CI)
    # =========================================================
    idxs1 = rng.integers(0, n1, size=(n_iters, n1))
    idxs2 = rng.integers(0, n2, size=(n_iters, n2))

    sel1, sel2 = ph1[idxs1], ph2[idxs2]  # select samples

    sin1, cos1 = _sum_sin_cos(sel1)
    sin2, cos2 = _sum_sin_cos(sel2)

    boot_means1 = arctan2(sin1, cos1)
    boot_means2 = arctan2(sin2, cos2)
    boot_diffs = wrap_angle(boot_means1 - boot_means2)

    # Calculate CIs for individual groups (for polar plot)
    lower1, upper1 = get_safe_ci(boot_means1, obs_mean1, ci_level=ci_level)
    lower2, upper2 = get_safe_ci(boot_means2, obs_mean2, ci_level=ci_level)

    # Calculate CI for the difference (for histogram/reporting)
    ci_lower, ci_upper = get_safe_ci(boot_diffs, obs_diff, ci_level=ci_level)

    # =========================================================
    # PERMUTATION TEST
    # =========================================================
    # ---- For real phases
    sin1, cos1 = _sum_sin_cos(ph1)
    sin2, cos2 = _sum_sin_cos(ph2)
    _, _, obs_T = _circular_var_and_T(sin1, cos1, n1, sin2, cos2, n2, obs_diff)

    # ---- For permutations
    pool = np.concatenate([ph1, ph2])
    n_total = len(pool)
    shuffled_idxs = rng.random((n_iters, n_total)).argsort(axis=1)
    shuffled_pool = pool[shuffled_idxs]

    pseudo1 = shuffled_pool[:, :n1]
    pseudo2 = shuffled_pool[:, n1:]

    sin1, cos1 = _sum_sin_cos(pseudo1)
    sin2, cos2 = _sum_sin_cos(pseudo2)

    perm_diffs = wrap_angle(arctan2(sin1, cos1) - arctan2(sin2, cos2))
    _, _, perm_T = _circular_var_and_T(sin1, cos1, n1, sin2, cos2, n2, perm_diffs)
    p_value = (np.sum(perm_T >= obs_T) + 1) / (n_iters + 1)

    return {
        "obs_mean1": obs_mean1,
        "obs_mean2": obs_mean2,
        "observed_diff": obs_diff,
        "lower1": lower1,
        "upper1": upper1,
        "lower2": lower2,
        "upper2": upper2,
        "ci_lower": np.degrees(ci_lower),
        "ci_upper": np.degrees(ci_upper),
        "p_value": p_value,
        "boot_diffs": boot_diffs,
        "perm_diffs": perm_diffs,
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
    uniform_scores = 2 * pi * ranks / n_total

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
        C_j = np.sum(cos(group_scores))
        S_j = np.sum(sin(group_scores))

        # Add to the overall statistic
        W += (C_j ** 2 + S_j ** 2) / n_j

    W *= 2

    # 4. Calculate p-value based on Degrees of Freedom
    df = 2 * (k - 1)
    p_value = 1 - chi2.cdf(W, df)

    return W, p_value
