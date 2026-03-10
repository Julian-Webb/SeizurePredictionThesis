import numpy as np
import pandas as pd


def weighted_mean(w, z) -> float:
    return (w * z).sum() / w.sum()


def weighted_std(w, z) -> float:
    return np.sqrt((weighted_mean(w, z ** 2)) - (weighted_mean(w, z) ** 2))


def correlation_of_prediction_errors(
        pi: np.ndarray,
        pj: np.ndarray,
        L: np.ndarray,
) -> float:
    r"""
    Compute the weighted correlation of prediction errors.

    .. math::
    c_w =
    \frac{
    \left\langle (\vec{e}_i - \langle \vec{e}_i \rangle_w)
    (\vec{e}_j - \langle \vec{e}_j \rangle_w)
    \right\rangle_w }{
    \sigma_w(\vec{e}_i)\,\sigma_w(\vec{e}_j)
    }

    with:
    Prediction errors:
    .. math::
    \vec{e}_i = |\vec{p}_i - \vec{L}|

    Weights:
    .. math::
    w = \max(\vec{e}_i, \vec{e}_j)

    Weighted means:
    .. math::
    \langle z \rangle_w = \frac{\sum_k w_k z_k}{\sum_k w_k}

    Weighted standard deviation:
    .. math::
    \sigma_w(z) = \sqrt{\langle z^2 \rangle_w - \langle z \rangle_w^2}

    :param pi: Predictions of model i
    :param pj: Predictions of model j
    :param L: Ground truth labels (y_true)
    """
    # Errors
    ei = np.abs(pi - L)  # (n_samples,)
    ej = np.abs(pj - L)  # (n_samples,)
    errors = np.vstack((ei, ej))  # (2, n_samples)
    w = errors.max(axis=0)  # (n_samples,) - weights

    mi = weighted_mean(w, ei)  # float
    mj = weighted_mean(w, ej)  # float

    prod = (ei - mi) * (ej - mj)  # (n_samples,)
    nominator = weighted_mean(w, prod)  # float

    std_i = weighted_std(w, ei)  # float
    std_j = weighted_std(w, ej)  # float
    denominator = std_i * std_j  # float

    # Guard against zero variance leading to division by zero
    if denominator == 0.0:
        # If both error vectors have zero variance
        if std_i == 0.0 and std_j == 0.0:
            # If errors are identical, define perfect correlation; otherwise undefined -> 0.0
            return 1.0 if np.allclose(ei, ej) else 0.0
        # If only one has zero variance, no linear relationship can be established -> 0.0
        return 0.0

    return nominator / denominator


if __name__ == '__main__':
    from config.paths import PATHS
    from utils.io import pickle_path

    for pdir in PATHS.patient_dirs():
        clips = pd.read_pickle(pickle_path(pdir.clips_table))
        clips = clips[clips['valid']]
        y_true = clips['preictal'].values
        pi = clips['ensemble_probability'].values
        pj = clips['CNN_probability'].values

        corr = correlation_of_prediction_errors(pi, pj, y_true)
        print(f'{pdir.name}: {corr}')
