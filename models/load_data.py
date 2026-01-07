import numpy as np
import pandas as pd
from pandas import DataFrame

from config.constants import MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO
from config.paths import PATHS
from utils.io import pickle_path


def _subsample_interictal_train_segs(ts: DataFrame, random_state: int = None) -> DataFrame:
    """
    Randomly choose a subset of the interictal segments to be used for training, taking into consideration the
    MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO
    :param ts: train_segs - All segments used for training
    :return: A subset of the interictal segs in random order. Same format as input DataFrame.
    """
    # Note: index stays the same -> still works for original DataFrame
    n_preictal = ts[ts['type'] == 'preictal'].shape[0]
    max_interictal = n_preictal * MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO

    interictal_mask = ts['type'] == 'interictal'
    n_interictal = interictal_mask.sum()
    # In case there are less interictal segments than max_interictal, use all:
    n = min(n_interictal, max_interictal)

    selected_interictal = ts[interictal_mask].sample(n, random_state=random_state)
    return selected_interictal


def subsample_shuffle_convert_train_segs(train_segs: DataFrame, feature_cols: list[str], random_state: int = None) -> \
        tuple[np.ndarray, np.ndarray]:
    """
    Subsample interictal segments to achieve an acceptable ratio between preictal and interictal segs.
    Shuffle the segments and convert them to numpy arrays.
    Segs that aren't of type preictal or interictal are dropped.
    :param train_segs:
    :param feature_cols:
    :param random_state:
    :return: x_train, y_train
    """
    # Subsample
    selected_interictal = _subsample_interictal_train_segs(train_segs, random_state)
    # Shuffle
    preictal = train_segs[train_segs['type'] == 'preictal']
    train_segs = pd.concat([preictal, selected_interictal])
    train_segs = train_segs.sample(frac=1, random_state=random_state)  # shuffle
    # Convert to numpy arrays
    x_train, y_train = seg_features_to_numpy(train_segs, feature_cols)
    return x_train, y_train


def seg_features_to_numpy(partial_segs: DataFrame, feature_cols: list[str]):
    x = partial_segs.loc[:, feature_cols].to_numpy()
    y = (partial_segs['type'] == 'preictal').to_numpy(dtype=np.int32)
    return x, y


def load_features_and_labels(esegs: DataFrame, train_test_split_idx: int, feature_cols: list[str],
                             random_state: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the features and labels for the training and test sets for the patient as numpy array.
    :param esegs: All existing segments of the patient
    :param random_state: random_state parameter for pd.DataFrame.sample()
    :return: x_train, y_train, x_test, y_test
    """
    # Split into train and test
    train_segs = esegs.loc[:train_test_split_idx - 1]
    test_segs = esegs.loc[train_test_split_idx:]

    x_train, y_train = subsample_shuffle_convert_train_segs(train_segs, feature_cols, random_state)
    x_test, y_test = seg_features_to_numpy(test_segs, feature_cols)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    pdir = PATHS.patient_dirs()[0]
    segs_ = pd.read_pickle(pickle_path(pdir.segments_table))
