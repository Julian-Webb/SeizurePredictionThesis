import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from config.constants import MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO
from config.paths import PATHS
from utils.io import pickle_path


def choose_interictal_train_segs(train_segs: DataFrame, random_state: int = None) -> DataFrame:
    """
    Randomly choose a subset of the interictal segments to be used for training, taking into consideration the
    MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO
    :param train_segs: All segments used for training
    :return: A subset of the interictal segs in random order. Same format as input DataFrame.
    """
    # Note: index stays the same -> still works for original DataFrame
    ts = train_segs  # alias
    n_preictal = ts[ts['type'] == 'preictal'].shape[0]
    max_interictal = n_preictal * MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO

    interictal_mask = ts['type'] == 'interictal'
    n_interictal = interictal_mask.sum()
    # In case there are less interictal segments than max_interictal, use all:
    n = min(n_interictal, max_interictal)

    selected_interictal = ts[interictal_mask].sample(n, random_state=random_state)
    return selected_interictal


def process_train_segs(train_segs: DataFrame, random_state: int = None):
    """
    Convert the train_segs DataFrame into shuffled numpy arrays for training.
    Segs that aren't of type preictal or interictal are dropped.
    :return: x_train, y_train
    """
    interictal = choose_interictal_train_segs(train_segs, random_state)
    preictal = train_segs[train_segs['type'] == 'preictal']
    train_segs = pd.concat([preictal, interictal])
    train_segs = train_segs.sample(frac=1, random_state=random_state)  # shuffle
    return train_segs


def _seg_features_to_numpy(partial_segs: DataFrame, feature_cols: list[str]):
    x = partial_segs.loc[:, feature_cols].to_numpy()
    y = (partial_segs['type'] == 'preictal').to_numpy(dtype=np.int32)
    return x, y


def load_features_and_labels(segs: DataFrame, train_test_split: Series, feature_cols: list[str],
                             random_state: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the features and labels for the training and test sets for the patient as numpy array.
    :param random_state: random_state parameter for pd.DataFrame.sample()
    :return: x_train, y_train, x_test, y_test
    """
    # Load values
    segs = segs[segs['exists']]
    split_idx = train_test_split.segment_index

    train_segs = segs.loc[:split_idx - 1]
    test_segs = segs.loc[split_idx:]

    # noinspection PyTypeChecker
    train_segs = process_train_segs(train_segs, random_state)
    x_train, y_train = _seg_features_to_numpy(train_segs, feature_cols)
    x_test, y_test = _seg_features_to_numpy(test_segs, feature_cols)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    pdir = PATHS.patient_dirs()[0]
    segs_ = pd.read_pickle(pickle_path(pdir.segments_table))
    choose_interictal_train_segs(segs_)
