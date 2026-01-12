from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from pyedflib import EdfReader

from config.constants import MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO, N_CHANNELS
from config.intervals import SEGMENT
from config.paths import PATHS
from feature_extraction.extract_features import Features
from utils.edf_utils import load_segmented_sigs
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


def subsample_shuffle_train_segs(train_segs: DataFrame, random_state: int = None) -> \
        DataFrame:
    """
    Subsample interictal segments to achieve an acceptable ratio between preictal and interictal segs.
    Shuffle the segments.
    Segs that aren't of type preictal or interictal are dropped.
    :param train_segs:
    :param random_state:
    :return: x_train, y_train
    """
    # Subsample
    selected_interictal = _subsample_interictal_train_segs(train_segs, random_state)
    # Shuffle
    preictal = train_segs[train_segs['type'] == 'preictal']
    train_segs = pd.concat([preictal, selected_interictal])
    train_segs = train_segs.sample(frac=1, random_state=random_state)  # shuffle
    return train_segs


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

    train_segs = subsample_shuffle_train_segs(train_segs, random_state)
    x_train, y_train = seg_features_to_numpy(train_segs, feature_cols)
    x_test, y_test = seg_features_to_numpy(test_segs, feature_cols)
    return x_train, y_train, x_test, y_test


def _load_eeg_x_train(train_segs: DataFrame, edf_dir: Path) -> np.ndarray:
    """
    :param train_segs: with reset index
    :return: x_train
    """
    x_train = np.empty([len(train_segs), SEGMENT.n_samples, N_CHANNELS])
    # Load data by file for efficiency
    for file_name, file_segs in train_segs.groupby('file'):
        with EdfReader(str(edf_dir / file_name)) as edf:
            for seg_i, seg in file_segs.iterrows():
                for chn in range(N_CHANNELS):
                    # logging.debug(f"{seg_i}, {chn}, {seg.start_index}, {file_name}")
                    x_train[seg_i, :, chn] = edf.readSignal(chn, seg.start_index, SEGMENT.n_samples)
    return x_train


def _load_eeg_x_test(test_segs: DataFrame, edf_dir: Path) -> np.ndarray:
    """
    :param test_segs: Sorted by start. With reset index
    :return: x_test
    """
    x_test = np.empty([len(test_segs), SEGMENT.n_samples, N_CHANNELS])
    # Load test segs by file for efficiency
    for file_name, file_segs in test_segs.groupby('file'):
        x_test[file_segs.index, :, :] = load_segmented_sigs(file_path=edf_dir / file_name,
                                                            first_idx=file_segs.iloc[0]['start_index'],
                                                            n_segs=len(file_segs),
                                                            channels_last=True)
    return x_test


def load_eeg_train_data(esegs: DataFrame, split_idx: int, edf_dir: Path, random_state: int = None):
    """
    Split segments for training and testing. Select a subset of interictal training segs. Return data as numpy arrays.
    :param esegs: existing segments
    :param split_idx: index of the train-test split
    :param random_state:  random_state parameter for pd.DataFrame.sample()
    :return: x_train, y_train
    """
    train_segs = esegs.loc[:split_idx - 1]
    train_segs = subsample_shuffle_train_segs(train_segs, random_state)
    # Reset the index so that it corresponds to the index in the array
    train_segs.reset_index(inplace=True, drop=True)
    # Drop features (not needed here)
    train_segs.drop(columns=list(Features.ORDERED_NAMES), inplace=True)
    x_train = _load_eeg_x_train(train_segs, edf_dir)
    y_train = (train_segs['type'] == 'preictal').to_numpy(dtype=np.int32)
    return x_train, y_train


def load_eeg_test_data(esegs: DataFrame, split_idx: int, edf_dir: Path):
    """
    Split segments for training and testing. Return data as numpy arrays.
    :param esegs: existing segments
    :param split_idx: index of the train-test split
    :return: x_test, y_test
    """
    test_segs = esegs.loc[split_idx:].copy()
    # Reset the index so that it corresponds to the index in the array
    test_segs.reset_index(inplace=True, drop=True)
    # Drop features (not needed here)
    test_segs.drop(columns=list(Features.ORDERED_NAMES), inplace=True)
    x_test = _load_eeg_x_test(test_segs, edf_dir)
    y_test = (test_segs['type'] == 'preictal').to_numpy(dtype=np.int32)
    return x_test, y_test


if __name__ == '__main__':
    pdir = PATHS.patient_dirs()[0]
    segs_ = pd.read_pickle(pickle_path(pdir.segments_table))
    esegs_ = segs_[segs_['exists']]
    split_idx_ = pd.read_pickle(pickle_path(pdir.train_test_split)).segment_index
    load_eeg_train_data(esegs_, split_idx_, pdir.edf_dir)
