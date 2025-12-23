from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pyedflib import EdfReader

from config.constants import MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO, N_CHANNELS
from config.intervals import SEGMENT
from config.paths import PATHS
from feature_extraction.extract_features import Features
from utils.edf_utils import load_segmented_sigs
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


def _split_train_test_data_and_process(segs: DataFrame, train_test_split: Series, random_state: int = None):
    """
    Drop non-existing segs. Split segments for training and testing. Select a subset of interictal training segs.
    Combine the preictal and interictal segs and shuffle. Segs that aren't of type preictal or interictal are dropped.
    :return: train_segs, test_segs
    """
    segs = segs[segs['exists']]
    # Split segs into train and test
    split_idx = train_test_split.segment_index
    train_segs = segs.loc[:split_idx - 1]
    test_segs = segs.loc[split_idx:]

    # Filter interictal segs. Then, combine preictal and interictal segs and shuffle
    interictal = choose_interictal_train_segs(train_segs, random_state)
    preictal = train_segs[train_segs['type'] == 'preictal']
    train_segs = pd.concat([preictal, interictal])
    train_segs = train_segs.sample(frac=1, random_state=random_state)  # shuffle

    return train_segs, test_segs


def load_features_and_labels(segs: DataFrame, train_test_split: Series, feature_cols: list[str],
                             random_state: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the features and labels for the training and test sets for the patient as numpy arrays.
    :param random_state: random_state parameter for pd.DataFrame.sample()
    :return: x_train, y_train, x_test, y_test
    """
    train_segs, test_segs = _split_train_test_data_and_process(segs, train_test_split, random_state)

    # Extract features as numpy arrays
    def seg_features_to_numpy(partial_segs: DataFrame, feature_cols: list[str]):
        x = partial_segs.loc[:, feature_cols].to_numpy()
        y = (partial_segs['type'] == 'preictal').to_numpy(dtype=np.int32)
        return x, y

    x_train, y_train = seg_features_to_numpy(train_segs, feature_cols)
    x_test, y_test = seg_features_to_numpy(test_segs, feature_cols)
    return x_train, y_train, x_test, y_test


def _load_eeg_x_train(train_segs: DataFrame, edf_dir: Path) -> np.ndarray:
    """
    :param train_segs: with reset index
    :return: x_train
    """
    x_train = np.empty([len(train_segs), N_CHANNELS, SEGMENT.n_samples])
    # Load data by file for efficiency
    for file_name, file_segs in train_segs.groupby('file'):
        with EdfReader(str(edf_dir / file_name)) as edf:
            for seg_i, seg in file_segs.iterrows():
                for chn in range(N_CHANNELS):
                    # logging.debug(f"{seg_i}, {chn}, {seg.start_index}, {file_name}")
                    x_train[seg_i, chn, :] = edf.readSignal(chn, seg.start_index, SEGMENT.n_samples)
    return x_train


def _load_eeg_x_test(test_segs: DataFrame, edf_dir: Path) -> np.ndarray:
    """
    :param test_segs: Sorted by start. With reset index
    :return: x_test
    """
    x_test = np.empty([len(test_segs), N_CHANNELS, SEGMENT.n_samples])
    # Load test segs by file for efficiency
    for file_name, file_segs in test_segs.groupby('file'):
        x_test[file_segs.index, :, :] = load_segmented_sigs(file_path=edf_dir / file_name,
                                                            first_idx=file_segs.iloc[0]['start_index'],
                                                            n_segs=len(file_segs))
    return x_test


def load_eeg_data(segs: DataFrame, train_test_split: Series, edf_dir: Path, random_state: int = None) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split segments for training and testing. Select a subset of interictal training segs. Return data as numpy arrays.
    :param segs:
    :param train_test_split:
    :param random_state:  random_state parameter for pd.DataFrame.sample()
    :return: x_train, y_train, x_test, y_test
    """
    # todo how will I need to load it to evaluate it later on?
    train_segs, test_segs = _split_train_test_data_and_process(segs, train_test_split, random_state)

    for df in [train_segs, test_segs]:
        # Drop features (not needed here)
        df.drop(columns=list(Features.ORDERED_NAMES), inplace=True)
        # Reset the index so that it corresponds to the index in the array
        df.reset_index(inplace=True, drop=True)

    x_train = _load_eeg_x_train(train_segs, edf_dir)
    x_test = _load_eeg_x_test(test_segs, edf_dir)
    y_train = (train_segs['type'] == 'preictal').to_numpy(dtype=np.int32)
    y_test = (test_segs['type'] == 'preictal').to_numpy(dtype=np.int32)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    pdir = PATHS.patient_dirs()[0]
    segs_ = pd.read_pickle(pickle_path(pdir.segments_table))
    split_ = pd.read_pickle(pickle_path(pdir.train_test_split))
    load_eeg_data(segs_, split_, pdir.edf_dir, random_state=42)
