import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Timestamp
from pyedflib import EdfReader

from config import PATHS
from config.constants import MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO, N_CHANNELS
from config.intervals import SEGMENT
from feature_extraction.extract_features import FeatureNames
from utils.edf_utils import load_segmented_sigs


def subsample_shuffle_and_subselect_types_for_segs(esegs: DataFrame, random_state: int = None) -> DataFrame:
    """
    Apply operations that are used for model training:
        * only use interictal and preictal segments, exclude other types.
        * subsample interictal segments, taking into consideration the MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO
        * shuffle segments
    :param esegs: existing segments
    :param random_state: random_state parameter for pd.DataFrame.sample()
    :return: processed segments
    """
    # Subsample interictal
    n_preictal = esegs[esegs['type'] == 'preictal'].shape[0]
    max_interictal = n_preictal * MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO
    # Note: index stays the same -> still works for original DataFrame
    interictal_mask = esegs['type'] == 'interictal'
    n_interictal = interictal_mask.sum()
    # In case there are less interictal segments than max_interictal, use all:
    n = min(n_interictal, max_interictal)
    subsampled_interictal = esegs[interictal_mask].sample(n, random_state=random_state, replace=False)

    # Combine subsampled interictal and preictal - only use these types
    preictal = esegs[esegs['type'] == 'preictal']
    subsampled = pd.concat([subsampled_interictal, preictal])

    # Shuffle
    shuffled = subsampled.sample(frac=1, random_state=random_state, replace=False)

    return shuffled


def seg_features_to_numpy(partial_segs: DataFrame, feature_cols: list[str]):
    x = partial_segs.loc[:, feature_cols].to_numpy()
    y = (partial_segs['type'] == 'preictal').to_numpy(dtype=np.int32)
    return x, y


def load_data(
        segs: DataFrame,
        type_: str,
        subsample_shuffle_and_subselect_types: bool,
        train: bool = False,
        test: bool = False,
        test_start_mtz: Timestamp = None,
        edf_dir: Path = None,
        feature_names: list[str] = FeatureNames.ALL_ORDERED,
        random_state: int = None,
):
    """
    Load features and/or EEG data for a patient. Specify whether to include training and test data.
    :param segs: all segments of this patient: ``pd.read_pickle(pdir.segments_table.pickle)``
    :param type_: either 'features' or 'eeg'
    :param subsample_shuffle_and_subselect_types: Apply operations that are used for model training:
        * only use interictal and preictal segments, exclude other types.
        * subsample interictal segments, taking into consideration the MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO
        * shuffle segments
    :param train: Whether to load training data
    :param test: Whether to load testing data
    :param test_start_mtz: The `Timestamp` where the test set starts. Only needed if train and test are not both True.
    ``pd.read_pickle(pdir.dataset_partition.pickle).loc['test', 'start_mtz']``
    :param edf_dir: The patient's edf_dir, if type_ is 'eeg'.
    :param feature_names: The names of the features to load, if type_ is 'features'.
    :param random_state: Optional random state for subsampling and shuffling

    :return: dict with keys x (data), y (labels), and index_and_start (DataFrame with segment index and start_mtz to
    associate x and y values with segments)
    """
    # Check arguments
    if not (train or test):
        raise ValueError('Specify either train, test, or both')
    if type_ not in ['features', 'eeg']:
        raise ValueError(f'Parameter type_ must be either "features" or "eeg", but is {type_}')
    if type_ == 'eeg' and edf_dir is None:
        raise ValueError('If type_ is "eeg", specify edf_dir')
    if train != test and test_start_mtz is None:  # train or test, but not both
        raise ValueError('If train and test are not both True, specify test_start_mtz')
    if not segs.index.is_monotonic_increasing:
        raise RuntimeError("segs index should be monotonic increasing, but is not.")

    segs = segs.copy()

    # Select train / test data or keep both, depending on args
    if train and not test:
        segs = segs[segs['start_mtz'] < test_start_mtz]
    elif test and not train:
        segs = segs[segs['start_mtz'] >= test_start_mtz]

    # Select only existing segs for further processing
    segs = segs[segs['exists']].drop(columns=['exists'])

    if subsample_shuffle_and_subselect_types:
        # noinspection PyTypeChecker
        segs = subsample_shuffle_and_subselect_types_for_segs(segs, random_state)

    # Save the index and start for all remaining segs in the correct order to be able to associate the array values with
    # segments
    segs_index_and_start = segs[['start_mtz']]

    if type_ == 'features':
        x, y = seg_features_to_numpy(segs, feature_names)
    else:  # EEG
        segs.drop(columns=FeatureNames.ALL_ORDERED, inplace=True)

        # Reset the index so that it corresponds to the index in the array x
        segs.reset_index(drop=True, inplace=True)
        x = np.empty([len(segs), SEGMENT.n_samples, N_CHANNELS])

        # load data depending on whether we shuffled (for efficiency)
        if subsample_shuffle_and_subselect_types:
            # Load data by file for efficiency
            for file_name, file_segs in segs.groupby('file'):
                with EdfReader(str(edf_dir / file_name)) as edf:
                    for seg in file_segs.itertuples():
                        for chn in range(N_CHANNELS):
                            # seg.Index: index in segs DataFrame
                            # seg.start_index: index within the EDF file
                            x[seg.Index, :, chn] = edf.readSignal(chn, seg.start_index, SEGMENT.n_samples)
        else:
            for file_name, file_segs in segs.groupby('file'):
                x[file_segs.index, :, :] = load_segmented_sigs(
                    file_path=edf_dir / file_name,
                    first_idx=file_segs.iloc[0]['start_index'],
                    n_segs=len(file_segs),
                    channels_last=True
                )

        y = (segs['type'] == 'preictal').to_numpy(dtype=np.int32)

    return {'x': x, 'y': y, 'index_and_start': segs_index_and_start}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    pdir = PATHS.patient_dirs()[0]
    res = load_data(
        segs=pd.read_pickle(pdir.segments_table.pickle),
        type_='features',
        train=True,
        test=False,
        subsample_shuffle_and_subselect_types=True,
        test_start_mtz=pd.read_pickle(pdir.dataset_partition.pickle).loc['test', 'start_mtz'],
        edf_dir=pdir.edf_dir,
    )
