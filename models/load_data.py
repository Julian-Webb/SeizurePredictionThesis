import numpy as np
import pandas as pd
from pandas import DataFrame
from pyedflib import EdfReader

from config.constants import MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO, N_CHANNELS
from config.intervals import SEGMENT
from config.paths import PatientDir, PATHS
from feature_extraction.extract_features import Features
from utils.edf_utils import load_segmented_sigs
from utils.io import pickle_path


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
    subsampled_interictal = esegs[interictal_mask].sample(n, random_state=random_state)

    # Combine subsampled interictal and preictal - only use these types
    preictal = esegs[esegs['type'] == 'preictal']
    subsampled = pd.concat([subsampled_interictal, preictal])

    # Shuffle
    shuffled = subsampled.sample(frac=1, random_state=random_state)

    return shuffled


def seg_features_to_numpy(partial_segs: DataFrame, feature_cols: list[str]):
    x = partial_segs.loc[:, feature_cols].to_numpy()
    y = (partial_segs['type'] == 'preictal').to_numpy(dtype=np.int32)
    return x, y

# noinspection PyUnboundLocalVariable
def load_data(
        pdir: PatientDir,
        type_: str,
        subsample_shuffle_and_subselect_types: bool,
        train: bool = False,
        test: bool = False,
        random_state: int = None
):
    """
    Load features and/or EEG data for a patient. Specify whether to include training and test data.
    :param pdir:
    :param type_: either 'features' or 'eeg'
    :param subsample_shuffle_and_subselect_types: Apply operations that are used for model training:
        * only use interictal and preictal segments, exclude other types.
        * subsample interictal segments, taking into consideration the MAX_INTERICTAL_TO_PREICTAL_SEGMENT_RATIO
        * shuffle segments
    :param train: Whether to load training data
    :param test: Whether to load testing data
    :param random_state: Optional random state for subsampling and shuffling

    :return: x, y. (data, labels)
    """
    # Check arguments
    if not (train or test): raise ValueError('Specify either train, test, or both')
    if type_ not in ['features', 'eeg']:
        raise ValueError(f'Parameter type_ must be either "features" or "eeg", but is {type_}')

    # Load segments
    segs = pd.read_pickle(pickle_path(pdir.segments_table))

    # Select train / test data or keep both, depending on args
    if train or test:
        split_idx = pd.read_pickle(pickle_path(pdir.train_test_split)).segment_index
        if split_idx not in segs.index and not segs.index.is_monotonic_increasing:
            raise RuntimeError("split_idx does not correspond to segment index; check train_test_split contents")

    if train and not test:
        segs = segs[segs.index < split_idx]
    elif test and not train:
        segs = segs[segs.index >= split_idx]

    # Select only existing segs for further processing
    segs = segs[segs['exists']].drop(columns=['exists']).reset_index(drop=True)

    if subsample_shuffle_and_subselect_types:
        # noinspection PyTypeChecker
        segs = subsample_shuffle_and_subselect_types_for_segs(segs, random_state)

    if type_ == 'features':
        return seg_features_to_numpy(segs, Features.ORDERED_NAMES)
    else:  # EEG
        segs.drop(columns=Features.ORDERED_NAMES, inplace=True)

        # Reset the index so that it corresponds to the index in the array x
        segs.reset_index(drop=True, inplace=True)
        x = np.empty([len(segs), SEGMENT.n_samples, N_CHANNELS])

        # load data depending on whether we shuffled (for efficiency)
        if subsample_shuffle_and_subselect_types:
            # Load data by file for efficiency
            for file_name, file_segs in segs.groupby('file'):
                with EdfReader(str(pdir.edf_dir / file_name)) as edf:
                    for seg in file_segs.itertuples():
                        for chn in range(N_CHANNELS):
                            # seg.Index: index in segs DataFrame
                            # seg.start_index: index within the EDF file
                            x[seg.Index, :, chn] = edf.readSignal(chn, seg.start_index, SEGMENT.n_samples)
        else:
            for file_name, file_segs in segs.groupby('file'):
                x[file_segs.index, :, :] = load_segmented_sigs(
                    file_path=pdir.edf_dir / file_name,
                    first_idx=file_segs.iloc[0]['start_index'],
                    n_segs=len(file_segs),
                    channels_last=True
                )

        y = (segs['type'] == 'preictal').to_numpy(dtype=np.int32)
        return x, y


if __name__ == '__main__':
    pdir = PATHS.patient_dirs()[7]
    load_data(
        pdir,
        type_='eeg',
        train=True,
        test=False,
        subsample_shuffle_and_subselect_types=True,
    )
