import multiprocessing
from typing import List

import pandas as pd
from pandas import Series, Timestamp, Timedelta, DataFrame

from config.constants import RATIO_OF_TIMESPAN_FOR_TRAINING
from config.paths import PATHS, PatientDir
from models.load_data import choose_interictal_train_segs
from utils.io import pickle_path, save_dataframe_multiformat


def _compute_ptnt_split(recordings_start: Timestamp, timespan: Timedelta, seg_starts: Series) -> dict:
    """
    Find the location which splits the data into train and test for a patient.
    The prior segments will be for training and the latter for testing.

    :param recordings_start: The start of the first recording of this patient
    :param timespan: The entire timespan between the start of the first recording and the end of the last.
    :param seg_starts: The starts of all this patient's segments
    :return: Timestamp and segment index of the end of training
    """
    timespan_train = timespan * RATIO_OF_TIMESPAN_FOR_TRAINING
    # noinspection PyTypeChecker
    train_end_approx: Timestamp = recordings_start + timespan_train

    # Make the train end correspond to a specific segment ("round it")
    # Finds the index where train_end_approx would be inserted to maintain order, i.e. which starts it's between.
    idx = int(seg_starts.searchsorted(train_end_approx))
    train_end_exact: Timestamp = seg_starts.iloc[idx]
    return {'Timestamp': train_end_exact, 'segment_index': idx}


def find_ptnt_split(ptnt_dir: PatientDir, all_ptnts_info: DataFrame):
    dataset = ptnt_dir.parent.name
    ptnt_info = all_ptnts_info.loc[(dataset, ptnt_dir.name)]
    segs = pd.read_pickle(pickle_path(ptnt_dir.segments_table))

    # noinspection PyTypeChecker
    train_end = _compute_ptnt_split(ptnt_info['recordings_start'], ptnt_info['timespan'], segs['start'])
    train_end = Series(train_end, name='train_end')
    save_dataframe_multiformat(train_end, ptnt_dir.train_test_split)


def find_ptnt_splits(ptnt_dirs: List[PatientDir]):
    all_ptnts_info = pd.read_pickle(pickle_path(PATHS.patient_info_exact))

    # Serial Processing
    # for ptnt_dir in ptnt_dirs:
    #     find_ptnt_split(ptnt_dir, all_ptnts_info)

    # Parallel Processing
    args = [(ptnt_dir, all_ptnts_info) for ptnt_dir in ptnt_dirs]
    with multiprocessing.Pool() as pool:
        pool.starmap(find_ptnt_split, args)


if __name__ == '__main__':
    find_ptnt_splits(PATHS.patient_dirs())

