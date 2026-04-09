from __future__ import annotations

import logging
import math
import multiprocessing
from typing import Tuple, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Timestamp, Timedelta, Series

from config import PatientDir, PATHS, save_dataframe_multiformat
from config.constants import SAMPLING_FREQUENCY_HZ
from config.intervals import SEGMENT, INTERVENTION, PREICTAL, INTER_PRE, POSTICTAL, INTER_POST, INTERICTAL, SPH
from utils.edf_utils import time_to_index


# noinspection PyUnresolvedReferences
def find_existing_segs(valid_edf_intervals: DataFrame, edfs: DataFrame, segs: DataFrame) -> DataFrame:
    """In the segs DataFrame, fill in which segments exist in the valid EDF data."""
    # Loop through each interval and set the segs it contains to True
    segs['exists'] = False

    for file_name, group in valid_edf_intervals.groupby('file_name'):
        file_start = edfs.loc[edfs['file_name'] == file_name, 'start_mtz'].item()
        for iv in group.itertuples():
            # We only want segments completely contained in the interval because we only want full segments
            iv_segs_mask = (iv.start_mtz <= segs['start_mtz']) & (segs['end_mtz'] <= iv.end_mtz)
            segs.loc[iv_segs_mask, 'exists'] = True
            segs.loc[iv_segs_mask, 'file'] = file_name

            # Calculate the start index based on the start of the file
            segs.loc[iv_segs_mask, 'start_index'] = segs.loc[iv_segs_mask, 'start_mtz'].apply(
                lambda seg_start: round(time_to_index(file_start, seg_start, SAMPLING_FREQUENCY_HZ)))

            # Since we converted from time to index, which is slightly messy, we assert that the distance between
            #  the start indexes is correct.
            index_diffs = segs.loc[iv_segs_mask, 'start_index'].diff()
            # Ignore the first start entry because there is no previous start and make sure the differences are correct
            assert (index_diffs.iloc[1:] == SEGMENT.n_samples).all(), \
                f'The differences between two starts indexes is not {SEGMENT.n_samples=}'

    return segs


def find_seg_type(segs: DataFrame, szrs: DataFrame) -> DataFrame:
    """Fill in the segment type for the segs DataFrame.
    :param segs: All the segments
    :param szrs: The valid seizures"""
    # Since we are working with valid seizures only, we can assume that starts are more than PREICTAL + INTERVENTION apart
    for i, szr in szrs.iterrows():
        pre_szr_ivs = [PREICTAL, INTERVENTION]
        if szr['lead']:
            # There will be an inter_pre only if this is a lead szr
            pre_szr_ivs = [INTER_PRE] + pre_szr_ivs
        ivs = pre_szr_ivs + [POSTICTAL, INTER_POST]

        # How much before the szr the first iv starts
        pre_szr_offset = sum([iv.exact_dur for iv in pre_szr_ivs], Timedelta(0))
        iv_start = szr['start_mtz'] - pre_szr_offset

        # Iterate through the intervals and set the properties of segs
        # NOTE: If the next szr is non-lead, the preictal and intervention interval will overlap with inter_post and
        #  possibly postictal. However, it will naturally be overwritten by the next seizure, since they are in order.
        for iv in ivs:
            iv_end = iv_start + iv.exact_dur
            # Find segs in this interval:
            # base it on the end, since the SPH is predicted from the end of clips, which is the end of it's last seg
            in_iv_mask = (iv_start < segs['end_mtz']) & (segs['end_mtz'] <= iv_end)
            segs.loc[in_iv_mask, 'type'] = iv.label
            segs.loc[in_iv_mask, 'lead_szr'] = szr['lead']
            iv_start = iv_end
    segs['type'] = segs['type'].fillna(INTERICTAL.label)
    return segs


def check_szrs_in_sphs(sph_starts: np.ndarray, sph_ends: np.ndarray, szr_starts: np.ndarray):
    """Return a matrix that shows which SPHs contain which seizures (n_szrs, n_windows)"""
    szr_in_sph = (sph_starts <= szr_starts[:, None]) & (szr_starts[:, None] <= sph_ends)
    return szr_in_sph


def assert_window_type_matches_sph(
        win_ends: Series[pd.Timestamp],
        is_preictal: Series[bool],
        szr_starts: NDArray[np.datetime64],
        intervention_duration: Timedelta = INTERVENTION.exact_dur,
        sph_duration: Timedelta = SPH.exact_dur,
):
    sph_starts = win_ends + intervention_duration
    sph_ends = sph_starts + sph_duration
    sphs = DataFrame({'start': sph_starts, 'end': sph_ends}, index=win_ends.index)

    preict = sphs[is_preictal]
    nonpre = sphs[~is_preictal]

    if len(szr_starts) == 0:
        assert len(preict) == 0, 'Found preictal windows although there are no seizures.'

    preict_szr_in_sph = check_szrs_in_sphs(preict['start'].values, preict['end'].values, szr_starts)
    preict_sph_has_szr = preict_szr_in_sph.any(axis=0)  # Whether there's any seizure in each preictal window's SPH
    assert preict_sph_has_szr.all(), "Not all preictal window's SPH contains a seizure."

    nonpre_szr_in_sph = check_szrs_in_sphs(nonpre['start'].values, nonpre['end'].values, szr_starts)
    assert not nonpre_szr_in_sph.any(), "There are seizures in non-preictal windows' SPH."


def create_segs_for_ptnt(
        first_recording_start: Timestamp,
        timespan: Timedelta,
        valid_edf_intervals: DataFrame,
        edfs: DataFrame,
        valid_szrs: DataFrame,
):
    # We floor here because we only want full segments
    n_segs = math.floor(timespan / SEGMENT.exact_dur)
    segs = DataFrame(index=np.arange(n_segs),
                     columns=['start_mtz', 'end_mtz', 'type', 'lead_szr', 'exists', 'file', 'start_index'])

    # The start is shifted by the duration of a segment per segment
    segs['start_mtz'] = first_recording_start + segs.index * SEGMENT.exact_dur
    segs['end_mtz'] = segs['start_mtz'] + SEGMENT.exact_dur
    segs = find_existing_segs(valid_edf_intervals, edfs, segs)
    segs = find_seg_type(segs, valid_szrs)

    assert_window_type_matches_sph(segs['end_mtz'], segs['type'] == PREICTAL.label, valid_szrs['start_mtz'].to_numpy())

    return segs


def load_ptnt_timespan_info(pdir: PatientDir) -> Tuple[Timestamp, Timestamp, Timedelta]:
    """
    :return: start of the recordings, end of the recordings, timespan
    """
    ptnts_info = pd.read_pickle(PATHS.patient_info_exact.pickle)
    dataset = pdir.parent.name
    ptnt = pdir.name
    ptnt_info = ptnts_info.loc[dataset, ptnt]
    # noinspection PyTypeChecker
    return ptnt_info['recordings_start'], ptnt_info['recordings_end'], ptnt_info['timespan']


def create_segs_for_pdir(pdir: PatientDir):
    logging.info(f'[{pdir.name}] 🎬 Creating segs table')
    first_recording_start, _, timespan = load_ptnt_timespan_info(pdir)
    segs = create_segs_for_ptnt(
        first_recording_start, timespan,
        pd.read_pickle(pdir.valid_edf_intervals.pickle),
        pd.read_pickle(pdir.edf_files_table.pickle),
        pd.read_pickle(pdir.valid_szr_starts_file.pickle)
    )
    save_dataframe_multiformat(segs.drop(columns=['end_mtz']), pdir.segments_table)
    logging.info(f'[{pdir.name}] ✅ Finished segs table')


def create_segs_for_pdirs(pdirs: List[PatientDir], serial_processing: bool = False):
    if serial_processing:
        for pdir in pdirs:
            create_segs_for_pdir(pdir)
    else:
        with multiprocessing.Pool() as p:
            p.map(create_segs_for_pdir, pdirs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    pdirs_ = PATHS.patient_dirs()
    create_segs_for_pdirs(pdirs_, serial_processing=False)
