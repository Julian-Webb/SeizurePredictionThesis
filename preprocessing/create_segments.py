import logging
import math
import multiprocessing
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Timestamp, Timedelta

from config.constants import SAMPLING_FREQUENCY_HZ
from config.intervals import SEGMENT, INTERVENTION, PREICTAL, INTER_PRE, POSTICTAL, INTER_POST, INTERICTAL
from config import PatientDir, PATHS, save_dataframe_multiformat
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
            # Find segs in this interval
            in_iv_mask = (iv_start <= segs['start_mtz']) & (segs['start_mtz'] < iv_end)
            segs.loc[in_iv_mask, 'type'] = iv.label
            segs.loc[in_iv_mask, 'lead_szr'] = szr['lead']
            iv_start = iv_end
    segs['type'] = segs['type'].fillna(INTERICTAL.label)
    return segs


def make_segs_for_ptnt(
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


def make_segs_for_pdir(pdir: PatientDir):
    first_recording_start, _, timespan = load_ptnt_timespan_info(pdir)

    return make_segs_for_ptnt(
        first_recording_start, timespan,
        pd.read_pickle(pdir.valid_edf_intervals.pickle),
        pd.read_pickle(pdir.edf_files_table.pickle),
        pd.read_pickle(pdir.valid_szr_starts_file.pickle)
    )


def plot_segs(segs: DataFrame,
              szrs: DataFrame,
              edfs: DataFrame = None,
              title: str = None,
              figsize=(30, 8),
              save_path: str = None,
              show: bool = True):
    types = [INTERICTAL.label, INTER_PRE.label, PREICTAL.label, INTERVENTION.label, POSTICTAL.label, INTER_POST.label]
    type_to_y = {t: i for i, t in enumerate(types)}

    y = segs['type'].map(type_to_y)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot seg types
    ax.set_yticks(np.arange(len(types)))
    ax.set_yticklabels(types)
    ax.set_xlabel('Time')
    ax.set_ylabel('Segment type')

    # If title provided, place it below the x-axis and reserve space
    if title:
        ax.set_title(title, y=-0.12)  # y in axes fraction; negative moves it below
        fig.subplots_adjust(bottom=0.18)  # make room at bottom so title is not clipped

    # Different types of segments' plotting properties
    type_props = [
        {'label': 'seg starts lead', 'color': 'blue', 'mask': segs['lead_szr'] == True},
        {'label': 'seg start non-lead', 'color': 'turquoise', 'mask': segs['lead_szr'] == False},
        {'label': 'seg starts interictal', 'color': 'grey', 'mask': segs['lead_szr'].isna()},
    ]

    for exists in [True, False]:
        for tp in type_props:
            marker = '>' if exists else 'x'
            mask = tp['mask'] & (segs['exists'] == exists)
            ax.scatter(segs.loc[mask, 'start_mtz'], y[mask], s=7, label=tp['label'], c=tp['color'], marker=marker)

    # Plot seizures
    for t in szrs['start_mtz']:
        ax.axvline(t, color='r', linestyle='--', linewidth=0.5)
        ax.annotate(t.strftime("%d.%m.%y %H:%M:%S"), xy=(t, 1.0),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, 4),  # offset in points (x,y)
                    textcoords='offset points',
                    rotation=90,
                    ha='center', va='bottom',
                    fontsize=7,
                    color='r',
                    clip_on=False)

    # Plot edf times
    if edfs is not None:
        for edf in edfs.itertuples(index=False):
            ax.axvspan(edf.start_mtz, edf.end_mtz, color='green', alpha=0.2)

    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.legend(loc='upper left')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def make_segs_and_plot_for_pdir(pdir: PatientDir, from_preexisting_segs: bool = False):
    # Make segs table and save it to csv
    logging.info(f"Processing {pdir.name}")
    if from_preexisting_segs:
        segs = pd.read_pickle(pdir.segments_table.pickle)
    else:
        segs = make_segs_for_pdir(pdir)
        save_dataframe_multiformat(segs.drop(columns=['end_mtz']), pdir.segments_table)

    # Make the plot
    szrs = pd.read_pickle(pdir.valid_szr_starts_file.pickle)
    edfs = pd.read_pickle(pdir.edf_files_table.pickle)

    plot_segs(segs, szrs, edfs, pdir.name, show=False, save_path=pdir.segments_plot)


def create_segs_for_pdirs(pdirs: List[PatientDir], serial_processing: bool = False):
    if serial_processing:
        for pdir in pdirs:
            make_segs_and_plot_for_pdir(pdir)
    else:
        max_workers = min(len(pdirs), multiprocessing.cpu_count())
        logging.info(f"Using {max_workers} max workers")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(make_segs_and_plot_for_pdir, pt): pt for pt in pdirs}
            for fut in as_completed(futures):
                pdir = futures[fut]
                try:
                    fut.result()
                    logging.info(f"Finished seg table for: {pdir.name}")
                except:
                    logging.warning(f"Failed seg table for: {pdir.name}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    create_segs_for_pdirs(PATHS.patient_dirs())

    # Just make plots
    # for pdir_ in PATHS.patient_dirs():
    #     make_segs_and_plot_for_pdir(pdir_, True)
