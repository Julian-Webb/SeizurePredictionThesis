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
from config.intervals import SEGMENT, HORIZON, PREICTAL, INTER_PRE, POSTICTAL, INTER_POST, INTERICTAL
from config.paths import PatientDir, PATHS
from utils.edf_utils import time_to_index


# todo do this locally
# from pyvirtualdisplay import Display
# disp = Display(visible=False, size=(800, 600))
# disp.start()


def load_ptnt_timespan_info(ptnt_dir: PatientDir) -> Tuple[Timestamp, Timestamp, Timedelta]:
    """
    :return: start of the recordings, end of the recordings, timespan
    """
    ptnts_info = pd.read_pickle(PATHS.patient_info_exact_pkl)
    dataset = ptnt_dir.parent.name
    ptnt = ptnt_dir.name
    ptnt_info = ptnts_info.loc[dataset, ptnt]
    # noinspection PyTypeChecker
    return ptnt_info['recordings_start'], ptnt_info['recordings_end'], ptnt_info['timespan']


# noinspection PyUnresolvedReferences
def find_existing_segs(edf_files: DataFrame, segs: DataFrame) -> DataFrame:
    """In the segs DataFrame, fill in which segments exist in the edf data."""
    # Loop through each edf and set the segs it contains to True
    segs['exists'] = False
    for _, edf in edf_files.iterrows():
        # We only want segments completely contained in the interval, because we only want full segments
        edf_segs_mask = (edf['start'] <= segs['start']) & (segs['end'] <= edf['end'])
        segs.loc[edf_segs_mask, 'exists'] = True
        segs.loc[edf_segs_mask, 'file'] = edf['file_name']

        # Calculate the start index based on the start of the file
        segs.loc[edf_segs_mask, 'start_index'] = segs.loc[edf_segs_mask, 'start'].apply(
            lambda start: round(time_to_index(file_start=edf['start'], timestamp=start,
                                              sampling_freq_hz=SAMPLING_FREQUENCY_HZ)))
        # Since we converted from time to index, which is slightly messy, we assert that the distance between
        #  the start indexes is correct.
        index_diffs = segs.loc[edf_segs_mask, 'start_index'].diff()
        # Ignore the first start entry because there is no previous start and make sure the differences are correct
        assert (index_diffs.iloc[1:] == SEGMENT.n_samples).all(), \
            f'The differences between two starts indexes is not {SEGMENT.n_samples=}'
    return segs


def find_seg_type(segs: DataFrame, szrs: DataFrame) -> DataFrame:
    """Fill in the segment type for the segs DataFrame.
    :param segs: All the segments
    :param szrs: The valid seizures"""
    # Since we are working with valid seizures only, we can assume that starts are more than PREICTAL + HORIZON apart
    for i, szr in szrs.iterrows():
        pre_szr_ivs = [PREICTAL, HORIZON]
        if szr['lead']:
            # There will be an inter_pre only if this is a lead szr
            pre_szr_ivs = [INTER_PRE] + pre_szr_ivs
        ivs = pre_szr_ivs + [POSTICTAL, INTER_POST]

        # How much before the szr the first iv starts
        pre_szr_offset = sum([iv.exact_dur for iv in pre_szr_ivs], Timedelta(0))
        iv_start = szr['start'] - pre_szr_offset

        # Iterate through the intervals and set the properties of segs
        # NOTE: If the next szr is non-lead, the preictal interval and horizon will overlap with inter_post and
        #  possibly postictal. However, it will naturally be overwritten by the next seizure, since they are in order.
        for iv in ivs:
            iv_end = iv_start + iv.exact_dur
            # Find segs in this interval
            in_iv_mask = (iv_start <= segs['start']) & (segs['start'] < iv_end)
            segs.loc[in_iv_mask, 'type'] = iv.label
            segs.loc[in_iv_mask, 'lead_szr'] = szr['lead']
            iv_start = iv_end
    segs['type'] = segs['type'].fillna(INTERICTAL.label)
    return segs


def make_segs_table(ptnt_dir: PatientDir):
    edf_files = pd.read_csv(ptnt_dir.edf_files_sheet, parse_dates=['start', 'end'])
    first_start, last_end, timespan = load_ptnt_timespan_info(ptnt_dir)

    # We floor here because we only want full segments
    n_segs = math.floor(timespan / SEGMENT.exact_dur)

    segs = DataFrame(columns=['start', 'end', 'type', 'lead_szr', 'exists', 'file', 'start_index'], index=np.arange(n_segs))

    # The start is shifted by the duration of a segment per segment
    segs['start'] = first_start + segs.index * SEGMENT.exact_dur
    segs['end'] = segs['start'] + SEGMENT.exact_dur
    segs = find_existing_segs(edf_files, segs)
    valid_szrs = pd.read_csv(ptnt_dir.valid_szr_starts_file, usecols=['start', 'lead'], parse_dates=['start'])
    segs = find_seg_type(segs, valid_szrs)
    return segs


def plot_segs(segs: DataFrame, szrs: DataFrame, edfs: DataFrame = None, title: str = None, figsize=(30, 8),
              savepath: str = None,
              show: bool = True):
    types = [INTERICTAL.label, INTER_PRE.label, PREICTAL.label, HORIZON.label, POSTICTAL.label, INTER_POST.label]
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
            ax.scatter(segs.loc[mask, 'start'], y[mask], s=7, label=tp['label'], c=tp['color'], marker=marker)

    # Plot seizures
    for t in szrs['start']:
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
            ax.axvspan(edf.start, edf.end, color='green', alpha=0.2)

    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.legend(loc='upper left')

    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def make_segs_table_and_plot(ptnt_dir: PatientDir, from_preexisting_segs: bool = False):
    # Make segs table and save it to csv
    logging.info(f"Processing {ptnt_dir.name}")
    if from_preexisting_segs:
        segs = pd.read_csv(ptnt_dir.segments_table, parse_dates=['start'], dtype={'lead': 'boolean'})

    else:
        segs = make_segs_table(ptnt_dir)
        segs.drop(columns=['end']).to_csv(ptnt_dir.segments_table, index=False)

    # Make the plot
    szrs = pd.read_csv(ptnt_dir.valid_szr_starts_file, usecols=['start', 'lead'], parse_dates=['start'])
    edfs = pd.read_csv(ptnt_dir.edf_files_sheet, parse_dates=['start', 'end'])

    plot_segs(segs, szrs, edfs, ptnt_dir.name, show=False, savepath=ptnt_dir.segments_plot)


def segment_tables(ptnt_dirs: List[PatientDir]):
    # Serial Processing:
    # for ptnt_dir in ptnt_dirs:
    #     logging.info(f'Seg table for : {ptnt_dir.name}')
    #     make_segs_table(ptnt_dir)

    # Parallel Processing:
    max_workers = min(len(ptnt_dirs), multiprocessing.cpu_count())
    logging.info(f"Using {max_workers} max workers")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(make_segs_table_and_plot, pt): pt for pt in ptnt_dirs}
        for fut in as_completed(futures):
            ptnt_dir = futures[fut]
            try:
                fut.result()
                logging.info(f"Finished seg table for : {ptnt_dir.name}")
            except:
                logging.info(f"Failed seg table for : {ptnt_dir.name}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    segment_tables(PATHS.patient_dirs())

    # ptnt_dir = PatientDir('/Users/julian/Developer/SeizurePredictionData/20240201_UNEEG_ForMayo/ptnt1')
    # ptnt_dir = PatientDir('/data/home/webb/UNEEG_data/20240201_UNEEG_ForMayo/K37N36L4D')
    # ptnt_dir = PatientDir('/data/home/webb/UNEEG_data/20250217_UNEEG_Extended/F5TW95P3X')
    # make_segs_table_and_plot(ptnt_dir, False)

    # Just make plots
    # for ptnt_dir in PATHS.patient_dirs():
    #     make_segs_table_and_plot(ptnt_dir, True)
