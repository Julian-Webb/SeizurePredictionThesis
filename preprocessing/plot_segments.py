from __future__ import annotations

import logging
import multiprocessing
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, Timestamp

from config import PatientDir, PATHS
from config.intervals import INTERICTAL, INTER_PRE, PREICTAL, INTERVENTION, POSTICTAL, INTER_POST


def plot_segs(
        segs: DataFrame,
        szrs: DataFrame,
        edfs: DataFrame = None,
        test_partition_start: Timestamp = None,
        title: str = None,
        figsize=(30, 8),
):
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

    # Plot test partition start
    if test_partition_start is not None:
        ax.axvline(test_partition_start, color='blue', linestyle='--', linewidth=2, label='test partition start',
                   ymin=-0.05, ymax=1.05, clip_on=False)

    # Plot edf times
    if edfs is not None:
        for edf in edfs.itertuples(index=False):
            ax.axvspan(edf.start_mtz, edf.end_mtz, color='green', alpha=0.2)

    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.legend(loc='upper left')

    return fig


def plot_segs_for_pdir(pdir: PatientDir):
    segs = pd.read_pickle(pdir.segments_table.pickle)
    szrs = pd.read_pickle(pdir.valid_szr_starts_file.pickle)
    edfs = pd.read_pickle(pdir.edf_files_table.pickle)
    test_start_mtz = pd.read_pickle(pdir.dataset_partition.pickle).loc['test', 'start_mtz']

    fig = plot_segs(segs, szrs, edfs, test_start_mtz, title=pdir.name)

    fig.savefig(pdir.segments_plot, bbox_inches='tight')
    plt.close(fig)


def plot_segs_for_pdirs(pdirs: List[PatientDir], serial_processing: bool = False):
    logging.info(f'🎬 Creating segments plots')
    if serial_processing:
        for pdir in pdirs:
            plot_segs_for_pdir(pdir)
    else:
        with multiprocessing.Pool() as p:
            p.map(plot_segs_for_pdir, pdirs)
    logging.info(f'✅ Created segments plots')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    pdirs = PATHS.patient_dirs()
    plot_segs_for_pdirs(pdirs, serial_processing=False)
