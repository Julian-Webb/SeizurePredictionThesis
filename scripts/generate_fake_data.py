import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import Interval
from pyedflib import highlevel

from config import PatientDir, PATHS, Dataset
from config.constants import SAMPLING_FREQUENCY_HZ, N_CHANNELS, CHANNELS

PHYSICAL_MIN = -1374.21
PHYSICAL_MAX = 1373.54
DIGITAL_MIN = -2048
DIGITAL_MAX = 2047


def generate_edf(interval: Interval, file_path: Path, ptnt_name: str):
    duration = interval.right - interval.left
    n_samples = round(duration.total_seconds() * SAMPLING_FREQUENCY_HZ)
    signals = np.random.uniform(DIGITAL_MIN / 2, DIGITAL_MAX / 2, size=[N_CHANNELS, n_samples])

    signal_headers = highlevel.make_signal_headers(CHANNELS,
                                                   sample_frequency=SAMPLING_FREQUENCY_HZ,
                                                   physical_min=PHYSICAL_MIN, physical_max=PHYSICAL_MAX,
                                                   digital_min=DIGITAL_MIN, digital_max=DIGITAL_MAX,
                                                   dimension='uV',
                                                   )
    header = highlevel.make_header(patientname=ptnt_name)
    highlevel.write_edf(str(file_path), signals, signal_headers, header)


def generate_fake_ptnt_data(pdir: PatientDir):
    edf_files = pd.read_pickle(pdir.edf_files_table.pickle)
    pdir.edf_dir.mkdir(parents=True, exist_ok=True)
    for i, edf in edf_files.iterrows():
        interval = Interval(edf['start'], edf['end'])
        generate_edf(interval, pdir.edf_dir / edf['file_name'], pdir.name)
        print(f"\rFiles generated: {i} | {edf['file_name']}", end='')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    pdir = PatientDir(PATHS.competition_dir / 'competition-01-MINIFAKE', dataset=Dataset.competition)

    # process_fake_ptnt(pdir)
