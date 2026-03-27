import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import Interval
from pyedflib import highlevel

import model_eval.calc_segment_probabilities
import model_eval.clips
import model_eval.eval_models
import model_eval.event_based_metrics
from cleaning_annotations.localize_annotations import drop_duplicates_and_localize
from config import PatientDir, PATHS, Dataset
from config.constants import SAMPLING_FREQUENCY_HZ, N_CHANNELS, CHANNELS
from feature_extraction.extract_features import run_feature_extraction
from preprocessing.filter_signals import filter_all_edfs
from preprocessing.segment_tables import make_segment_tables
from preprocessing.train_test_allocation import find_ptnt_splits
from preprocessing.validate_patients import validate_patients

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


def process_fake_ptnt(pdir: PatientDir):
    pdirs = [pdir]
    drop_duplicates_and_localize(pdirs)
    validate_patients(PATHS.patient_dirs(include_invalid_ptnts=True), move_invalid_pdirs=False)
    filter_all_edfs(pdirs)
    make_segment_tables(pdirs)
    find_ptnt_splits(pdirs)
    run_feature_extraction(pdirs)
    model_eval.calc_segment_probabilities.main(pdirs)
    model_eval.clips.main(pdirs)
    model_eval.event_based_metrics.calc_metrics(pdirs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    pdir = PatientDir(PATHS.competition_dir / 'competition-01-MINIFAKE', dataset=Dataset.competition)

    # process_fake_ptnt(pdir)
