import logging
import shutil

import model_eval.calc_segment_probabilities
import model_eval.clips
import model_eval.eval_models
import model_eval.event_based_metrics
from cleaning_annotations.localize_annotations import drop_duplicates_and_localize
from config import Paths
from config import PatientDir, PATHS
from feature_extraction.extract_features import run_feature_extraction
from preprocessing.filter_signals import filter_all_edfs
from preprocessing.segment_tables import make_segment_tables
from preprocessing.train_test_allocation import find_ptnt_splits
from preprocessing.validate_patients import validate_patients


def remove_unnecessary_files():
    root = Paths('/data/home/webb/d/UNEEG_base')
    for pdir in root.patient_dirs():
        dirs_to_del = [
            pdir.cycle_extraction_dir,
            pdir.model_eval_dir,
            pdir.models_dir,
            pdir.predictions_dir,
        ]
        multipath_files_to_del = [
            pdir.valid_edf_intervals,
            pdir.invalid_edf_intervals,
            pdir.segments_table,
            pdir.train_test_split,
            pdir.all_szr_starts_file,
            pdir.valid_szr_starts_file,
        ]
        files_to_del = [pdir.segments_plot]

        for d in dirs_to_del:
            # print(d)
            try:
                shutil.rmtree(d)
            except FileNotFoundError:
                pass

        for f in multipath_files_to_del:
            # print(f)
            f.csv.unlink(missing_ok=True)
            f.pickle.unlink(missing_ok=True)

        for f in files_to_del:
            f.unlink(missing_ok=True)


def process_ptnt(pdir: PatientDir):
    logging.basicConfig(level=logging.INFO, format=f'[%(levelname)s] - %(message)s')
    pdirs = [pdir]
    drop_duplicates_and_localize(pdirs)
    validate_patients(PATHS.patient_dirs(include_invalid_ptnts=True), move_invalid_pdirs=False)
    # filter_all_edfs(pdirs)
    make_segment_tables(pdirs)
    find_ptnt_splits(pdirs)
    run_feature_extraction(pdirs)
    # model_eval.calc_segment_probabilities.main(pdirs, serial_processing=False)
    model_eval.clips.main(pdirs)
    model_eval.event_based_metrics.calc_metrics(pdirs)
    model_eval.eval_models.main(pdirs)


if __name__ == '__main__':
    # remove_unnecessary_files()
    pdir = PATHS.patient_dirs()[6]
    print(pdir)
    process_ptnt(pdir)
    pass
