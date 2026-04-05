import shutil

from config import PATHS
from config import Paths


def remove_unnecessary_files():
    root = Paths('/data/home/webb/d/UNEEG_base')
    for pdir in root.patient_dirs():
        dirs_to_del = [
            pdir.cycle_extraction_dir,
            pdir.model_eval_dir,
            pdir.models_dir,
            pdir.predicted_scores_dir,
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




if __name__ == '__main__':
    # remove_unnecessary_files()
    pdir = PATHS.patient_dirs()[6]
    print(pdir)
    pass
