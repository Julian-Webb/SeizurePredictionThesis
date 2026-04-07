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
            pdir.dataset_partition,
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



def view_patient_info():
    import pandas as pd
    pi = pd.read_pickle(PATHS.patient_info_exact.pickle)
    return


def rename_probability():
    import pandas as pd
    from config import PATHS, MultiPath, save_dataframe_multiformat

    pdirs = PATHS.patient_dirs()
    for pdir in pdirs:
        print(pdir.name)
        seg_scores = pd.read_pickle(pdir.segment_scores_table.pickle)
        seg_scores.rename(columns={'ensemble': 'ensemble_score', 'CNN': 'CNN_score'}, inplace=True)
        save_dataframe_multiformat(seg_scores, pdir.segment_scores_table)

        clip_scores = pd.read_pickle(pdir.clip_scores_table.pickle)
        clip_scores.rename(columns={'ensemble_probability': 'ensemble_score', 'CNN_probability': 'CNN_score'},
                           inplace=True)
        save_dataframe_multiformat(clip_scores, pdir.clip_scores_table)


if __name__ == '__main__':
    # remove_unnecessary_files()
    # view_patient_info()
    rename_probability()
    pdir = PATHS.patient_dirs()[6]
    print(pdir)
    pass
