import logging
from pathlib import Path

import pandas as pd
from pandas import DataFrame


def combine_ptnt_annotations(pdir_src: Path, pdir_dst: Path, ptnt_files: DataFrame):
    # Add files together in order of visits
    relevant_types = ['Consensus', 'AllAutomaticDetections']
    for type in relevant_types:
        dfs_per_type = []
        for visit, visit_files in ptnt_files.groupby('visit'):
            if visit == 'V4' and type == 'AllAutomaticDetections':
                # Visit 4 doesn't have automatic detections
                continue

            match = visit_files.loc[visit_files['annotation_type'] == type]
            if len(match) != 1:
                logging.error(
                    f"Expected exactly 1 row for patient={pdir_src.name}, visit={visit}, annotation_type={type}, "
                    f"but found {len(match)}.")
                continue

            file_row = match.iloc[0]

            csv_path = pdir_src / (file_row['file_name'] + '.csv')
            df = pd.read_csv(csv_path)
            df.insert(df.columns.get_loc("end_naive") + 1, "visit", visit)
            df['visit'] = visit
            dfs_per_type.append(df)

        if dfs_per_type:
            szrs = pd.concat(dfs_per_type, ignore_index=True)

            save_path = pdir_dst / f'{pdir_src.name}_{type}.csv'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            szrs.to_csv(save_path, index=False)
        else:
            logging.warning(f"No annotations found for {pdir_src.name}, annotation_type={type}")


def combine_all_annotations(base: Path, fileindex_path: Path, src: Path, dst: Path):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    fileindex = pd.read_csv(fileindex_path)

    for pdir_src in sorted(src.iterdir()):
        combine_ptnt_annotations(pdir_src,
                                 pdir_dst=dst / pdir_src.name,
                                 ptnt_files=fileindex[fileindex['patient_id'] == pdir_src.name])



def main():
    base = Path('/data/home/webb/seizure_annotations')
    src = Path(base / 'STEP3_csv_anns')
    dst = Path(base / 'STEP4_combined_csv_anns')
    fileindex_path = base / 'STEP2_cleaned_anns/file_index.csv'

    combine_all_annotations(base, fileindex_path, src, dst)


if __name__ == '__main__': main()
