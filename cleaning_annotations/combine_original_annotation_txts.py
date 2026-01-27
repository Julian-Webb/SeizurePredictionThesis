import logging
import re
import shutil
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from cleaning_edfs.file_correction import clean_mac_files
from config.paths import PATHS, Dataset

PATIENT_ID_LINE_RE = re.compile(
    r'^(?P<label>patient id|patienten-id)\s*:\s*(?P<id>U002-DE01-\d{2})\s*$',
    flags=re.IGNORECASE,
)


# def combine_original_annotation_csvs(root, dst):
#     ptnts_with_pattern = set()  # Pattern_Rhythmic-Theta, etc.
#     pdirs = sorted(list(root.iterdir()))
#
#     for pdir in pdirs:
#         dfs = {'Consensus': DataFrame(), 'AllAutomaticDetections': DataFrame()}
#         for ann_path in (pdir / 'annotation_csv_files').iterdir():
#             for key, df in dfs.items():
#                 if key in ann_path.name:
#                     anns = pd.read_csv(ann_path, parse_dates=['start', 'single_marker', 'end'])
#                     dfs[key] = pd.concat([df, anns], ignore_index=True)
#                 else:
#                     logging.debug(f'Ignoring {ann_path.name}')
#
#         dst_pdir = dst / pdir.name
#         dst_pdir.mkdir(exist_ok=True)
#
#         for key, df in dfs.items():
#             if not df.empty and df['type'].str.contains('pattern', case=False, na=False).any():
#                 ptnts_with_pattern.add(pdir.name)
#             df.to_csv(dst_pdir / f'{pdir.name}_{key}.csv', index=False)
#
#     logging.warning(f'Pattern detected in {sorted(ptnts_with_pattern)}')


def get_szr_lines_from_txt(path: Path, ptnt_id: str):
    """Preprocess an annotation file to get just the clean lines containing seizures."""
    with (open(path, 'r') as file):
        # filter out empty lines and lines with only whitespace
        lines = [line.strip() for line in file.readlines() if line.strip()]
    # sometimes, there are multiple tabs or spaces after the seizure type. We replace this with a single tab
    lines = [re.sub(r'\s{2,}', r'\t', line) for line in lines]

    # assert that the file starts with the patient ID and has correct patient ID
    m = re.fullmatch(PATIENT_ID_LINE_RE, lines[0])
    if not m:
        logging.error(f"The following annotation's line 0 doesn't have correct format:\n{lines[0]}\n{path.name}\n")
    found_id = m.group('id')
    if not found_id == ptnt_id:
        logging.error(f"Annotation {path.name} has patient ID {found_id}, instead of {ptnt_id}.")

    lines.pop(0)  # remove the first line

    # Remove lines that are "no seizure"
    lines = [line for line in lines if not line.lower().startswith('no seizure')]

    return lines


def combine_original_annotation_txts(original_anns_dir: Path, dst: Path, fileindex: DataFrame):
    """

    :param fileindex: The index of all cleaned annotation files, sorted by patient and visit
    """

    for pdir in sorted(list(original_anns_dir.iterdir())):
        ptnt_files = fileindex[fileindex['patient_id'] == pdir.name]
        # Read txt files
        szr_lines = {'Consensus': [], 'AllAutomaticDetections': []}

        # Add files together in order of visits
        for visit, visit_files in ptnt_files.groupby('visit'):
            for type, lines in szr_lines.items():
                if visit == 'V4' and type == 'AllAutomaticDetections':
                    # Visit 4 doesn't have automatic detections
                    continue

                match = visit_files.loc[visit_files['annotation_type'] == type]
                if len(match) != 1:
                    logging.error(
                        f"Expected exactly 1 row for patient={pdir.name}, visit={visit}, annotation_type={type}, "
                        f"but found {len(match)}.")
                    continue

                file_row = match.iloc[0]
                filepath = original_anns_dir / pdir.name / (file_row['file_name'] + '.txt')
                lines.extend(get_szr_lines_from_txt(filepath, pdir.name))

        # Write txt files
        dst_pdir = dst / pdir.name
        dst_pdir.mkdir(exist_ok=True)
        for key, lines in szr_lines.items():
            with open(dst_pdir / f'{pdir.name}_{key}.txt', 'w') as f:
                f.write("\n".join(lines))


def delete_csv_files(dst: Path):
    for pdir in dst.iterdir():
        for csv_file in pdir.glob('*.csv'):
            csv_file.unlink()


def copy_txts_to_dataset(new_anns_path: Path):
    for pdir in PATHS.patient_dirs([Dataset.uniclinic]):
        src = new_anns_path / pdir.name
        dst = pdir.szr_anns_original_dir
        dst.mkdir(exist_ok=True)
        for f in src.glob('*.txt'):
            shutil.copy2(f, dst)


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    root = Path('/data/home/webb/seizure_annotations/STEP2_cleaned_anns/')
    clean_mac_files(root)
    dst = Path('/data/home/webb/seizure_annotations/STEP3_combined_anns')
    dst.mkdir(exist_ok=True)

    fileindex = pd.read_csv(root / 'file_index.csv')
    combine_original_annotation_txts(root / 'data', dst, fileindex)
    # delete_csv_files(dst)
    copy_txts_to_dataset(dst)


if __name__ == '__main__': main()
