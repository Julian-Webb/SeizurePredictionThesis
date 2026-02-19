import logging
import re
import shutil
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from utils.utils import clean_mac_files

PATIENT_ID_LINE_RE = re.compile(
    r'^(?P<label>patient id|patienten-id)\s*:\s*(?P<id>U002-DE01-\d{2})\s*$',
    flags=re.IGNORECASE,
)


def get_annotation_lines_from_txt(path: Path, ptnt_id: str):
    """Preprocess an annotation file to get just the clean lines containing seizures."""
    lines = path.read_text().splitlines()

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


SZR_END = 'Seizure_End'
SZR_START = 'Seizure_Start'
USER_SZR_MARKER = 'User seizure marker'
SZR_ANN_TYPES = {
    'Pattern-Rhythmic': ['Pattern-Rhythmic-Theta', 'Pattern-Rhythmic-Delta-SW'],
    'Seizure': ['Seizure-rhythmic', 'Seizure-rhythmic +', 'Seizure-tonic'],
    'Button': ['BUTTON PRESSED', 'BUTTON DOUBLE PRESSED'],
    'Other': ['EDF Annotation', 'Undefined'],
    'User seizure marker': [USER_SZR_MARKER],
    'Seizure_boundary': [SZR_END, SZR_START],
}


def interpret_line(line: str, filename: str = ''):
    """Interpret a line from a text annotation file"""
    values = line.split('\t')
    # If it has 4 values, the last one is a comment
    comment = values.pop() if len(values) == 4 else ''
    type_, datetime1, datetime2 = values

    # Check Pattern-Rhythmic (different start and end dates)
    if type_ in SZR_ANN_TYPES["Pattern-Rhythmic"]:
        assert datetime1 != datetime2, f"Pattern Rhythmic, but the dates are the same:\n{datetime1}\n{datetime2}\n{filename=}"
        return {'type': type_, 'start_naive': datetime1, 'end_naive': datetime2, 'comment': comment}
    # Other types have a duplicated date
    elif any(type_ in SZR_ANN_TYPES[k] for k in
             ('Seizure', 'Button', 'Other', 'User seizure marker', 'Seizure_boundary')):
        assert datetime1 == datetime2, f"The dates are not the same:\n{datetime1}\n{datetime2}\n{filename=}"
        # Check for start and end with a User seizure marker
        if type_ == USER_SZR_MARKER:
            assert comment in SZR_ANN_TYPES['Seizure_boundary'], f"{type_=} but comment not in [{SZR_START}, {SZR_END}]"
            type_ = comment
        return {'type': type_, 'start_naive': datetime1, 'comment': comment}
    else:
        raise ValueError(f"Unknown type: {type_}\n{filename=}")


def txt_to_dataframe(path: Path, ptnt_id: str):
    lines = get_annotation_lines_from_txt(path, ptnt_id)

    i = 0
    szrs = []
    while i < len(lines):
        vals = interpret_line(lines[i])  # line values
        # Check if line should be ignored
        if any(vals['type'] in SZR_ANN_TYPES[k] for k in ('Button', 'Other')):
            logging.info(f"[{path.name}] Ignoring line: {lines[i]}")
        elif any(vals['type'] in SZR_ANN_TYPES[k] for k in ('Pattern-Rhythmic', 'Seizure')):
            szrs.append(vals)
        elif vals['type'] in SZR_ANN_TYPES["Seizure_boundary"]:
            # Look ahead to find the end
            szr = {'type': '', 'start_naive': vals['start_naive'], 'end_naive': None, 'comment': ''}

            original_i = i
            while szr['end_naive'] is None:
                i += 1

                # Check if we've reached the maximum lookahead
                if (i - original_i) > 2:
                    raise ValueError(f"Seizure boundary with no end found around line {i} for {path}")

                vals = interpret_line(lines[i])
                if vals['type'] in SZR_ANN_TYPES['Seizure']:
                    # If there's a seizure marker in between the start and end, we use its type
                    if not 'AllAutomaticDetections' in path.name:
                        logging.error(f"Wrapped seizure marker in non-AllAutomaticDetections file: {path}")
                        # raise ValueError(f"Wrapped seizure marker in non-AllAutomaticDetections file: {path}")

                    logging.info(f"Wrapped seizure marker found. Using its datetime as start. In {path}")
                    szr['start_naive'] = vals['start_naive']
                    if szr['type'] != '':
                        logging.warning(f'Multiple seizure types found around line {i} for {path}')
                    szr['type'] = vals['type']
                    szr['comment'] += ', ' + vals['comment']

                elif vals['type'] == SZR_END:
                    szr['end_naive'] = vals['start_naive']
                else:
                    raise ValueError(
                        f"Unexpected type in between seizure with start and end: {vals['type']}\n{lines[i]}")

            szrs.append(szr)
        else:
            raise ValueError(f"Unknown type: {vals['type']}\n{lines[i]}")

        i += 1  # look at the next line

    szrs_df = DataFrame(szrs, columns=['type', 'start_naive', 'end_naive', 'comment'])
    return szrs_df


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    root = Path('/data/home/webb/seizure_annotations/STEP2_cleaned_anns')
    clean_mac_files(root)
    dst = Path('/data/home/webb/seizure_annotations/STEP3_csv_anns')
    if dst.exists():  shutil.rmtree(dst)

    for pdir in sorted(list((root / 'data').iterdir())):
        txt_ann_paths = list(pdir.glob('*.txt'))
        for txt_path in txt_ann_paths:
            if 'SeizureStartEnd' in txt_path.stem:
                logging.debug(f'Skipping SeizureStartEnd file: {txt_path.name}')
                continue

            try:
                anns_df = txt_to_dataframe(txt_path, pdir.name)
                save_path = (dst / pdir.name / txt_path.stem).with_suffix('.csv')
                save_path.parent.mkdir(exist_ok=True, parents=True)
                anns_df.to_csv(save_path, index=False)
            except Exception as e:
                print(txt_path.name)
                print(e)
                print()


if __name__ == '__main__': main()
