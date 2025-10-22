# See data_handling.ipynb for detailed documentation
import logging
import re
from enum import Enum
from typing import Tuple

import pandas as pd
from pathlib import Path

from config import PATHS
from data_cleaning.file_correction import clean_mac_files
from utils.paths import Dataset

# The strings that indicate that there are no seizures in a file
NO_SEIZURES_STRINGS = ['no seizures', 'no seizure']

# the possible ways a line can start
LINE_STARTS_SINGLE_MARKER = ['Seizure-rhythmic', 'Seizure-rhythmic +', 'Seizure-tonic']
LINE_STARTS_SEIZURE_STARTS = ['Seizure_Start']
LINE_STARTS_SEIZURE_ENDS = ['Seizure_End']
USER_SEIZURE_MARKER_STR = 'User seizure marker'
LINE_STARTS_TO_IGNORE = ['BUTTON DOUBLE PRESSED', 'BUTTON PRESSED']
LINE_STARTS_ALL = LINE_STARTS_SINGLE_MARKER + LINE_STARTS_TO_IGNORE + LINE_STARTS_SEIZURE_STARTS + LINE_STARTS_SEIZURE_ENDS + [
    USER_SEIZURE_MARKER_STR]

# The possible ways the end of a line indicates that it's the start/end of a seizure
SEIZURE_START_STRINGS = ['seizure_start', 'seizure start', 'seizure-start']
SEIZURE_END_STRINGS = ['seizure_end', 'seizure end', 'seizure-end']
# How far we look ahead at most for an end after a seizure start. In between there should be single markers
START_END_MAXIMUM_LOOKAHEAD = 2


def get_seizure_lines_from_file(annotation_path: Path):
    """Preprocess an annotation file to get just the clean lines containing seizures."""
    with (open(annotation_path, 'r') as file):
        # filter out empty lines and lines with only whitespace
        lines = []
        for line in file.readlines():
            stripped_line = line.strip()
            if stripped_line:
                lines.append(stripped_line)

    # assert that the file starts with the patient ID
    assert lines[0].startswith('Patient ID') or lines[0].startswith(
        'Patienten-ID'), "Annotation files doesn't start with 'Patient ID' or 'Patienten-ID'"
    lines.pop(0)  # remove the first line
    return lines


def read_line(line: str):
    """Converts a line into the type, dates, and comment. Asserts that the dates are equal, and the type is valid.
    :returns: The type, datetime, and comment"""
    comment = ''
    # sometimes, there are multiple tabs or spaces after the seizure type. We replace this with a single tab
    line = re.sub(r'\s{2,}', r'\t', line)
    values = line.split('\t')
    if len(values) == 4:
        # it contains a comment
        comment = values.pop()
    szr_type, datetime1, datetime2 = values

    if szr_type not in LINE_STARTS_ALL:
        print(f'{line}')
        print(f'{szr_type=}')
        raise ValueError(f"The following line doesn't start with {LINE_STARTS_ALL}: {line}")
    if datetime1 != datetime2:
        raise ValueError(f"The dates are not the same. {datetime1=}  ,  {datetime2=}")

    # If it's a user marker, it should start with something like "Seizure_Start", but may contain more
    comment_parts = comment.split(', ', maxsplit=1)

    return szr_type, datetime1, comment, comment_parts


def add_comment(seizure: dict, comment: str):
    """Add a comment to a seizure annotation and insert a comma to separate comments."""
    if comment != '':
        if seizure['comment'] != '':
            seizure['comment'] += ', '
        seizure['comment'] += comment


def process_single_marker(line):
    """Converts a line with just a single marker into a seizure annotation."""
    szr_type, datetime, comment, _ = read_line(line)
    return {'type': szr_type, 'single_marker': datetime, 'comment': comment}


def process_start_end(lines_from_start: list[str]) -> Tuple[dict, int]:
    """Converts successive lines with a start, possible single markers, and a possible end into a seizure annotation.
        :param lines_from_start: The file lines where the first line is the start line of the seizure.
        :returns: The seizure annotation dict, and the number of lines that were processed.
    """
    seizure = {'type': None, 'start': None, 'single_markers': [], 'end': None, 'comment': ''}

    lines = lines_from_start  # alias
    line = lines[0]

    # make sure this is the start of a seizure
    boundary = detect_start_end(line)
    assert boundary == Boundary.START, f'This line should be the start of a seizure: {line}'

    szr_type_start, datetime, _, comment_parts = read_line(line)
    seizure['start'] = datetime

    # find the markers and end, if present. These values will only be added if an end is found.
    single_markers = []
    seizure_types = []
    szr_end = None
    line_idx = 1
    # continue searching until an end is found or the maximum lookahead is reached
    while (not szr_end) and (line_idx < len(lines)) and (line_idx - 1) < START_END_MAXIMUM_LOOKAHEAD:
        line = lines[line_idx]
        szr_type_i, datetime, _, comment_parts = read_line(line)
        # check for a single marker
        if szr_type_i in LINE_STARTS_SINGLE_MARKER:
            # Handle the single marker and seizure type. Take into account that there may be multiple single markers.
            single_markers.append(datetime)
            seizure_types.append(szr_type_i)
        # check for the end
        elif detect_start_end(line) == Boundary.END:
            # it's a seizure end
            assert szr_end is None, "Seizure end found after previous seizure end."
            szr_end = datetime
        else:
            raise ValueError(f"This line should be a single marker or seizure end: {line}")
        line_idx += 1

    # only use the single marker values if an end was found
    if szr_end is not None:
        seizure['end'] = szr_end

        if seizure_types:
            if len(set(seizure_types)) > 1:
                logging.warning(f'Multiple seizure types found for seizure: {seizure_types}')
            # additional seizure types are ignored
            seizure['type'] = seizure_types[0]

        if single_markers:
            seizure['single_marker'] = single_markers.pop(0)
            # handle multiple single markers
            if single_markers:
                # make sure there's only one more single marker
                assert len(single_markers) == 1, 'There are more than two single markers for this seizure.'
                logging.warning(
                    f"Multiple single markers found for seizure: {seizure['single_marker']} and {single_markers[0]}")
                seizure['comment'] += f'Additional single marker: |{single_markers[0]}|'

        lines_processed = line_idx
    else:
        seizure['type'] = szr_type_start
        lines_processed = 1

    return seizure, lines_processed


class Boundary(Enum):
    START = 'start'
    END = 'end'


def detect_start_end(line: str) -> Boundary | None:
    """Identify whether a line indicates the boundary (start or end) of a seizure.
    :returns: The type of the boundary ('start' or 'end'), or None if it's not a boundary."""
    szr_type, datetime, comments, comment_parts = read_line(line)
    if szr_type == USER_SEIZURE_MARKER_STR:
        if comment_parts[0].lower() in SEIZURE_START_STRINGS:
            return Boundary.START
        elif comment_parts[0].lower() in SEIZURE_END_STRINGS:
            return Boundary.END
        else:
            raise ValueError(
                f"The following line doesn't start with {USER_SEIZURE_MARKER_STR} but doesn't end with a marker for a seizure boundary: {line}")
    elif szr_type in LINE_STARTS_SEIZURE_STARTS:
        return Boundary.START
    elif szr_type in LINE_STARTS_SEIZURE_ENDS:
        return Boundary.END
    else:
        return None


def annotations_txt_to_dataframe(annotation_path: Path):
    # go through the lines (seizures) and store them in a dataframe
    lines = get_seizure_lines_from_file(annotation_path)
    seizures = pd.DataFrame(columns=['type', 'start', 'single_marker', 'end', 'comment'], dtype=str)

    # Check if it contains no seizures
    if not lines or lines[0].lower() in NO_SEIZURES_STRINGS:
        return seizures

    # convert the lines into seizure annotations
    i = 0
    while i < len(lines):
        line = lines[i]
        # single marker:
        if line.startswith(tuple(LINE_STARTS_SINGLE_MARKER)):
            seizure = process_single_marker(line)
            i += 1
        # line to ignore
        elif line.startswith(tuple(LINE_STARTS_TO_IGNORE)):
            logging.info(f"Ignoring line: {line}")
            i += 1
            continue
        # seizure boundary
        elif detect_start_end(line):
            seizure, n_lines_processed = process_start_end(lines[i:])
            i += n_lines_processed
        else:
            raise ValueError(f"Line {i} isn't valid: {line}")

        seizures.loc[len(seizures)] = seizure

    return seizures


def convert_uneeg_extended_and_for_mayo():
    for patient_dir in PATHS.patient_dirs(Dataset.for_mayo, Dataset.uneeg_extended):
        logging.info(f'--- {patient_dir.name} ---')

        # find annotation txt files
        txt_anns = [*patient_dir.szr_anns_original_dir.glob('*.txt')]
        for txt_annotation in txt_anns:
            seizures = annotations_txt_to_dataframe(txt_annotation)
            save_path = patient_dir.szr_anns_original_dir / f'{txt_annotation.stem}.csv'
            seizures.to_csv(save_path, index=False)
            # txt_annotation.unlink()


def convert_competition_data():
    sheet_path = PATHS.dataset_dirs[Dataset.competition] / "SeizureDatesTraining.xls"
    for patient_dir in PATHS.patient_dirs(Dataset.competition):
        patient = patient_dir.name
        logging.info(f'--- {patient} ---')
        # make a folder for the annotations
        patient_dir.szr_anns_dir.mkdir(exist_ok=True)

        # Retrieve the annotations xls file from the joined annotations file
        if sheet_path.exists():
            sheet = pd.read_excel(sheet_path, sheet_name=patient)
        else:
            logging.warning(f"{sheet_path} not found — could not split sheets")
            return

        # save seizure onset data
        seizures = pd.DataFrame(columns=['start'], dtype=str)
        seizures['start'] = sheet['onset']
        # Sort by 'start' column and get fresh numeric index
        seizures = seizures.sort_values('start').reset_index(drop=True)
        seizures.to_csv(patient_dir.all_szr_starts_file, index=True)

        # Save the start of recording and approximate day span data
        additional_info = sheet[['Day Start', 'Days Span approx.']]
        additional_info.to_csv(patient_dir.szr_anns_dir / "Time Span Info.csv", index=False)

    # delete the original sheet
    sheet_path.unlink()


def annotations_to_csv():
    convert_uneeg_extended_and_for_mayo()
    convert_competition_data()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    clean_mac_files(PATHS.base_dir)
    annotations_to_csv()
