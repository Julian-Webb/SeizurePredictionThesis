# See data_handling.ipynb for detailed documentation
import logging
import re
import pandas as pd
from pathlib import Path

from config import SEIZURE_ANNOTATIONS_FOLDER_NAME, SEIZURE_ANNOTATIONS_FILE_NAME, BASE_DIR, \
    FOR_MAYO_DIR, UNEEG_EXTENDED_DIR, COMPETITION_DIR
from data_cleaning.file_correction import clean_mac_files

# the possible ways a line can start
SINGLE_MARKER_STARTS = ['Seizure-rhythmic', 'Seizure-rhythmic +', 'Seizure-tonic', 'Seizure_Start', 'Seizure_End',
                        'BUTTON DOUBLE PRESSED']

LINE_START_STRINGS = SINGLE_MARKER_STARTS + ['User seizure marker']

SEIZURE_START_STRINGS = ['Seizure_Start', 'Seizure Start']
SEIZURE_END_STRINGS = ['Seizure_End', 'Seizure End']


def read_line(line: str):
    """Converts a line into the type, dates, and comment. Asserts that the dates are equal, and the type is valid.
    :returns: The type, datetime, and comment"""
    comment = ''
    line = line.rstrip()  # remove whitespace and new line from end of string
    # sometimes, there are multiple tabs or spaces after the seizure type. We replace this with a single tab
    line = re.sub(r'\s{2,}', r'\t', line)
    values = line.split('\t')
    if len(values) == 4:
        # it contains a comment
        comment = values.pop()
    szr_type, datetime1, datetime2 = values

    if szr_type not in LINE_START_STRINGS:
        print(f'{line}')
        print(f'{szr_type=}')
        raise ValueError(f"The following line doesn't start with {LINE_START_STRINGS}: {line}")
    if datetime1 != datetime2:
        raise ValueError(f"The dates are not the same. {datetime1=}  ,  {datetime2=}")

    # If it's a user marker, it should start with something like "Seizure_Start", but may contain more
    comment_parts = comment.split(', ', maxsplit=1)

    return szr_type, datetime1, comment, comment_parts


def get_seizure_lines_from_file(annotation_path: Path):
    """Preprocess an annotation file to get just the clean lines containing seizures."""
    with open(annotation_path, 'r') as file:
        # read file and remove trailing white space and get separate lines of file
        lines = file.read().rstrip().split('\n')

    # assert that the file starts with the patient ID
    assert lines[0].startswith('Patient ID') or lines[0].startswith(
        'Patienten-ID'), "Annotation files doesn't start with 'Patient ID' or 'Patienten-ID'"
    lines.pop(0)  # remove the first line
    return lines


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


def process_start_end(start_line: str, end_line: str):
    """Converts successive lines with a start and end into a seizure annotation."""
    seizure = {'comment': ''}

    # process start line
    szr_type, datetime, comment, comment_parts = read_line(start_line)
    assert szr_type == 'User seizure marker' and comment_parts[0] in SEIZURE_START_STRINGS, \
        f"This line should be the start of a seizure: {start_line}"
    seizure['type'] = szr_type
    seizure['start'] = datetime
    if len(comment_parts) > 1:
        add_comment(seizure, comment_parts[1])

    # process end line
    szr_type, datetime, comment, comment_parts = read_line(end_line)
    assert szr_type == 'User seizure marker' and comment_parts[0] in SEIZURE_END_STRINGS, \
        f"This line should be the end of a seizure: {end_line}"
    seizure['end'] = datetime
    if len(comment_parts) > 1:
        add_comment(seizure, comment_parts[1])

    return seizure


def process_start_single_marker_end(start_line: str, single_marker_line: str, end_line: str):
    """Converts successive lines with a start, single marker, and end into a seizure annotation."""
    seizure = process_start_end(start_line, end_line)

    # add single marker info
    szr_type, datetime, comment, _ = read_line(single_marker_line)
    seizure['single_marker'] = datetime
    seizure['type'] = szr_type
    add_comment(seizure, comment)

    return seizure


def annotations_txt_to_dataframe(annotation_path: Path):
    lines = get_seizure_lines_from_file(annotation_path)

    # go through the lines (seizures) and store them in a dataframe
    seizures = pd.DataFrame(columns=['type', 'start', 'single_marker', 'end', 'comment'], dtype=str)

    # Check if it contains no seizures
    if lines[0].lower() == 'no seizures':
        return seizures

    # convert the lines into seizure annotations
    i = 0
    while i < len(lines):
        if lines[i].startswith('User seizure marker'):
            if lines[i + 1].startswith('User seizure marker'):
                seizure = process_start_end(lines[i], lines[i + 1])
                i += 2  # The next line already gets processed
            else:
                # a single marker is between the start and end
                seizure = process_start_single_marker_end(lines[i], lines[i + 1], lines[i + 2])
                i += 3  # The next two lines already get processed
        else:
            seizure = process_single_marker(lines[i])
            i += 1

        # Add seizure to DataFrame
        seizures.loc[len(seizures)] = seizure

    return seizures


def convert_uneeg_extended_and_for_mayo(for_mayo: Path, uneeg_extended: Path):
    for patient_folder in [*for_mayo.iterdir(), *uneeg_extended.iterdir()]:
        patient = patient_folder.name
        logging.info(f'--- {patient} ---')

        # find annotation txt files
        annotations_folder = patient_folder / SEIZURE_ANNOTATIONS_FOLDER_NAME
        txt_annotations = [*annotations_folder.glob('*.txt')]
        for txt_annotation in txt_annotations:
            seizures = annotations_txt_to_dataframe(txt_annotation)
            save_path = annotations_folder / f'{txt_annotation.stem}.csv'
            seizures.to_csv(save_path, index=False)
            txt_annotation.unlink()


def convert_competition_data(competition_dir: Path):
    patient_folders = [item for item in competition_dir.iterdir() if item.is_dir()]
    sheet_path = competition_dir / "SeizureDatesTraining.xls"
    for patient_folder in patient_folders:
        patient = patient_folder.name
        logging.info(f'--- {patient} ---')
        # make a folder for the annotations
        annotations_folder = patient_folder / SEIZURE_ANNOTATIONS_FOLDER_NAME
        annotations_folder.mkdir(exist_ok=True)

        # Retrieve the annotations xls file from the joined annotations file
        if sheet_path.exists():
            sheet = pd.read_excel(sheet_path, sheet_name=patient)
        else:
            logging.warning(f"{sheet_path} not found — could not split sheets")
            return

        # save seizure onset data
        seizures = pd.DataFrame(columns=['type', 'start', 'single_marker', 'end', 'comment'], dtype=str)
        seizures['start'] = sheet['onset']
        seizures.to_csv(annotations_folder / SEIZURE_ANNOTATIONS_FILE_NAME, index=False)

        # Save the start of recording and approximate day span data
        additional_info = sheet[['Day Start', 'Days Span approx.']]
        additional_info.to_csv(annotations_folder / "Time Span Info.csv", index=False)

    # delete the original sheet
    sheet_path.unlink()


def annotations_to_csv(for_mayo_dir: Path, uneeg_extended_dir: Path, competition_dir: Path):
    convert_uneeg_extended_and_for_mayo(for_mayo_dir, uneeg_extended_dir)
    convert_competition_data(competition_dir)


if __name__ == '__main__':
    clean_mac_files(BASE_DIR)
    annotations_to_csv(FOR_MAYO_DIR, UNEEG_EXTENDED_DIR, COMPETITION_DIR)
