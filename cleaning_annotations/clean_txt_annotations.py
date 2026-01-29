import re
import shutil
from enum import Enum
from pathlib import Path

import pandas as pd
from pandas import DataFrame


# Take the raw original txt annotations and transform them into something usable with a consistent format

class AnnotationType(Enum):
    Consensus = 'Consensus'
    AllAutomaticDetections = 'AllAutomaticDetections'
    SeizureStartEnd = 'SeizureStartEnd'
    SeizureStartEndPureVisual = 'SeizureStartEndPureVisual'


def clean_filename(path: Path):
    replacements_correct2wrong = {
        '_': ['__'],
        'Consensus.txt': [
            'CONSENSUS.txt',
            'CONSENUS.txt',
            'CONESNUS.txt',
            'KONSENSUS.txt',
            'COSENSUS.txt',
            'COSENSU.txt',
            'consensus.txt',
            'CONSENSUs.txt',
            'CONSENSUe.txt',
        ],
        '_SeizureStartEnd': ['_StartEnd'],
        'SeizureStartEnd': [
            'StartAndEnd',
            'startandend',
            'SeizureStartEnde',
            'STARTANDEND'
        ],
        'YN_SeizureStartEnd': [
            'YNStartandENd',
            'YNStartEnd',
            'YNSeiuzreStartEnd',
            'YNSeizureStartEnd'
        ],
        'SeizureStartEndPureVisual': ['StartAndEndPureVisual'],
        'AllAutomaticDetections.txt': [
            'all automatic detections.txt',
            'all automatic detection.txt',
            'AllautomaticAnnotations.txt',
            'AllautomaticAnnotatios.txt',
            'All_Automatic_Detections.txt',
            'AllAutomatciDetections.txt',
            'AllAutomatciDetections.txt',
            'all automatic annotations.txt',
        ],
        'SUBQ_ANN_YN': [
            'SUBQ_ANN_YU',
            'SUBQ_ANN_NY'
        ],
        'SUBQ_V5c': ['SUBQ_V5C'],
        # underscore instead of dash or dash instead of underscore
        'U002-DE01-': ['U002_DE01-', 'U002-DE01_', 'U002-De01-'],
        '_OUTPT': ['-OUTPT'],
        '_EMU_V4_': ['-EMU-V4_', '_EMU_V4-'],
        'V5b_AllAutomaticDetections': ['V5b-AllAutomaticDetections'],
    }

    new_name = path.name
    for correct, wrong_list in replacements_correct2wrong.items():
        for wrong in wrong_list:
            new_name = new_name.replace(wrong, correct)

    if new_name != path.name:
        # print(f"Renamed\n  {path.name}\n> {new_name}")
        path.rename(path.parent / new_name)
    # else:
    #     print(f"No change needed for {path.name}")


def move_duplicated_files(data: Path, removed_dir: Path):
    """
    Certain data is duplicated.
    Often because some visits are stored individually and as bundled visits (e.g. V5e, and V5e-f)
    """
    files_to_move = {
        'U002-DE01-05': [
            # This file is the same as U002-DE01-05_OUTPT_V5j_SUBQ_ANN_ASB_Consensus.txt
            'U002-DE01-05_OUTPT_V5_SUBQ_ANN_ASB_Consensus',
            # completely empty file
            'U002-DE01-05_OUTPT_V5k_ANN_Consensus'
        ],
        'U002-DE01-15': [
            'U002-DE01-15_OUTPT_V5a-c_SUBQ_MH',
            'U002-DE01-15_OUTPT_V5a-c_SUBQ_YN',
            'U002-DE01-15_OUTPT_V5d_e_SUBQ_YN',
            'U002-DE01-15_OUTPT_V5d_f_SUBQ_AllAutomaticDetections',
            'U002-DE01-15_OUTPT_V5d_f_SUBQ_MH',
            'U002-DE01-15_OUTPT_V5f_SUBQ_YN',  # this is covered by 'U002-DE01-15_OUTPT_V5f_SUBQ_ANN_YN'
            # there's already an equivalent file 'U002-DE01-15_OUTPT_V5g_SUBQ_ANN_YN'
            'U002-DE01-15_OUTPT_V5g_SUBQ_YN',
            'U002-DE01-15_OUTPT_V5i_SUBQ_YN',  # this is covered by 'U002-DE01-15_OUTPT_V5i_SUBQ_ANN_YN'
            # This file belongs to patient 16 (as by patient ID in file and automatic detections of patient 16)
            # It is already contined in patient 16's annotations
            'U002-DE01-15_OUTPT_V5h_SUBQ_ANN_MH'
        ],
        'U002-DE01-16': [
            'U002-DE01-16_OUTPT_V5a-c_SUBQ_MH',
            'U002-DE01-16_OUTPT_V5a-c_SUBQ_YN',
            'U002-DE01-16_OUTPT_V5d-f_SUBQ_MH',
        ],
        'U002-DE01-23': [
            'U002-DE01-23_OUTPT_V5e_SUBQ_ANN_YNupto20250930',
        ]
    }

    for ptnt, files in files_to_move.items():
        for file in files:
            path = data / ptnt / (file + '.txt')
            if path.exists():
                new_path = removed_dir / 'duplicate' / ptnt / path.name
                new_path.parent.mkdir(exist_ok=True, parents=True)
                path.rename(new_path)
                # print(f"Moved            : {path}")
            else:
                print(f'x non-existent file: {path}')


def move_unwanted_files(data: Path, removed_dir: Path):
    for pdir in data.iterdir():
        for file in pdir.iterdir():
            if 'scalp' in file.name.lower():
                new_path = removed_dir / 'scalp' / pdir.name / file.name
                new_path.parent.mkdir(exist_ok=True, parents=True)
                file.rename(new_path)
                # print(f"Moved            : {file}")


def file_content_corrections(data: Path):
    for pdir in sorted(list(data.iterdir())):
        for filepath in sorted(list(pdir.iterdir())):
            content = filepath.read_text()

            # Normalize whitespace
            new = re.sub(r' {2,}', ' ', content)  # replace multiple spaces with a single space
            new = re.sub(r'[ \t]{2,}', '\t', new)  # replace multiple spaces/tabs with a single tab
            # Remove empty lines and remove trailing whitespace
            new = "\n".join(line.strip() for line in new.splitlines() if line.strip())

            new = re.sub(r'Patienten-ID', r'Patient ID', new)
            new = re.sub('Patient ID', 'Patient ID', new, flags=re.IGNORECASE)
            if not new.startswith('Patient ID: '):
                # Sometimes the files just start with the patient ID itself, without this prefix
                new = 'Patient ID: ' + new

            if pdir.name == 'U002-DE01-01':
                # This patient sometimes has the raw patientcode
                new = new.replace('0004107', 'U002-DE01-01')

            if pdir.name == 'U002-DE01-02':
                new = new.replace('U002-DE01-2', 'U002-DE01-02')

            if pdir.name == 'U002-DE01-03':
                new = new.replace('0004111', 'U002-DE01-03')

            # Additional replacements
            repl_correct2wrong = {
                'U002-DE01-': [
                    'U002_DE01_',
                    'U003-DE01-'
                ],
                # Remove "Visit 5a" on its own line
                '\n': ['\nVisit 5a\n'],
                'No seizures by reviewer MH.': ['No registred seazures by rewiever MH.'],
                'Seizure_Start': [
                    'Seizure_Start_Right',
                    'seizure_Start_Right',
                    'Seizure_Start_Left',
                    'Seizure Start',
                    'SUB_Start',
                ],
                'Seizure_End': [
                    'Seizure_End (UNEEG)',
                    'Seizure_End_Right',
                    'seizure_End_Right',
                    'Seizure_End_Left',
                    'Seizure End',
                    'SUB-Ende',
                ],
                '\nSeizure-rhythmic\t': ['\n6Seizure-rhythmic\t']
            }

            for correct, wrong_list in repl_correct2wrong.items():
                for wrong in wrong_list:
                    new = new.replace(wrong, correct)

            # Save
            if new != content:
                # print(f'Altering {filepath.name}')
                filepath.write_text(new)
            # else:
            #     print(f'File ok: {filepath.name}')


def clean_txt_annotations(root: Path, data: Path):
    """
    Clean up the txt annotation files
    """
    for pdir in sorted(data.iterdir()):
        for ann_path in pdir.iterdir():
            clean_filename(ann_path)

    move_duplicated_files(data, root / 'removed')
    move_unwanted_files(data, root / 'removed')

    file_content_corrections(data)


# ----------- BUILD FILE INDEX -------------------

PRIMARY_REVIEWERS = {
    'MH': 'Martin Hirsch',
    'YN': 'Yulia Novitskaya',
    'EM': 'Eva Martinez',
}

SUPERVISOR_REVIEWERS = {
    'ASB': 'Andreas Schulze-Bonhage'
}

ALL_REVIEWERS = {**PRIMARY_REVIEWERS, **SUPERVISOR_REVIEWERS}

LOCATIONS = {
    'EMU': 'Epilepsy monitoring unit',
    'OUTPT': 'Outpatient'
}

# just some additional information from the file name
NOTES = {
    'SUBQ': 'subcutaneous',
    'ANN': 'annotations'
}
VISIT_PATTERN = re.compile(r'^[vV]\d(?:[a-z](?:-[a-z])?)?$')


def interpret_file_name(file_name: str, patient_id: str):
    file_name = file_name.replace('.txt', '')

    parts = file_name.split('_')

    # handle patient id
    assert parts[0] == patient_id, f'Patient id mismatch: {parts[0]} != {patient_id} for file: {file_name}'

    # handle remaining parts
    location = None
    visit = None
    annotation_type = None
    reviewer_id = None

    for part in parts[1:]:
        if part in LOCATIONS.keys():
            location = part
            if location == 'EMU':
                visit = 'V4'
        elif VISIT_PATTERN.fullmatch(part):
            visit = part
        elif part in NOTES.keys():
            pass
        elif part in [a.value for a in AnnotationType]:
            assert annotation_type is None, f'Multiple annotation types found in file: {file_name}'
            annotation_type = part
        elif part in ALL_REVIEWERS.keys():
            reviewer_id = part
        else:
            raise ValueError(f'Unknown part: {part} in file: {file_name}')

    return {
        'file_name': file_name,
        'patient_id': patient_id,
        'location': location,
        'visit': visit,
        'annotation_type': annotation_type,
        'reviewer_id': reviewer_id,
    }


def build_file_index(data: Path):
    index = []
    for pdir in data.iterdir():
        for file in pdir.iterdir():
            index.append(interpret_file_name(file.name, pdir.name))
    index = pd.DataFrame(index)
    index.sort_values(['patient_id', 'visit', 'location', 'annotation_type', 'reviewer_id'], inplace=True)
    index.reset_index(drop=True, inplace=True)
    return index


def check_file_index(index: DataFrame):
    """
    Check that each patient has visit 4.
    Check that each visit contains the following files:
    * AllAutomaticDetections
    * Consensus
    * Additional 2 files by different reviewers
    * SeizureStartEnd (optional)
    """
    # Loop over patients:
    for ptnt, pfiles in index.groupby('patient_id', dropna=False):
        if 'V4' not in pfiles['visit'].values:
            print(f'x Patient {ptnt} is missing visit 4')

    # Loop over visits
    for (ptnt, visit), vfiles in index.groupby(['patient_id', 'visit'], dropna=False):
        if 'AllAutomaticDetections' not in vfiles['annotation_type'].values:
            # Visit 4 doesn't have AllAutomaticDetections
            if visit != 'V4':
                print(f'x Visit {ptnt} {visit} is missing AllAutomaticDetections')
        if 'Consensus' not in vfiles['annotation_type'].values:
            print(f'x Visit {ptnt} {visit} is missing Consensus')

        single_reviewer_files = vfiles[vfiles['annotation_type'].isna()]
        if len(single_reviewer_files) != 2:
            print(f'x Visit {ptnt} {visit} has {len(single_reviewer_files)} SingleReviewer files')
        if not single_reviewer_files['reviewer_id'].is_unique:
            print(f'x Visit {ptnt} {visit} has non-unique SingleReviewer files')


def main():
    original = Path("/data/home/webb/seizure_annotations/STEP1_original_anns")
    root = Path("/data/home/webb/seizure_annotations/STEP2_cleaned_anns")
    # Copy original annotations
    if root.exists(): shutil.rmtree(root)
    shutil.copytree(original, root)

    data = root / 'data'
    clean_txt_annotations(root, data)
    index = build_file_index(data)
    index.to_csv(root / 'file_index.csv', index=False)

    check_file_index(index)


if __name__ == '__main__': main()
