import logging
import os
import re
from pprint import pprint

import pandas as pd
from pyedflib import EdfReader

from config.paths import Dataset, PATHS


def check_edf_names_with_patient_id_correct_format():
    other_types = []
    for pdir in PATHS.patient_dirs([Dataset.uniclinic]):
        logging.info(f'Processing {pdir.name}')
        for edf in pdir.edf_dir.iterdir():
            if edf.name.lower().startswith('u'):
                # If it starts with U, it should start with the following format
                if not re.match(r'^U002-DE01-\d{2}', edf.name):
                    logging.warning(f'{[pdir.name]} x Incorrect format: {edf.name}')
                else:
                    logging.debug(f'{[pdir.name]} ✓ : {edf.name}')
            elif re.match(r'^\d{6}_\d{7}_\d{8}_\d{2,3}_EEGdata', edf.name):
                logging.debug(f'{[pdir.name]} {edf.name}')
            else:
                # other_types.append(edf.relative_to(PATHS.datasets_dir))
                other_types.append(edf)

    if other_types:
        print(f'other edf formats:')
        for path in other_types:
            print(str(path))
    return other_types


def correct_edf_names_with_patient_id():
    correct = 'U002-DE01-'
    replacements = [
        ['U002_DE01_', correct],
        ['U002-DE01_', correct],
        ['U002_DE01-', correct],
    ]

    for pdir in PATHS.patient_dirs([Dataset.uniclinic]):
        logging.info(f'Processing {pdir.name}')
        for edf in pdir.edf_dir.iterdir():
            if edf.name.lower().startswith('u'):
                new_name = edf.name
                for rep in replacements:
                    new_name = new_name.replace(*rep)
                if new_name != edf.name:
                    logging.info(f'{[pdir.name]} :\n   {edf.name}\n-> {new_name}')
                    edf.rename(edf.parent / new_name)


def show_edf_info_per_ptnt():
    keys = ['patientcode', 'equipment']
    all_ptnt_infos = {}

    # Iterate over all patients and edfs
    for pdir in PATHS.patient_dirs([Dataset.uniclinic]):
        print(f'---- {pdir.name}:')

        # Read Info from files
        ptnt_edfs = list(pdir.edf_dir.iterdir())
        ptnt_edf_names = [edf.name for edf in ptnt_edfs]
        edfs = pd.DataFrame(index=ptnt_edf_names, columns=keys)

        for edf_path in ptnt_edfs:
            with EdfReader(str(edf_path)) as edf:
                header = edf.getHeader()
            for key in keys:
                edfs.loc[edf_path.name, key] = header[key]

        for key in keys:
            print(edfs[key].value_counts())
            print()

        all_ptnt_infos[pdir.name] = edfs

    return all_ptnt_infos



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    # other_types = check_edf_names_with_patient_id_correct_format()
    # move_unknown_implant_files()
    # correct_edf_names_with_patient_id()
    show_edf_info_per_ptnt()
