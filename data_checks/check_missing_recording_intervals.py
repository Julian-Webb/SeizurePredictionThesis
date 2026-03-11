import pandas as pd
from pandas import DataFrame, Timedelta

from config import PATHS
from utils.io import pickle_path

pdirs = PATHS.patient_dirs()

missing_by_patient: dict[str, DataFrame] = {}

for pdir in pdirs:
    edfs = pd.read_pickle(pickle_path(pdir.edf_files_table))

    missing_by_patient[pdir.name] = DataFrame({
        'previous_file': edfs['file_name'].values,
        'next_file': edfs['file_name'].iloc[1:].tolist() + [None],
        'start': edfs['end_mtz'].values,
        # shift times up by one row to align with the start of the next recording
        'end': edfs['start_mtz'].iloc[1:].tolist() + [None],
    })

missing = pd.concat(
    missing_by_patient,
    names=['patient', 'interval_i'],
)

missing['duration'] = missing['end'] - missing['start']

overlapping = missing[missing['duration'] < Timedelta(0)]
overlapping['overlap'] = overlapping['start'] - overlapping['end']
print(f'Total missing intervals: {len(missing)}')
print(f'Negative missing intervals: {len(overlapping)}')

