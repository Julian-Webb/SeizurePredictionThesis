import pandas as pd

from config import PATHS
from config.paths import pickle_path

pdirs = PATHS.patient_dirs()

for pdir in pdirs:

    edfs = pd.read_pickle(pickle_path(pdir.edf_files_table))
    szrs = pd.read_pickle(pickle_path(pdir.all_szr_starts_file))
    szr_starts = szrs['start_mtz']
    szr_contained = {}

    for szr in szr_starts:
        in_edf = (edfs['start_mtz'] <= szr) & (szr <= edfs['end_mtz'])
        szr_contained[szr] = in_edf.any()

    szr_contained = pd.Series(szr_contained, name='contained')

    not_contained = szr_contained[~szr_contained]

    if len(not_contained) == 0:
        print(f'✓ {pdir.name}')
    else:
        print(f'x {pdir.name}')
        print(not_contained)


