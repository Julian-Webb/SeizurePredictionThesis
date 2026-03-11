# todo delete this temp file
import shutil
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from config import PATHS
from preprocessing.validate_patients import ptnt_timespan_info
from utils.io import save_dataframe_multiformat, pickle_path

# ----- PATIENT INFOS ------
# ptnt_infos = {'exact': {}, 'readable': {}}
#
# # Get infos
# for pdir in PATHS.patient_dirs(include_invalid_ptnts=True):
#     ptnt_time_info = ptnt_timespan_info(pdir)
#     for k in ptnt_infos.keys():
#         ptnt_infos[k][(pdir.dataset.value, pdir.name)] = ptnt_time_info[k]
#
# # Save
# for k, ptnt_info in ptnt_infos.items():
#     index = pd.MultiIndex.from_tuples(ptnt_info.keys(), names=['dataset', 'patient'])
#     ptnt_info = DataFrame(ptnt_info.values(), index=index)
#     ptnt_info.sort_values(by=['dataset', 'patient'], inplace=True, ascending=[True, True])
#     if k == 'readable':
#         ptnt_info.to_csv(PATHS.patient_info_readable.with_suffix('.csv'))
#     elif k == 'exact':
#         save_dataframe_multiformat(ptnt_info, PATHS.patient_info_exact)

# ------ VALID SEIZURE FILES FOR SEGMENT TABLES ------
# for pdir in PATHS.patient_dirs(include_invalid_ptnts=True):
#     valid_szrs = DataFrame(columns=['start'])
#     valid_szrs.to_pickle(pickle_path(pdir.valid_szr_starts_file))
#     valid_szrs.to_csv(pdir.valid_szr_starts_file.with_suffix('.csv'))

# ------ COPY FEATURES TO POOL ------
# for pdir in PATHS.patient_dirs():
#     dst = Path('/data/pool/sotirios/UNEEG_Features_with_interval_type_and_signals_filtered', pdir.name)
#     dst.mkdir(exist_ok=True, parents=True)
#
#     shutil.copy2(pdir / 'segments.csv', dst / 'segments.csv')
#     shutil.copy2(pdir / '.segments.pkl', dst / 'segments.pkl')


