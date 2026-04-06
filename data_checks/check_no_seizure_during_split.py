import pandas as pd
from pandas import Timedelta, Series

from config import PATHS

pdirs = PATHS.patient_dirs()

for pdir in pdirs:
    segs = pd.read_pickle(pdir.segments_table.pickle)
    szrs = pd.read_pickle(pdir.all_szr_starts_file.pickle)
    partition = pd.read_pickle(pdir.dataset_partition.pickle)
    first_test_seg_idx = partition.loc['first_idx_segs', 'test']

    print(pdir.name)
    split_seg: Series = segs.loc[first_test_seg_idx]
    type_ = split_seg['type']

    if type_ == 'interictal':
        print(f'✅ Segment type: {type_}')
    else:
        print(f'❌ Segment type: {type_}')

    start = split_seg['start_mtz']
    end = start + Timedelta(minutes=65)
    if ((start <= szrs['start_mtz']) & (szrs['start_mtz'] < end)).any():
        print('❌ Seizure after split')
    else:
        print('✅ No seizure after split')

    print()
