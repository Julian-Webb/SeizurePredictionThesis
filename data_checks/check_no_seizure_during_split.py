import pandas as pd
from pandas import Timedelta, Series

from config import PATHS

pdirs = PATHS.patient_dirs()

for pdir in pdirs:
    segs = pd.read_pickle(pdir.segments_table.pickle)
    szrs = pd.read_pickle(pdir.all_szr_starts_file.pickle)
    split_idx = pd.read_pickle(pdir.train_test_split.pickle).segment_index

    print(pdir.name)
    split_seg: Series = segs.loc[split_idx]
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
