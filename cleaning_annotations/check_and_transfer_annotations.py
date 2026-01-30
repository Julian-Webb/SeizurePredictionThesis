import shutil
from pathlib import Path

import pandas as pd

from config.paths import PATHS, PatientDir


# noinspection PyUnresolvedReferences
def check_duplicate_anns(root: Path):
    dt_cols = ['start_naive', 'end_naive']

    all_anns = {'Consensus': [], 'AllAutomaticDetections': []}
    for type_, type_anns in all_anns.items():
        for pdir in sorted(root.iterdir()):
            ptnt_anns_path = pdir / f'{pdir.name}_{type_}.csv'
            try:
                ptnt_anns = pd.read_csv(ptnt_anns_path, parse_dates=dt_cols)
                ptnt_anns['patient'] = pdir.name
                type_anns.append(ptnt_anns)
            except FileNotFoundError:
                print(f'Could not find {ptnt_anns_path}')

        type_anns = pd.concat(type_anns, ignore_index=True)
        all_anns[type_] = type_anns

        for col in dt_cols:
            if not type_anns[col].dropna().is_unique:
                print(f'x {col} is not unique for {type_}:')
                dup_mask = type_anns[col].notna() & type_anns[col].duplicated(keep=False)
                dups_df = type_anns.loc[dup_mask].sort_values(col)
                print(dups_df)
                print()
            else:
                print(f'✓ {col} is unique for {type_}')



def copy_anns(src: Path, dst_dataset_dir: Path):
    for pdir in src.iterdir():
        dst_pdir = PatientDir(dst_dataset_dir / pdir.name)
        shutil.rmtree(dst_pdir.szr_anns_original_dir, ignore_errors=True)
        dst_pdir.szr_anns_original_dir.mkdir(exist_ok=True, parents=True)
        for file in pdir.iterdir():
            dst = dst_pdir.szr_anns_original_dir / file.name
            shutil.copy2(file, dst)


def main():
    root = Path('/data/home/webb/seizure_annotations/STEP4_combined_csv_anns')
    check_duplicate_anns(root)
    copy_anns(root, PATHS.ultra2_dir)


if __name__ == '__main__': main()
