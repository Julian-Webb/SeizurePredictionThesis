import logging

import pandas as pd
from pandas import DataFrame

from config.paths import Dataset, PATHS, PatientDir
from utils.io import save_dataframe_multiformat
from utils.timezone import PatientTimezone


def localize_anns_dataframe(anns: DataFrame, tz: PatientTimezone) -> DataFrame:
    """
    Localizes the annotations Timestamps. Everything is converted to the patient's main timezone, but the localized
    start is also saved.
    :return: The localized annotations
    """
    if anns.empty:
        return anns
    s_naive = pd.to_datetime(anns['start_naive'])

    # NOTE: ambiguous could be changed to 'infer' if necessary
    # noinspection PyUnresolvedReferences
    s_localized = s_naive.dt.tz_localize(tz.location, ambiguous='raise')
    # Convert to the main timezone and don't save timezone information
    start_mtz = s_localized.dt.tz_convert(tz.main_timezone).dt.tz_localize(None)

    # Insert start_localized left start
    idx = anns.columns.get_loc('start_naive')
    anns.insert(idx + 1, 'start_localized', s_localized)
    anns.insert(idx + 2, 'start_mtz', start_mtz)

    # Adjust ends
    if 'end_naive' in anns.columns:
        has_end = anns['end_naive'].notna()
        durations = anns.loc[has_end, 'end_naive'] - s_naive.loc[has_end]
        end_mtz = anns.loc[has_end, 'start_mtz'] + durations
        anns.insert(anns.columns.get_loc('end_naive') + 1, 'end_mtz', end_mtz)

    return anns


def drop_duplicates_and_localize(pdirs: list[PatientDir]):
    for pdir in pdirs:
        # Get the correct annotation path
        is_competition = pdir.dataset == Dataset.competition
        path = pdir.szr_starts_naive_file.with_suffix(
            '.csv') if is_competition else pdir.szr_anns_original_dir / f'{pdir.name}_Consensus.csv'

        # Localize
        if path.exists():
            datetime_cols = ['start_naive'] if is_competition else ['start_naive', 'end_naive']
            anns = pd.read_csv(path, parse_dates=datetime_cols)

            # Drop duplicates
            dup_mask = anns['start_naive'].notna() & anns['start_naive'].duplicated(keep=False)
            if dup_mask.any():
                dups_df = anns.loc[dup_mask].sort_values('start_naive')
                logging.warning(
                    f"[{pdir.name}] Dropping {dup_mask.sum()} rows with duplicated start_naive. "
                    f"Keeping first occurrence.\n{dups_df}"
                )

                anns = anns.drop_duplicates(subset=['start_naive'], keep='first').reset_index(drop=True)

            # Localize
            anns = localize_anns_dataframe(anns, PatientTimezone.from_competition(is_competition))

            # Sort
            anns.sort_values('start_mtz', inplace=True)

            # Save
            save_dataframe_multiformat(anns, pdir.all_szr_starts_file)
        else:
            logging.warning(f'No annotation file found for: {pdir.name}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    drop_duplicates_and_localize(PATHS.patient_dirs())
