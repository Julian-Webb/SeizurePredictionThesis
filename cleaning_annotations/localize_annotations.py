import logging

import pandas as pd
from pandas import DataFrame

from config import Dataset, PATHS, PatientDir
from config import save_dataframe_multiformat
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


def drop_duplicates_and_localize_for_pdirs(pdirs: list[PatientDir]):
    for pdir in pdirs:
        is_competition = pdir.dataset == Dataset.competition
        datetime_cols = ['start_naive'] if is_competition else ['start_naive', 'end_naive']

        path = pdir.szr_starts_naive_file
        if path.csv.exists():
            anns = pd.read_csv(path.csv, parse_dates=datetime_cols)
        elif path.ods.exists():
            logging.warning(f'No csv annotation file found for: {pdir.name}. Using .ods instead.')
            anns = pd.read_excel(path.ods, parse_dates=datetime_cols)
        else:
            logging.warning(f'No annotation file found for: {pdir.name}')
            continue

        if anns.empty:
            logging.warning(f"Zero seizures in original annotation file of {pdir.name}.")
        else:
            # Drop duplicates
            dup_mask = anns['start_naive'].notna() & anns['start_naive'].duplicated(keep=False)
            if dup_mask.any():
                dups_df = anns.loc[dup_mask].sort_values('start_naive')
                logging.warning(
                    f"[{pdir.name}] Dropping {dup_mask.sum()} rows with duplicated start_naive. "
                    f"Keeping first occurrence.\n{dups_df}"
                )
                anns = anns.drop_duplicates(subset=['start_naive'], keep='first').reset_index(drop=True)

            # Localize, Sort, Save
            anns = localize_anns_dataframe(anns, PatientTimezone.from_competition(is_competition))
            anns.sort_values('start_mtz', inplace=True)
            save_dataframe_multiformat(anns, pdir.all_szr_starts_file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    drop_duplicates_and_localize_for_pdirs(PATHS.patient_dirs())
