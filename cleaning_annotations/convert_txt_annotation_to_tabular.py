import logging
from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series

from config.paths import Dataset, PATHS, PatientDir
from utils.io import save_dataframe_multiformat
from utils.timezone import PatientTimezone

LINE_STARTS = {
    'Seizure': ['Seizure-rhythmic', 'Seizure-rhythmic +', 'Seizure-tonic'],
    'Pattern-Rhythmic': ['Pattern-Rhythmic-Theta', 'Pattern-Rhythmic-Delta-SW'],
    'Seizure_boundary': ['Seizure_Start', 'Seizure_End'],
    'Button': ['BUTTON PRESSED', 'BUTTON DOUBLE PRESSED'],
}


def interpret_line(line: str, filename: str = ''):
    """Interpret a line from a text annotation file"""
    values = line.split('\t')
    # If it has 4 values, the last one is a comment
    comment = values.pop() if len(values) == 4 else ''
    #todo delete
    if comment:
        print(comment)
    type_, datetime1, datetime2 = values
    if any(type_ in LINE_STARTS[k] for k in ("Seizure", "Seizure_boundary", "Button")):
        assert datetime1 == datetime2, f"The dates are not the same:\n{datetime1}\n{datetime2}\n{filename=}"
        return {'type': type_, 'start_naive': datetime1, 'comment': comment}
    elif type_ in LINE_STARTS["Pattern-Rhythmic"]:
        assert datetime1 != datetime2, f"Pattern Rhythmic, but the dates are the same:\n{datetime1}\n{datetime2}\n{filename=}"
        return {'type': type_, 'start_naive': datetime1, 'end_naive': datetime2, 'comment': comment}
    else:
        raise ValueError(f"Unknown type: {type_}\n{filename=}")


def txt_to_dataframe(path: Path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    szrs = []
    for i, line in enumerate(lines):
        try:
            res = interpret_line(line, filename=path.name)
            # Note starts and ends are ignored
            if res['type'] in LINE_STARTS['Seizure_boundary']:
                logging.warning(f"[{path.name}] Ignoring line: {line}")
            else:
                szrs.append(res)
        except Exception as e:
            logging.error(f"Error in\n{path}\n{i}: {line}\n{e}\n")
            # todo uncomment
            # logging.error(f"Error in\n{path}\n{i}: {line}")
            # raise e

    szrs_df = DataFrame(szrs, columns=['type', 'start_naive', 'end_naive', 'comment'])
    return szrs_df


# todo check this (delete?)
def _localize_annotations_dataframe(anns: DataFrame, is_competition_ptnt: bool, datetime_cols: list[str]) -> DataFrame:
    """
    Localizes the annotations Timestamps. Everything is converted to the patient's main timezone.
    :return: The localized annotations
    """
    tz = PatientTimezone.from_competition(is_competition_ptnt)

    for col in datetime_cols:
        c = pd.to_datetime(anns[col])
        if c.notna().any():  # Check if there are any values in column
            # NOTE: ambiguous could be changed to 'infer' if necessary
            c = c.dt.tz_localize(tz.location, ambiguous='raise')
            c = c.dt.tz_convert(tz.main_timezone)  # Convert to main timezone
            c = c.dt.tz_localize(None)  # Don't save timezone information
        anns[col] = c
    return anns


def _localize_anns_dataframe(anns: DataFrame, tz: PatientTimezone) -> DataFrame:
    """
    Localizes the annotations Timestamps. Everything is converted to the patient's main timezone, but the localized
    start is also saved.
    :return: The localized annotations
    """
    if anns.empty:
        return anns
    s_naive = pd.to_datetime(anns['start_naive'])

    # NOTE: ambiguous could be changed to 'infer' if necessary
    s_localized = s_naive.dt.tz_localize(tz.location, ambiguous='raise')
    # Convert to the main timezone and don't save timezone information
    start_mtz = s_localized.dt.tz_convert(tz.main_timezone).dt.tz_localize(None)

    # Insert start_localized left start
    idx = anns.columns.get_loc("start_naive")
    anns.insert(idx+1, "start_localized", s_localized)
    anns.insert(idx+2, "start_mtz", start_mtz)

    # Adjust ends
    if 'end_naive' in anns.columns:
        has_end = anns['end_naive'].notna()
        durations = anns.loc[has_end, 'end_naive'] - s_naive.loc[has_end]
        anns.loc[has_end, 'end_mtz'] = anns.loc[has_end, 'start_mtz'] + durations

    return anns


def convert_uniclinic_anns(pdirs: list[PatientDir]):
    for pdir in pdirs:
        logging.info(f'Processing {pdir.name}')
        assert pdir.dataset == Dataset.uniclinic
        # find annotation txt files
        txt_anns = [*pdir.szr_anns_original_dir.glob('*.txt')]
        # Convert each file
        for txt_path in txt_anns:
            anns_df = txt_to_dataframe(txt_path)
            save_dataframe_multiformat(anns_df, txt_path)


def localize_anns(pdirs: list[PatientDir]):
    for pdir in pdirs:
        # Get the correct annotation path
        is_competition = pdir.dataset == Dataset.competition
        path = pdir.szr_starts_naive_file.with_suffix(
            '.csv') if is_competition else pdir.szr_anns_original_dir / f'{pdir.name}_Consensus.csv'

        # Localize
        if path.exists():
            datetime_cols = ['start_naive'] if is_competition else ['start_naive', 'end_naive']
            anns = pd.read_csv(path,parse_dates=datetime_cols)
            anns = _localize_anns_dataframe(anns, PatientTimezone.from_competition(is_competition))
            save_dataframe_multiformat(anns, pdir.all_szr_starts_file)
        else:
            logging.warning(f'No annotation file found for: {pdir.name}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    # convert_uniclinic_anns(sorted(list(Path('/data/home/webb/STEP3_combined_anns').iterdir())))
    convert_uniclinic_anns(PATHS.patient_dirs([Dataset.uniclinic]))
    localize_anns(PATHS.patient_dirs())
