from pathlib import Path

from pandas import DataFrame


def save_annotations(anns: DataFrame, path: Path):
    """
    Save seizure annotations to a CSV and pickle file
    """
    anns.to_csv(path.with_suffix('.csv'), index=False)
    anns.to_pickle(path.with_suffix('.pkl'))
