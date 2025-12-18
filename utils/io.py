from pathlib import Path
from typing import Union

from pandas import DataFrame, Series


def save_dataframe_multiformat(df: Union[DataFrame, Series], path: Path, csv_index: bool = False):
    """Save a pd.DataFrame in multiple formats (csv, pickle)."""
    df.to_csv(path.with_suffix('.csv'), index=csv_index)
    df.to_pickle(pickle_path(path))


def pickle_path(path: Path):
    path = path.with_suffix('.pkl')
    return path.with_name('.' + path.name)  # Hide it with the .
