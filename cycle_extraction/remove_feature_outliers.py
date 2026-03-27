import math
from typing import Literal

import pandas as pd
from pandas import Timedelta, DataFrame


def rolling_average(
        df: DataFrame,
        window: Timedelta,
        min_valid_duration: Timedelta,
        sampling_period: Timedelta,
        agg: Literal["mean", "median"],
        on: str = 'start_mtz',
) -> DataFrame:
    """
    Compute a centered time-based rolling aggregation for segment features while
    preserving original NaN rows.

    Parameters
    ----------
    df : DataFrame
        Input dataframe containing `start_mtz` (datetime) and feature columns.
        Aggregation is performed with `on='start_mtz'`.
    window : Timedelta
        Rolling window length (e.g. `Timedelta(minutes=10)`).
    min_valid_duration : Timedelta
        Minimum required valid duration inside each window for emitting a value.
        This is converted to `min_periods` as:
        `ceil(min_valid_duration / sampling_period)`.
    sampling_period : Timedelta
        Segment spacing (time between consecutive samples), e.g. 15 seconds.
    agg : {"mean", "median"}
        Aggregation function used on each window.
    on : str, default "start_mtz"
        Which column to use as index

    Returns
    -------
    DataFrame
        Rolling-aggregated dataframe with the same row count as `df`.
        Rows that were NaN in the input are forced back to NaN in the output
        to avoid implicit gap filling at this stage.

    Notes
    -----
    - `min_periods` counts non-NaN samples, even for time-based windows.
    - `center=True` and `closed='both'` are used for symmetric windows.

    Example
    -------
    segs_cleaned = rolling_average(
        segs,
        window=Timedelta(minutes=10),
        min_valid_duration=Timedelta(minutes=3),
        sampling_period=Timedelta(seconds=15),
        agg="median")
    """

    # min_periods: how many samples are required in a window for a value to be output
    min_periods = math.ceil(min_valid_duration / sampling_period)

    rolling = df.rolling(
        window=window,
        min_periods=min_periods,
        center=True,
        on=on,
        closed='both',
    )

    averages = getattr(rolling, agg)()

    # Make NaN rows stay NaN, rather than being filled with surrounding values
    na_rows = df.isna().any(axis='columns')
    averages.loc[na_rows, :] = pd.NA

    return averages
