import numpy as np
import pandas as pd

from cycle_extraction.fill_feature_gaps import fill_ptnt_gaps, fill_short_gaps_per_column

def test_min_donor_window_changes_sampling_pool():
    idx = np.arange(0, 11)
    df = pd.DataFrame(
        {
            'exists': [False] + [True] * 10,
            'feat': [np.nan, 10.0, 20.0, 30.0, 40.0, 50.0, 999.0, 999.0, 999.0, 999.0, 999.0],
        },
        index=idx,
    )

    filled_min_1 = fill_short_gaps_per_column(
        df,
        feature_cols=['feat'],
        min_donor_n_segs=1,
        max_distortion_pct=0.0,
        random_state=123,
    )
    assert filled_min_1.loc[0, 'feat'] == 10.0

    rng = np.random.default_rng(123)
    expected_from_pool_5 = rng.choice(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), size=1, replace=True)[0]

    filled_min_5 = fill_short_gaps_per_column(
        df,
        feature_cols=['feat'],
        min_donor_n_segs=5,
        max_distortion_pct=0.0,
        random_state=123,
    )
    assert filled_min_5.loc[0, 'feat'] == expected_from_pool_5


def test_distortion_is_bounded_by_max_distortion_pct():
    idx = np.arange(0, 60)
    exists = np.ones_like(idx, dtype=bool)
    exists[25:35] = False  # gap of length 10

    df = pd.DataFrame({'exists': exists, 'feat': np.full_like(idx, 100.0, dtype=float)}, index=idx)

    filled = fill_short_gaps_per_column(
        df,
        feature_cols=['feat'],
        min_donor_n_segs=10,
        max_distortion_pct=0.05,
        random_state=42,
    )

    gap_vals = filled.loc[25:34, 'feat'].to_numpy()
    assert np.all(gap_vals >= 95.0)
    assert np.all(gap_vals <= 105.0)
    assert np.any(gap_vals != 100.0)


def test_invalid_parameters_raise_value_error():
    df = pd.DataFrame({'exists': [False, True, True], 'feat': [np.nan, 1.0, 1.0]}, index=[0, 1, 2])

    try:
        fill_short_gaps_per_column(df, ['feat'], min_donor_n_segs=-1)
        raise AssertionError('Expected ValueError for negative min_donor_n_segs')
    except ValueError as e:
        assert 'min_donor_n_segs' in str(e)

    try:
        fill_short_gaps_per_column(df, ['feat'], max_distortion_pct=-0.1)
        raise AssertionError('Expected ValueError for negative max_distortion_pct')
    except ValueError as e:
        assert 'max_distortion_pct' in str(e)


def test_fill_short_gaps_no_missing_rows_keeps_structure_unchanged():
    df = pd.DataFrame({'exists': [True, True, True], 'feat': [1.0, 2.0, 3.0]}, index=[0, 1, 2])

    out = fill_short_gaps_per_column(df, ['feat'], max_distortion_pct=0.0, random_state=1)

    assert list(out.columns) == ['exists', 'feat']
    assert out.equals(df)


def test_fill_ptnt_gaps_reintegrates_chunks_and_keeps_long_gaps_nan():
    idx = np.arange(0, 10)
    df = pd.DataFrame(
        {
            'start_mtz': pd.date_range('2024-01-01', periods=10, freq='10min'),
            'exists': [True, True, False, True, True, False, False, False, True, True],
            'feat': [10.0, 20.0, np.nan, 40.0, 50.0, np.nan, np.nan, np.nan, 80.0, 90.0],
        },
        index=idx,
    )

    out = fill_ptnt_gaps(
        df,
        long_gap_min_segs=3,
        feature_cols=['feat'],
        min_donor_n_segs=1,
        max_distortion_pct=0.0,
        random_state=123,
    )

    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(df.index)
    assert 'exists' in out.columns
    assert 'filled' not in out.columns

    # Short gap at index 2 is filled.
    assert not bool(out.loc[2, 'exists'])
    assert not pd.isna(out.loc[2, 'feat'])

    # Long gap at indices 5-7 remains in structure and stays NaN.
    assert (~out.loc[5:7, 'exists']).all()
    assert out.loc[5:7, 'feat'].isna().all()


