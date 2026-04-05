import logging
import multiprocessing
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, Timedelta
from scipy.stats import false_discovery_control

from config import PatientDir, PATHS, pickle_path, save_dataframe_multiformat
from config.constants import UPPER_QUANTILE_BOUND_FOR_FEATURE_CLIPPING
from config.intervals import SEGMENT
from cycle_extraction import compute_plv_for_split_signal, rayleigh_test
from cycle_extraction.cycle_functions import nc_filter_multidien, plot_filtered_feature, \
    plot_phase_histogram_for_all_features
from feature_extraction.extract_features import FeatureNames
from model_eval.event_based_metrics import load_data_per_split


def map_events_to_interval_index(events: np.ndarray,
                                 interval_starts: np.ndarray,
                                 interval_ends: np.ndarray
                                 ) -> np.ndarray:
    """
    Map seizures to the interval index.
    :param events: event (e.g., seizure) timestamps (1D array).
    :param interval_starts: Start timestamps for non-overlapping intervals (1D array).
    :param interval_ends: End timestamps for non-overlapping intervals (1D array).
    :return: Interval index per seizure as ndarray (-1 if no interval contains it).
    """
    # Reshape for broadcasting: (n_seizures, 1) and (1, n_intervals)
    szr_values = events[:, np.newaxis]
    starts_broadcast = interval_starts[np.newaxis, :]
    ends_broadcast = interval_ends[np.newaxis, :]

    # Check if seizure is in interval: start <= seizure < end
    in_interval = (starts_broadcast <= szr_values) & (szr_values < ends_broadcast)

    # Find the interval index for each seizure (-1 if not found)
    interval_indices = np.where(in_interval.any(axis=1),
                                in_interval.argmax(axis=1),
                                -1)

    return interval_indices


def get_event_indices_per_chunk(chunked_intervals: list[DataFrame], events: np.ndarray):
    """Assign events (Timestamps) to chunks and get their relative indices in the chunks"""
    idxs_per_chunk = []
    total_events_in_chunks = 0
    for chunk in chunked_intervals:
        starts, ends = chunk['start_mtz'].values, chunk['end_mtz'].values
        chunk_start = starts[0]
        chunk_end = ends[-1]

        events_in_chunk = events[(chunk_start <= events) & (events <= chunk_end)]
        szr_idxs_in_chunk = map_events_to_interval_index(events_in_chunk, starts, ends)
        idxs_per_chunk.append(szr_idxs_in_chunk)
        total_events_in_chunks += len(events_in_chunk)

    assert total_events_in_chunks == len(events), \
        (f"Total number of seizures ({len(events)}) does not match the number of seizures in chunks "
         f"({total_events_in_chunks}). This might mean there are seizures during a large recording gap.")

    return idxs_per_chunk


def split_dataframe_by_nan_gaps(df: DataFrame, relevant_cols: Optional[list[str]] = None):
    """Index is not reset!"""
    cols = relevant_cols or df.columns
    # gap row = all features NA
    gaps = df[cols].isna().all(axis=1)
    # Assign group IDs to groups of continuous gaps/existing data
    group_id = gaps.ne(gaps.shift(fill_value=False)).cumsum()
    # Include chunks that aren't NA
    chunks = [chunk for _, chunk in df.groupby(group_id) if not chunk[cols].isna().to_numpy().all()]
    return chunks


def handle_feature_outliers(f: np.ndarray, upper_quantile_bound_for_clipping: float) -> np.ndarray:
    """Handle feature outliers by clipping them while handling NaNs."""
    # Get the value that corresponds to the upper bound of the quantile
    q = np.nanquantile(f, upper_quantile_bound_for_clipping)
    # Clip the extremely high values
    c = np.clip(f, a_min=None, a_max=q)
    return c


def normalize_feature(f: np.ndarray):
    """Normalize the feature while handling NaNs."""
    mu = np.nanmean(f)
    sigma = np.nanstd(f)
    res = f - mu
    if sigma != 0:
        res /= sigma
    return res


def cycle_extraction_for_ptnt(
        seg_features: DataFrame,
        event_timestamps: dict[str, np.ndarray],
        upper_quantile_bound_for_clipping: float = UPPER_QUANTILE_BOUND_FOR_FEATURE_CLIPPING,
        feature_names: list[str] = FeatureNames.ALL_ORDERED,
        patient: str = 'unknown patient',
):
    """

    Parameters
    ----------
    seg_features
        Features per segment and start_mtz
    event_timestamps
        Values are arrays of event ``Timestamps`` (e.g., seizures starts, False positive predictions of models).
        Keys are the event type/origin (e.g., seizures, ensemble, CNN).
    upper_quantile_bound_for_clipping
    feature_names
    patient
        For logging purposes

    Returns
    -------
    metrics: DataFrame,
    filtered features per segment with gaps: DataFrame,
    event phases per event type per feature: dict[str, dict[str, np.ndarray]]
    """
    seg_feats = seg_features.copy()
    # Create end_mtz column
    end_mtz = seg_feats['start_mtz'] + SEGMENT.exact_dur
    seg_feats.insert(seg_feats.columns.get_loc("start_mtz") + 1, "end_mtz", end_mtz)

    # Handle outliers and normalize features
    for feat in feature_names:
        h = handle_feature_outliers(seg_feats[feat].values, upper_quantile_bound_for_clipping)
        seg_feats[feat] = normalize_feature(h)

    # Split by long gaps (where the features are still NA)
    chunked_segs = split_dataframe_by_nan_gaps(seg_feats, feature_names)
    logging.info(f'[{patient}] Number of chunks: {len(chunked_segs)}')

    # Filter the features per chunk
    segs_per_hour = Timedelta(hours=1) / SEGMENT.exact_dur
    filtered_chunks = []
    # for plotting - single DataFrame equivalent to filtered_chunk, but with NaN gaps
    seg_feats_filt = seg_feats[['start_mtz', 'end_mtz']].copy()

    for chunk in chunked_segs:
        filt_chunk = chunk[['start_mtz', 'end_mtz']].reset_index(drop=True)

        for feat in feature_names:
            # Apply the filter
            x = chunk[feat].values
            filt_chunk[feat] = nc_filter_multidien(x, min_period=5 * 24, max_period=50 * 24, fs=segs_per_hour, order=10)

        filtered_chunks.append(filt_chunk)
        seg_feats_filt.loc[chunk.index, feature_names] = filt_chunk[feature_names].values

    # Assign events to chunks and get their relative indices in the chunks
    event_idxs_per_type_per_chunk = {k: get_event_indices_per_chunk(chunked_segs, timestamps)
                                     for k, timestamps in event_timestamps.items()}

    # Compute Phase Locking Values (PLV) and related metrics
    event_types = list(event_idxs_per_type_per_chunk.keys())
    base_metrics = ['plv', 'mean_angle', 'mean_angle_deg', 'n_events', 'p_value', 'z_stat']
    all_metrics = [*base_metrics, 'p_value_bh']
    cols = pd.MultiIndex.from_product([event_types, all_metrics], names=['event_type', 'metric'])
    metrics = DataFrame(index=feature_names, columns=cols, dtype='float64')

    event_phases_per_type_per_feat = {e: {} for e in event_types}  # for plotting circular histogram
    for event_type, event_idxs_per_chunk in event_idxs_per_type_per_chunk.items():
        n_events = sum(len(ix) for ix in event_idxs_per_chunk)

        for feat in feature_names:
            feat_per_chunk = [filt_chunk[feat].values for filt_chunk in filtered_chunks]

            plv, mean_angle, mean_angle_deg, event_phases = compute_plv_for_split_signal(feat_per_chunk,
                                                                                         event_idxs_per_chunk)
            p_value, z_stat = rayleigh_test(n_events, plv)

            # Store values
            event_phases_per_type_per_feat[event_type][feat] = event_phases
            metrics.loc[feat, (event_type, base_metrics)] = [plv, mean_angle, mean_angle_deg, n_events, p_value, z_stat]

        # Benjamini-Hochberg False Discovery Rate Control because of multiple p-values (p-value per feature).
        p_vals = metrics.loc[:, (event_type, 'p_value')]
        p_vals_adj = false_discovery_control(p_vals, method='bh')
        metrics.loc[:, (event_type, 'p_value_bh')] = p_vals_adj

    return metrics, seg_feats_filt, event_phases_per_type_per_feat


def get_false_positives_from_clips(clips: DataFrame, thresh: float, score_col: str):
    c = clips[clips['valid']]
    negative_label = ~c['preictal']
    positive_pred = c[score_col] >= thresh
    fp_clips = c[negative_label & positive_pred]
    fp_timestamps = fp_clips['end_mtz']
    return fp_timestamps.to_numpy()


def _make_filtered_feature_plots(seg_feats: DataFrame, seg_feats_filt: DataFrame, event_timestamps: dict,
                                 feature_names: list[str], save_dir: Path):
    seg_starts = seg_feats['start_mtz'].values
    assert (seg_starts == seg_feats_filt['start_mtz'].values).all(), 'Start times do not match.'
    assert (seg_feats.index == seg_feats_filt.index).all(), 'Segment indices do not match.'

    seg_ends = seg_feats_filt['end_mtz'].values
    idxs_per_event_type = {name: map_events_to_interval_index(timestamps, seg_starts, seg_ends)
                           for name, timestamps in event_timestamps.items()}

    for feat in feature_names:
        original = seg_feats[feat].values
        filtered = seg_feats_filt[feat].values
        assert (np.all(np.isnan(original) == np.isnan(filtered))), \
            'NaN values in filtered features do not match original features.'

        fig = plot_filtered_feature(
            original, filtered,
            samples_per_hour=Timedelta(hours=1) / SEGMENT.exact_dur,
            time=seg_starts,
            events=idxs_per_event_type,
        )
        fig.suptitle(feat, x=0.1)

        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f'{feat}.png')


def _make_phase_histogram_plots(
        event_phases_per_type_per_feat: dict,
        metrics: DataFrame,
        save_dir: Path,
        patient: str,
):
    # Make a figure per event.
    for event_type, event_phases_per_feat in event_phases_per_type_per_feat.items():
        m = metrics.loc[:, event_type]  # Metrics for this event type
        metrics_as_dict = [m[k].to_dict() for k in ['plv', 'mean_angle', 'p_value_bh']]

        fig = plot_phase_histogram_for_all_features(
            event_phases_per_feat,
            *metrics_as_dict,
            n_bins=24,
        )

        fig.suptitle(f'{patient} - {event_type}: Phase distribution of seizure occurrences', y=0.98, fontsize=16, )
        fig.tight_layout(h_pad=3.0, rect=[0, 0, 1, 0.97])

        save_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_dir / f'{event_type}.pdf')
        plt.close(fig)


def cycle_extraction_and_plot_for_pdir(
        pdir: PatientDir,
        feature_names: list[str] = FeatureNames.ALL_ORDERED,
        models: tuple[str] = ('CNN', 'ensemble'),
        upper_quantile_bound_for_clipping: float = UPPER_QUANTILE_BOUND_FOR_FEATURE_CLIPPING,
):
    logging.info(f'[{pdir.name}] 🚀 Starting Cycle Extraction...')

    # Load test data and discard irrelevant columns
    per_split = load_data_per_split(pdir)
    szrs = per_split['szr_starts']['test']
    clips = per_split['clips']['test']

    # Get features for just test set
    first_test_seg_idx = clips.iloc[0]['start_seg']
    _sf = pd.read_pickle(pdir.filled_features_for_segs.pickle).drop(columns=['exists'], errors='ignore')
    seg_feats = _sf.loc[first_test_seg_idx:].reset_index(drop=True)

    # Get the false positive timestamps for each model
    event_timestamps = {'seizures': szrs}
    for model in models:
        best_thresh = pd.read_pickle(pickle_path(pdir.model_eval_dir / 'test' / model / 'metrics'))['best_threshold']
        event_timestamps[f'{model} FPs'] = get_false_positives_from_clips(clips, best_thresh, f'{model}_score')

    metrics, seg_feats_filt, event_phases_per_type_per_feat = \
        cycle_extraction_for_ptnt(seg_feats, event_timestamps, upper_quantile_bound_for_clipping, feature_names,
                                  patient=pdir.name)

    logging.info(f'[{pdir.name}] 🎨 Making Cycle Extraction Figures...')
    # Save Metrics and Plots
    save_dataframe_multiformat(metrics, pdir.cycle_extraction_results_table, save_index=True,
                               csv_kwargs={'float_format': '%.3f'})
    _make_filtered_feature_plots(seg_feats, seg_feats_filt, event_timestamps, feature_names,
                                 pdir.filtered_feature_plots_dir)
    _make_phase_histogram_plots(event_phases_per_type_per_feat, metrics, pdir.circular_histograms_dir, pdir.name)

    logging.info(f'[{pdir.name}] ✅ Completed Cycle Extraction and Figures.')


def cycle_extraction_for_pdirs(
        pdirs: list[PatientDir],
        serial_processing: bool = False,
):
    if serial_processing:
        for pdir in pdirs:
            cycle_extraction_and_plot_for_pdir(pdir)
    else:
        with multiprocessing.Pool() as pool:
            pool.map(cycle_extraction_and_plot_for_pdir, pdirs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    pdirs_ = PATHS.patient_dirs()
    cycle_extraction_for_pdirs(pdirs_, serial_processing=False)
