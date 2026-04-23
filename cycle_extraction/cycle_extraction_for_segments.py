import itertools
import logging
import multiprocessing
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, Timedelta, Index
from scipy.stats import false_discovery_control

from config import PatientDir, PATHS, save_dataframe_multiformat
from config.constants import UPPER_QUANTILE_BOUND_FOR_FEATURE_CLIPPING
from config.intervals import SEGMENT
from cycle_extraction import compute_plv_for_split_signal, rayleigh_test
from cycle_extraction.cycle_functions import nc_filter_multidien, plot_filtered_feature, \
    plot_phase_histogram_for_all_features
from cycle_extraction.circular_comparison_functions import watson_wheeler_test, permutation_test
from feature_extraction.extract_features import FeatureNames
from preprocessing.dataset_partitioning import partition_dataframe
from utils.utils import timeit


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


# noinspection PyTypeChecker
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


def _compute_plv_metrics_for_ptnt(
        seg_features: DataFrame,
        event_timestamps: dict[str, np.ndarray],
        feature_names: list[str],
        upper_quantile_bound_for_clipping: float,
        patient: str,
):
    """Compute PLV, Rayleigh, and Benjamini-Hochberg metrics per event type.

    Returns
    -------
    metrics: DataFrame
        Index: feature_names. Columns: MultiIndex(event_type, [plv, mean_angle, mean_angle_deg, n_events, p_rayleigh, z_stat, p_rayleigh_bh])
    seg_feats_filt: DataFrame
        Filtered features with NaN gaps preserved.
    event_phases_per_type_per_feat: dict
        Phases keyed by event_type, then feature.
    """
    seg_feats = seg_features.copy()
    # Create end_mtz column
    end_mtz = seg_feats['start_mtz'] + SEGMENT.exact_dur
    seg_feats.insert(seg_feats.columns.get_loc("start_mtz") + 1, "end_mtz", end_mtz)

    # ---- Handle outliers and normalize features
    for feat in feature_names:
        h = handle_feature_outliers(seg_feats[feat].values, upper_quantile_bound_for_clipping)
        seg_feats[feat] = normalize_feature(h)

    # ---- Split by long gaps (where the features are still NA)
    chunked_segs = split_dataframe_by_nan_gaps(seg_feats, feature_names)
    logging.info(f'[{patient}] Number of chunks: {len(chunked_segs)}')

    # ---- Filter the features per chunk
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

    # ---- Assign events to chunks and get their relative indices in the chunks
    event_idxs_per_type_per_chunk = {k: get_event_indices_per_chunk(chunked_segs, timestamps)
                                     for k, timestamps in event_timestamps.items()}

    # ---- Compute Phase Locking Values (PLV) and related metrics
    event_types = list(event_idxs_per_type_per_chunk.keys())
    base_metrics = ['plv', 'mean_angle', 'mean_angle_deg', 'n_events', 'p_rayleigh', 'z_stat']
    all_metrics = [*base_metrics, 'p_rayleigh_bh']
    cols = pd.MultiIndex.from_product([event_types, all_metrics], names=['event_type', 'metric'])
    metrics = DataFrame(index=Index(feature_names, name='feature'), columns=cols, dtype='float64')

    event_phases_per_type_per_feat = {e: {} for e in event_types}
    for event_type, event_idxs_per_chunk in event_idxs_per_type_per_chunk.items():
        n_events = sum(len(ix) for ix in event_idxs_per_chunk)

        for feat in feature_names:
            feat_per_chunk = [filt_chunk[feat].values for filt_chunk in filtered_chunks]

            plv, mean_angle, mean_angle_deg, event_phases = compute_plv_for_split_signal(feat_per_chunk,
                                                                                         event_idxs_per_chunk)
            p_rayleigh, z_stat = rayleigh_test(n_events, plv)

            event_phases_per_type_per_feat[event_type][feat] = event_phases
            metrics.loc[feat, (event_type, base_metrics)] = [plv, mean_angle, mean_angle_deg, n_events, p_rayleigh,
                                                             z_stat]

        # Benjamini-Hochberg False Discovery Rate Control because of multiple p-values (p-value per feature).
        ps_rayleigh = metrics.loc[:, (event_type, 'p_rayleigh')]
        ps_bh = false_discovery_control(ps_rayleigh, method='bh')
        metrics.loc[:, (event_type, 'p_rayleigh_bh')] = ps_bh

    return metrics, seg_feats_filt, event_phases_per_type_per_feat


def _compute_circular_comparisons_for_ptnt(
        event_phases_per_type_per_feat: dict[str, dict[str, np.ndarray]],
        feature_names: list[str],
):
    """Compute Watson-Wheeler test and pairwise circular comparisons with bootstrapping.

    Returns
    -------
    circular_comp_results: DataFrame
        Index: feature_names. Columns: MultiIndex(comparison_name, [metric_key])
    """
    event_types = list(event_phases_per_type_per_feat.keys())
    event_pairs = list(itertools.combinations(event_types, 2))
    event_pairs = [(e1, e2, f"{e1} vs {e2}") for e1, e2 in event_pairs]  # add name for pair
    ww_metrics = ['w_stat', 'p_value', 'p_ww_bh']  # ww = watson wheeler (test)
    pair_metrics_no_bh = ['obs_mean1', 'obs_mean2', 'observed_diff', 'lower1', 'upper1', 'lower2', 'upper2', 'ci_lower',
                          'ci_upper', 'p_value']
    pair_metrics = pair_metrics_no_bh + ['p_perm_bh']

    # Build DataFrame with MultiIndex columns
    col_tuples = [('watson_wheeler', metric) for metric in ww_metrics]
    for _, _, pair_name in event_pairs:
        col_tuples.extend((pair_name, metric) for metric in pair_metrics)

    cols = pd.MultiIndex.from_tuples(col_tuples, names=['comparison', 'metric'])
    res = DataFrame(index=Index(feature_names, name='feature'), columns=cols, dtype='float64')

    # Compute Results
    for feat in feature_names:
        phases_per_type = {ev_type: per_feat[feat] for ev_type, per_feat in event_phases_per_type_per_feat.items()}
        # Watson-Wheeler test across all event types
        res.loc[feat, ('watson_wheeler', ['w_stat', 'p_value'])] = watson_wheeler_test(list(phases_per_type.values()))

        # Pairwise circular comparisons
        for e1, e2, pair_name in event_pairs:
            ph1, ph2 = phases_per_type[e1], phases_per_type[e2]
            comp = permutation_test(ph1, ph2, n_iters=5000, ci_level=95)
            res.loc[feat, (pair_name, pair_metrics_no_bh)] = [comp[key] for key in pair_metrics_no_bh]

    # Benjamini-Hochberg FDR correction per comparison across features
    pvals_ww = res.loc[:, ('watson_wheeler', 'p_value')]
    res.loc[:, ('watson_wheeler', 'p_ww_bh')] = false_discovery_control(pvals_ww, method='bh')

    for _, _, pair_name in event_pairs:
        pvals_pair = res.loc[:, (pair_name, 'p_value')]
        res.loc[:, (pair_name, 'p_perm_bh')] = false_discovery_control(pvals_pair, method='bh')

    return res


def cycle_extraction_for_ptnt(
        seg_features: DataFrame,
        event_timestamps: dict[str, np.ndarray],
        upper_quantile_bound_for_clipping: float = UPPER_QUANTILE_BOUND_FOR_FEATURE_CLIPPING,
        feature_names: list[str] = FeatureNames.ALL_ORDERED,
        patient: str = 'unknown patient',
):
    """Compute PLV metrics and circular comparisons per patient.

    Parameters
    ----------
    seg_features
        Features per segment and start_mtz
    event_timestamps
        Values are arrays of event `Timestamps` (e.g., seizure starts, False positive predictions of models).
        Keys are the event type/origin (e.g., seizures, ensemble, CNN).
    upper_quantile_bound_for_clipping
    feature_names
    patient
        For logging purposes

    Returns
    -------
    metrics: DataFrame
        PLV and related metrics per event type.
    circular_comp_results: DataFrame
        Watson-Wheeler and pairwise circular comparisons.
    seg_feats_filt: DataFrame
        Filtered features per segment with gaps preserved.
    event_phases_per_type_per_feat: dict
        Phases per event type per feature (for plotting).
    """
    metrics, seg_feats_filt, event_phases_per_type_per_feat = _compute_plv_metrics_for_ptnt(
        seg_features, event_timestamps, feature_names, upper_quantile_bound_for_clipping, patient
    )
    circular_comp_results = _compute_circular_comparisons_for_ptnt(event_phases_per_type_per_feat, feature_names)

    return metrics, circular_comp_results, seg_feats_filt, event_phases_per_type_per_feat


def get_false_positives_from_clips(clips: DataFrame, thresh: float, score_col: str):
    c = clips[clips['valid']]
    negative_label = ~c['preictal']
    positive_pred = c[score_col] >= thresh
    fp_clips = c[negative_label & positive_pred]
    fp_timestamps = fp_clips['end_mtz']
    return fp_timestamps.to_numpy()


# noinspection PyTypeChecker
def _make_filtered_feature_plots(
        seg_feats: DataFrame,
        seg_feats_filt: DataFrame,
        event_timestamps: dict,
        feature_names: list[str], save_dir: Path,
        test_start_mtz: pd.Timestamp
):
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

        # Mark test split start on both panels for orientation in timeline plots.
        for i, ax in enumerate(fig.axes):
            label = 'test_start' if i == 0 else '_nolegend_'
            ax.axvline(test_start_mtz, color='tab:green', linestyle='--', label=label, ymin=-0.02, ymax=1.02,
                       clip_on=False)
        fig.axes[0].legend(loc='upper left')

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
        metrics_as_dict = [m[k].to_dict() for k in ['plv', 'mean_angle', 'p_rayleigh_bh']]

        fig = plot_phase_histogram_for_all_features(
            event_phases_per_feat,
            *metrics_as_dict,
            n_bins=24,
        )

        fig.suptitle(f'Phase distribution compared to features: {patient} - {event_type}', y=0.98, fontsize=16, )
        fig.tight_layout(h_pad=3.0, rect=[0, 0, 1, 0.97])

        save_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_dir / f'{event_type}.pdf')
        plt.close(fig)


# noinspection PyTypeChecker
@timeit(arg_indices=[0])
def cycle_extraction_and_plot_for_pdir(
        pdir: PatientDir,
        feature_names: list[str] = FeatureNames.ALL_ORDERED,
        models: tuple[str] = ('CNN', 'ensemble'),
        upper_quantile_bound_for_clipping: float = UPPER_QUANTILE_BOUND_FOR_FEATURE_CLIPPING,
):
    logging.info(f'[{pdir.name}] 🚀 Starting Cycle Extraction...')

    # Load test clips (since model scores are extracted from these) and features and seizures for train and test.
    szrs = pd.read_pickle(pdir.valid_szr_starts_file.pickle)['start_mtz'].values
    seg_feats = pd.read_pickle(pdir.filled_features_for_segs.pickle).drop(columns=['exists'], errors='ignore')
    test_clips = partition_dataframe(pd.read_pickle(pdir.clip_scores_table.pickle), pdir)['test']
    test_start_mtz = test_clips['start_mtz'].iloc[0]

    # Get the false positive timestamps for each model
    event_timestamps = {'seizures': szrs}
    for model in models:
        best_thresh = pd.read_pickle(pdir.model_eval_subdir('test', model).metrics_table.pickle)['best_threshold']
        event_timestamps[f'{model} FPs'] = get_false_positives_from_clips(test_clips, best_thresh, f'{model}_score')

    metrics, circular_comp_results, seg_feats_filt, event_phases_per_type_per_feat = \
        cycle_extraction_for_ptnt(seg_feats, event_timestamps, upper_quantile_bound_for_clipping, feature_names,
                                  patient=pdir.name)

    # Save Metrics and Plots
    logging.info(f'[{pdir.name}] 🎨 Saving Results and Making Cycle Extraction Figures...')
    save_dataframe_multiformat(metrics, pdir.cycle_extraction_metrics_table,
                               save_index=True, formats=['pickle', 'xlsx'])
    save_dataframe_multiformat(circular_comp_results, pdir.circular_comparison_table,
                               save_index=True, formats=['pickle', 'xlsx'])

    _make_filtered_feature_plots(seg_feats, seg_feats_filt, event_timestamps, feature_names,
                                 pdir.filtered_feature_plots_dir, test_start_mtz)
    _make_phase_histogram_plots(event_phases_per_type_per_feat, metrics, pdir.circular_histograms_dir, pdir.name)

    logging.info(f'[{pdir.name}] ✅ Completed Cycle Extraction and Figures.')

    return metrics, circular_comp_results


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


def aggregate_results_per_pdir(pdirs: list[PatientDir]):
    cycle_extr = {pdir.name: pd.read_pickle(pdir.cycle_extraction_metrics_table.pickle) for pdir in pdirs}
    circ_comp = {pdir.name: pd.read_pickle(pdir.circular_comparison_table.pickle) for pdir in pdirs}
    cycle_extr_df = pd.concat(cycle_extr, names=['patient', 'feature'])
    circ_comp_df = pd.concat(circ_comp, names=['patient', 'feature'])
    save_dataframe_multiformat(cycle_extr_df, PATHS.cycle_extraction_metrics_per_ptnt_table, formats=['pickle', 'xlsx'],
                               save_index=True)
    save_dataframe_multiformat(circ_comp_df, PATHS.circular_comparison_per_ptnt_table, formats=['pickle', 'xlsx'],
                               save_index=True)
    return cycle_extr_df, circ_comp_df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    pdirs_ = PATHS.patient_dirs()
    cycle_extraction_for_pdirs(pdirs_, serial_processing=False)
    aggregate_results_per_pdir(PATHS.patient_dirs(include_fake_ptnts=False))
