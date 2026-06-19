import logging
import multiprocessing
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, Timedelta

from config import PatientDir, PATHS
from config.intervals import SEGMENT
from cycle_extraction.cycle_functions import plot_filtered_feature, plot_phase_histogram_for_all_features
from cycle_extraction.cycle_extraction_for_segments import map_events_to_interval_index
from feature_extraction.extract_features import FeatureNames
from written_thesis import matplotlib_style
from written_thesis.helpers import PRETTY_FEATURE_NAMES_MAP


# noinspection PyTypeChecker
def make_filtered_feature_plots(
        seg_feats: DataFrame,
        seg_feats_filt: DataFrame,
        event_timestamps: dict,
        feature_names: list[str],
        save_dir: Path,
        test_start_mtz: pd.Timestamp
):
    matplotlib_style.apply_style(use_small_font=True)

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
            subplots_kwargs={'figsize': matplotlib_style.latex_figsize(height_ratio=0.65)},
        )

        # Mark test split start on both panels for orientation in timeline plots.
        for i, ax in enumerate(fig.axes):
            label = 'start of test set' if i == 0 else '_nolegend_'
            ax.axvline(test_start_mtz, color='tab:green', linestyle='--', label=label, ymin=-0.02, ymax=1.02,
                       clip_on=False)
            ax.margins(x=0)

        fig.axes[0].legend(loc='upper left')

        fig.tight_layout(pad=0, h_pad=1)
        # fig.suptitle(PRETTY_FEATURE_NAMES_MAP[feat], x=0, y=1, fontweight='bold', ha='left', va='top')
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f'{PRETTY_FEATURE_NAMES_MAP[feat]}.pdf')


def _make_phase_histogram_plots(
        event_phases_per_type_per_feat: dict,
        metrics: DataFrame,
        save_dir: Path,
        patient: str,
):
    matplotlib_style.apply_style(use_small_font=True)
    # Make a figure per event.
    save_dir.mkdir(exist_ok=True, parents=True)
    for event_type, event_phases_per_feat in event_phases_per_type_per_feat.items():
        m = metrics.loc[:, event_type]  # Metrics for this event type
        metrics_as_dict = [m[k].to_dict() for k in ['plv', 'mean_angle', 'p_rayleigh_bh']]

        fig = plot_phase_histogram_for_all_features(
            event_phases_per_feat,
            *metrics_as_dict,
            n_bins=24,
            subplots_kwargs={'figsize': matplotlib_style.latex_figsize(height_ratio=0.75)},
        )
        fig.tight_layout(pad=0, w_pad=0.8)
        fig.savefig(save_dir / f'{event_type}.pdf')
        plt.close(fig)


def cycle_extraction_plots_for_pdir(
        pdir: PatientDir,
        feature_names: list[str] = FeatureNames.ALL_ORDERED,
):
    logging.info(f'[{pdir.name}] 🎨 Making Cycle Extraction Plots...')

    # Load Data
    seg_feats = pd.read_pickle(pdir.filled_features_for_segs.pickle).drop(columns=['exists'], errors='ignore')
    metrics = pd.read_pickle(pdir.cycle_extraction_metrics_table.pickle)
    seg_feats_filt = pd.read_pickle(pdir.filtered_features_for_segs.pickle)
    with pdir.event_phases_per_type_per_feat.open('rb') as f:
        event_phases_per_type_per_feat = pickle.load(f)
    with pdir.event_timestamps_dict.open('rb') as f:
        event_timestamps = pickle.load(f)
    test_start_mtz = pd.read_pickle(pdir.dataset_partition.pickle).loc['test', 'start_mtz']

    # Make Plots
    make_filtered_feature_plots(seg_feats, seg_feats_filt, event_timestamps, feature_names,
                                pdir.filtered_feature_plots_dir, test_start_mtz)
    _make_phase_histogram_plots(event_phases_per_type_per_feat, metrics, pdir.phase_histograms_dir, pdir.name)

    logging.info(f'[{pdir.name}] ✅ Completed Cycle Extraction Plots.')


def cycle_extraction_plots_for_pdirs(pdirs: list[PatientDir], serial_processing: bool = False):
    if serial_processing:
        for pdir in pdirs:
            cycle_extraction_plots_for_pdir(pdir)
    else:
        with multiprocessing.Pool() as pool:
            pool.map(cycle_extraction_plots_for_pdir, pdirs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    pdirs_ = PATHS.patient_dirs()
    cycle_extraction_plots_for_pdirs(pdirs_, serial_processing=False)
