from pathlib import Path
from shutil import copy2

import numpy as np
import pandas as pd
import pytest
from pyedflib import FILETYPE_EDFPLUS, EdfWriter

from config.constants import CHANNELS, SAMPLING_FREQUENCY_HZ, N_CHANNELS
from config.intervals import SEGMENT
from feature_extraction.extract_features import FeatureNames, extract_ptnt_features


def _write_two_channel_edf(file_path: Path, signals: np.ndarray) -> None:
    """Write a tiny 2-channel EDF used by the test."""
    writer = EdfWriter(str(file_path), n_channels=N_CHANNELS, file_type=FILETYPE_EDFPLUS)
    try:
        headers = []
        for channel_idx, label in enumerate(CHANNELS):
            sig = signals[channel_idx]
            physical_min = float(sig.min()) - 1.0
            physical_max = float(sig.max()) + 1.0
            if physical_max <= physical_min:
                physical_max = physical_min + 1.0

            headers.append({
                "label": label,
                "dimension": "uV",
                "sample_frequency": float(SAMPLING_FREQUENCY_HZ),
                "physical_min": physical_min,
                "physical_max": physical_max,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            })

        writer.setSignalHeaders(headers)
        writer.writeSamples([signals[0], signals[1]])
    finally:
        writer.close()


@pytest.fixture
def synthetic_patient_input(tmp_path):
    """Create synthetic EDF + segments table with explicit artifact gaps."""
    data_dir = tmp_path / "synthetic_edf_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    n_segs_total = 18
    seg_n = SEGMENT.n_samples
    total_n = n_segs_total * seg_n

    t = np.arange(total_n, dtype=float) / float(SAMPLING_FREQUENCY_HZ)

    # Three clean periods separated by artifact periods.
    ch_d = np.zeros(total_n, dtype=float)
    ch_p = np.zeros(total_n, dtype=float)

    def put_period(seg_start: int, seg_end: int, freq_hz: float, amp_d: float, amp_p: float, phase: float = 0.0):
        start = seg_start * seg_n
        end = seg_end * seg_n
        tt = t[start:end]
        ch_d[start:end] = amp_d * np.sin(2 * np.pi * freq_hz * tt)
        ch_p[start:end] = amp_p * np.sin(2 * np.pi * freq_hz * tt + phase)

    # Clean A: low amplitude, slow oscillation (delta-heavy)
    put_period(seg_start=0, seg_end=5, freq_hz=1.0, amp_d=10.0, amp_p=9.0, phase=0.05)
    # Artifact gap 1: huge-amplitude oscillation
    put_period(seg_start=5, seg_end=7, freq_hz=12.0, amp_d=5000.0, amp_p=5200.0, phase=0.6)

    # Clean B: higher amplitude, faster oscillation (theta-heavy)
    put_period(seg_start=7, seg_end=12, freq_hz=6.0, amp_d=30.0, amp_p=27.0, phase=0.08)
    # Artifact gap 2: huge-amplitude oscillation
    put_period(seg_start=12, seg_end=13, freq_hz=18.0, amp_d=7000.0, amp_p=6800.0, phase=1.1)

    # Clean C: medium amplitude, mid frequency
    put_period(seg_start=13, seg_end=18, freq_hz=3.0, amp_d=15.0, amp_p=13.0, phase=0.03)

    edf_path = data_dir / "synthetic_patient.edf"
    _write_two_channel_edf(edf_path, np.vstack([ch_d, ch_p]))

    # Persist a copy in the repo so you can inspect the file after test runs.
    preview_dir = Path(__file__).parent / "synthetic_edf_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    copy2(edf_path, preview_dir / edf_path.name)

    seg_ids = np.arange(n_segs_total)
    artifact_segs = {5, 6, 12}
    exists = np.array([seg not in artifact_segs for seg in seg_ids], dtype=bool)

    file_col = np.full(n_segs_total, None, dtype=object)
    file_col[exists] = edf_path.name
    start_index_col = np.where(exists, seg_ids * seg_n, np.nan)

    segs = pd.DataFrame({
        "exists": exists,
        "file": file_col,
        "start_index": start_index_col,
    })

    return segs, data_dir


class TestExtractPtntFeatures:
    def test_skips_artifact_segments_and_maps_features_correctly(self, synthetic_patient_input):
        segs, edf_dir = synthetic_patient_input

        out = extract_ptnt_features(segs.copy(), edf_dir=edf_dir, serial_processing=True)

        feature_cols = FeatureNames.ALL_ORDERED
        assert all(col in out.columns for col in feature_cols)

        artifact_mask = ~out["exists"]
        clean_mask = out["exists"]

        # Artifact rows were intentionally marked missing and must stay unassigned.
        assert out.loc[artifact_mask, feature_cols].isna().all().all()

        # Existing segments should all receive computed features.
        # noinspection PyUnresolvedReferences
        assert out.loc[clean_mask, feature_cols].notna().all().all()

        # Loose sanity checks across clean periods.
        period_a = out.loc[0:4]      # 1 Hz, low amplitude
        period_b = out.loc[7:11]     # 6 Hz, high amplitude
        period_c = out.loc[13:17]    # 3 Hz, medium amplitude

        # Correlation should be strongly positive in clean periods (similar channels).
        assert period_a["corrcoef"].median() > 0.95
        assert period_b["corrcoef"].median() > 0.95
        assert period_c["corrcoef"].median() > 0.95

        # Variance should reflect amplitude differences (B > C > A roughly).
        var_a = period_a["var_D"].median()
        var_b = period_b["var_D"].median()
        var_c = period_c["var_D"].median()
        assert var_b > var_c > var_a

        # Spectral sanity: 1 Hz period is delta-heavy; 6 Hz period is theta-heavy.
        assert period_a["Delta_D"].median() > period_a["Theta_D"].median()
        assert period_b["Theta_D"].median() > period_b["Delta_D"].median()
