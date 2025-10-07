from datetime import timedelta
from pathlib import Path
from typing import List


class Paths:
    BASE_DIR = Path('/data/home/webb/UNEEG_data')

    DATA_CLEANING_LOG_DIR = BASE_DIR / 'data_cleaning_logs'

    PROBLEMATIC_EDFS_DIR = DATA_CLEANING_LOG_DIR / 'problematic_edf_files'
    PROBLEMATIC_EDFS_FILE = PROBLEMATIC_EDFS_DIR / 'problematic_edf_files.csv'
    REMAINING_DUPLICATES_FILE = DATA_CLEANING_LOG_DIR / 'remaining_duplicates.txt'

    # dataset folders
    FOR_MAYO_DIR = BASE_DIR / '20240201_UNEEG_ForMayo'
    UNEEG_EXTENDED_DIR = BASE_DIR / '20250217_UNEEG_Extended'
    COMPETITION_DIR = BASE_DIR / '20250501_SUBQ_SeizurePredictionCompetition_2025final'

    @staticmethod
    def seizure_annotations_dir(patient_dir: Path):
        return patient_dir / 'seizure_annotations'

    @staticmethod
    def seizure_annotations_file(patient_dir: Path):
        return Paths.seizure_annotations_dir(patient_dir) / 'seizure_annotations.csv'

    @staticmethod
    def original_edf_dir(patient_dir: Path):
        """The directory containing the original edf files for the competition dataset, before they're renamed"""
        return patient_dir / 'original_edf_data'

    @staticmethod
    def edf_dir(patient_dir: Path):
        """The directory containing the edf files"""
        return patient_dir / 'edf_data'

    @staticmethod
    def edf_files_sheet(patient_dir: Path):
        """The name of the sheet containing the edf file names and their metadata for each patient"""
        return patient_dir / 'edf_files.csv'

    @staticmethod
    def patient_dirs() -> List[Path]:
        """Generator for all patient folders"""
        patient_dirs = []
        for parent_dir in [Paths.FOR_MAYO_DIR, Paths.UNEEG_EXTENDED_DIR, Paths.COMPETITION_DIR]:
            for patient_dir in parent_dir.iterdir():
                if patient_dir.is_dir():
                    patient_dirs.append(patient_dir)
        return patient_dirs


class Constants:
    # The sampling frequency of all EEG signals in the edf files
    SAMPLING_FREQUENCY_HZ = 207.0310546581987


def _n_samples_and_exact_length_from_approximate(length_approx: timedelta, segment_approx: timedelta,
                                                 segment_n_samples: int, sampling_frequency_hz: float) -> tuple[
    int, timedelta]:
    """Calculates the number of samples and the exact duration of a clip or segment from the desired approximate
     length, based on the length of a segment and the sampling frequency. This is done so that the number of
     samples is a whole number.
    """
    segments_per_window = length_approx / segment_approx
    assert segments_per_window % 1 == 0, "The approximate length isn't a multiple of the segment length"
    n_samples = segments_per_window * segment_n_samples
    assert n_samples % 1 == 0, "The number of samples isn't a whole number"
    n_samples = int(n_samples)
    exact_length = timedelta(seconds=n_samples / sampling_frequency_hz)
    return n_samples, exact_length


def _safe_float_to_int(num: float) -> int:
    if num != int(num):
        raise ValueError(f"Number {num} has decimal values")
    return int(num)


class Durations:
    # The desired approximate length of the segments which the clips are split into
    SEGMENT_APPROX = timedelta(seconds=15)
    # The desired approximate length of the clips
    CLIP_APPROX = timedelta(minutes=10)
    # How much time is between the end of the preictal interval and the seizure
    PREICTAL_OFFSET_APPROX = timedelta(minutes=5)
    # The duration of the preictal interval
    PREICTAL_INTERVAL_APPROX = timedelta(minutes=60)

    # Because the sample frequency is a decimal fraction, the slip and segment lengths will be based on it, so that
    # slices correspond to full indexes
    # The number of points (=samples) in a segment
    SEGMENT_N_SAMPLES = round(SEGMENT_APPROX.seconds * Constants.SAMPLING_FREQUENCY_HZ)
    # True segment length
    SEGMENT = timedelta(seconds=SEGMENT_N_SAMPLES / Constants.SAMPLING_FREQUENCY_HZ)

    # Calculate the number of samples and exact lengths of the intervals
    CLIP_N_SAMPLES, CLIP = _n_samples_and_exact_length_from_approximate(CLIP_APPROX, SEGMENT_APPROX,
                                                                        SEGMENT_N_SAMPLES,
                                                                        Constants.SAMPLING_FREQUENCY_HZ)
    PREICTAL_OFFSET_N_SAMPLES, PREICTAL_OFFSET = _n_samples_and_exact_length_from_approximate(
        PREICTAL_OFFSET_APPROX, SEGMENT_APPROX, SEGMENT_N_SAMPLES, Constants.SAMPLING_FREQUENCY_HZ)
    PREICTAL_INTERVAL_N_SAMPLES, PREICTAL_INTERVAL = _n_samples_and_exact_length_from_approximate(
        PREICTAL_INTERVAL_APPROX, SEGMENT_APPROX, SEGMENT_N_SAMPLES, Constants.SAMPLING_FREQUENCY_HZ)

    CLIPS_PER_PREICTAL_INTERVAL = _safe_float_to_int(PREICTAL_INTERVAL_N_SAMPLES / CLIP_N_SAMPLES)
    SEGMENTS_PER_PREICTAL_INTERVAL = _safe_float_to_int(PREICTAL_INTERVAL_N_SAMPLES / SEGMENT_N_SAMPLES)
    SEGMENTS_PER_CLIP = _safe_float_to_int(CLIP_N_SAMPLES / SEGMENT_N_SAMPLES)


if __name__ == "__main__":
    print('Segment parameters')
    print(f'n_samples: {Durations.SEGMENT_N_SAMPLES}')
    print(f'length: {Durations.SEGMENT} h')
    print()
    print('Clip parameters')
    print(f'n_samples: {Durations.CLIP_N_SAMPLES}')
    print(f'length: {Durations.CLIP} h')
    print()
    print('Preictal offset parameters')
    print(f'n_samples: {Durations.PREICTAL_OFFSET_N_SAMPLES}')
    print(f'length: {Durations.PREICTAL_OFFSET} h')
    print()
    print('Preictal interval parameters')
    print(f'n_samples: {Durations.PREICTAL_INTERVAL_N_SAMPLES}')
    print(f'length: {Durations.PREICTAL_INTERVAL} h')
    print()
    print(f'CLIPS_PER_PREICTAL_INTERVAL: {Durations.CLIPS_PER_PREICTAL_INTERVAL}')
    print(f'SEGMENTS_PER_PREICTAL_INTERVAL: {Durations.SEGMENTS_PER_PREICTAL_INTERVAL}')
    print(f'SEGMENTS_PER_CLIP: {Durations.SEGMENTS_PER_CLIP}')
