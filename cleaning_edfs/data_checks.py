import time
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, Dict

from config.constants import SAMPLING_FREQUENCY_HZ as sfreq
from config.paths import PATHS, PatientDir
import pyedflib


def check_patient_signals(patient_dir: PatientDir) -> Tuple[Path, Dict[str, str]]:
    """Check signal properties for a single patient
    Returns:
        Tuple containing patient name and dict with status and any error messages
    """
    result = {"status": "success", "errors": []}

    try:
        for edf_path in patient_dir.edf_dir.iterdir():
            try:
                header = pyedflib.highlevel.read_edf_header(str(edf_path))
                if len(header['SignalHeaders']) != 2:
                    result["errors"].append(f"There aren't 2 signals in {edf_path}")
                    result["status"] = "error"
                    continue

                for signal in header['SignalHeaders']:
                    if signal['sample_frequency'] != sfreq:
                        result["errors"].append(
                            f"The sampling rate is not {sfreq} Hz in {edf_path}"
                        )
                        result["status"] = "error"
            except Exception as e:
                result["errors"].append(f"Error processing {edf_path}: {str(e)}")
                result["status"] = "error"

    except Exception as e:
        result["errors"].append(f"Error accessing patient directory: {str(e)}")
        result["status"] = "error"

    print(f"Finished checking {patient_dir}")

    return patient_dir, result


def check_signal_properties() -> Dict[Path, Dict[str, str]]:
    """Assert that the sampling rate is the same for all files and all channels
    Returns:
        Dict mapping patient IDs to their check results
    """
    # # Sequential processing:
    # results = {}
    # for patient_dir in PATHS.patient_dirs():
    #     results[patient_dir] = check_patient_signals(patient_dir)

    # Parallel processing:
    with Pool() as pool:
        # Map the check_patient_signals function across all patients
        results = dict(pool.map(check_patient_signals, PATHS.patient_dirs()))

    return results


if __name__ == '__main__':
    start = time.time()
    check_results = check_signal_properties()

    # Print summary
    print('\nResults summary:')
    error_count = 0
    for patient, result in check_results.items():
        if result['status'] == 'error':
            print(f"\n{patient} - {result['status'].upper()}")
            for error in result['errors']:
                print(f'  - {error}')
            error_count += 1
        else:
            print(f"{patient} - {result['status'].upper()}")

    print(f'\nFinished in {time.time() - start:.1f} seconds')
    print(f'Found errors in {error_count} patients')
