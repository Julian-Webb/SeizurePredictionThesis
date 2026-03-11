import pickle
from typing import Iterable

import numpy as np
import pandas as pd

from models.load_data import load_data
from feature_extraction.extract_features import FeatureNames
from utils import QueuedCall, run_queued_calls_on_gpus
from utils.io import pickle_path, save_dataframe_multiformat
from config.paths import PatientDir, PATHS


def calc_ptnt_seg_probabilities(
        pdir: PatientDir,
        models: Iterable[str] = ('ensemble', 'CNN'),
        batch_size: int = 2 ** 19,
):
    """
    Make raw predictions (probabilities rather than classifications) for a patient per segment.
    :param batch_size: Batch size for training with tensorflow (segments per batch).
     Needs to be small enough to not overallocate GPU memory, but large enough to be efficient.
    """
    # print(f'{[pdir.name]} Making raw predictions...', flush=True)
    import tensorflow as tf  # Local import, so only GPUs available to this process are initialized

    specs_per_model = {
        'CNN': (pdir.cnn_model, 'eeg'),
        'ensemble': (pdir.ensemble_model, 'features'),
    }

    segs = pd.read_pickle(pickle_path(pdir.segments_table))
    seg_probs = segs[['start_mtz']].copy()  # duplicate segment's index and start

    for model_name in models:
        model_path, data_type = specs_per_model[model_name]
        model = tf.keras.models.load_model(model_path)

        print_prefix = f'[{pdir.name} - {model_name}]'
        print(print_prefix, 'Loading data...', flush=True)

        data = load_data(
            segs,
            data_type,
            subsample_shuffle_and_subselect_types=False,
            train=True,
            test=True,
            # feature_names=FeatureNames.ENSEMBLE, # todo
            feature_names=FeatureNames.ALL_ORDERED,
            edf_dir=pdir.edf_dir,
        )

        if data_type == 'features':
            # Perform z-scaling for features
            with open(pdir.feature_scaler, 'rb') as f:
                scaler = pickle.load(f)
            data['x'] = scaler.transform(data['x'])

        # Predict in batches to avoid overallocating GPU memory
        n_segs = data['x'].shape[0]
        model_probs = []
        for i in range(0, n_segs, batch_size):
            end = min(i + batch_size, n_segs)
            print(print_prefix, f'Processing segments {i}-{end - 1} of {n_segs}...', flush=True)

            batch = data['x'][i:end]
            batch_probs = model.predict(batch)
            model_probs.append(batch_probs)

        model_probs = np.concatenate(model_probs)

        # Associate probabilities with segments (original index and start time)
        seg_probs.loc[data['index_and_start'].index, model_name] = model_probs

    # Save segment probabilities
    save_dataframe_multiformat(seg_probs, pdir.segment_probabilities_table)
    print(f'{[pdir.name]} Saved segment probabilities to {pdir.segment_probabilities_table}', flush=True)


def main(
        pdirs: list[PatientDir],
        serial_processing: bool,
        available_gpus: list[int] = None,
):
    if serial_processing:
        for pdir in pdirs:
            calc_ptnt_seg_probabilities(pdir)
    else:
        tasks = [
            QueuedCall(func=calc_ptnt_seg_probabilities, args=(pdir,), label=f'{pdir.name} - calc_seg_probabilities')
            for pdir in pdirs]
        run_queued_calls_on_gpus(tasks=tasks, gpus=available_gpus)


if __name__ == '__main__':
    main(
        PATHS.patient_dirs()[1:],  # todo do for all patients
        # PATHS.patient_dirs(), # todo do for all patients
        serial_processing=False,
        available_gpus=[0, 1, 2, 3],
    )
