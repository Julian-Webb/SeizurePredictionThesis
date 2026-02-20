import logging
import numpy as np
import pandas as pd

from models.load_data import load_data
from utils.io import pickle_path, save_dataframe_multiformat
from utils.utils import import_tensorflow_with_available_gpus
from config.paths import PatientDir, PATHS

import_tensorflow_with_available_gpus([2])
import tensorflow as tf


def make_raw_predictions(
        pdir: PatientDir,
        batch_size: int = 2 ** 19,
):
    """
    Make raw predictions (probabilities rather than classifications) for a patient per segment.
    :param batch_size: Batch size for training with tensorflow (segments per batch).
     Needs to be small enough to not overallocate GPU memory, but large enough to be efficient.
    """
    logging.info(f'===== Making raw predictions for patient {pdir.name}...')

    model_specs = [
        ['CNN', pdir.cnn_model, 'eeg'],
        ['ensemble', pdir.ensemble_model, 'features'],
    ]

    segs = pd.read_pickle(pickle_path(pdir.segments_table))

    for model_name, model_path, data_type in model_specs:
        logging.info(f'{model_name=}. Loading data...')
        model = tf.keras.models.load_model(model_path)
        data = load_data(
            segs,
            data_type,
            subsample_shuffle_and_subselect_types=False,
            train=True,
            test=True,
            edf_dir=pdir.edf_dir,
        )

        # Predict in batches to avoid overallocating GPU memory
        n_segs = data['x'].shape[0]
        all_probabilities = []
        for i in range(0, n_segs, batch_size):
            end = min(i + batch_size, n_segs)
            logging.info(f'Processing segments {i}-{end - 1} of {n_segs}...')

            batch = data['x'][i:end]
            probs = model.predict(batch)
            all_probabilities.append(probs)

        probabilities = np.concatenate(all_probabilities)

        # Associate probabilities with segments (original index and start time)
        seg_probs = segs[['start_mtz']].copy()
        seg_probs.loc[data['index_and_start'].index, 'probabilities'] = probabilities

        # Save raw probabilities
        save_path = pdir.predictions_dir / model_name / 'segment_probabilities'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataframe_multiformat(seg_probs, save_path)
        logging.info(f'Saved segment probabilities to {save_path}')


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    pdirs = PATHS.patient_dirs()

    for pdir in pdirs:
        make_raw_predictions(pdir)


if __name__ == '__main__': main()
