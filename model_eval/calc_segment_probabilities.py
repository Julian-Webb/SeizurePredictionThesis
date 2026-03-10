import logging
import pickle

import numpy as np
import pandas as pd

from feature_extraction.extract_features import FeatureNames
from models.load_data import load_data
from utils.io import pickle_path, save_dataframe_multiformat
from utils.utils import import_tensorflow_with_available_gpus
from config.paths import PatientDir, PATHS

import_tensorflow_with_available_gpus([2])
import tensorflow as tf


def calc_seg_probabilities(
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
    seg_probs = segs[['start_mtz']].copy() # duplicate segment's index and start

    for model_name, model_path, data_type in model_specs:
        logging.info(f'{model_name=}. Loading data...')
        model = tf.keras.models.load_model(model_path)
        data = load_data(
            segs,
            data_type,
            subsample_shuffle_and_subselect_types=False,
            train=True,
            test=True,
            feature_names=FeatureNames.ENSEMBLE,
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
            logging.info(f'Processing segments {i}-{end - 1} of {n_segs}...')

            batch = data['x'][i:end]
            batch_probs = model.predict(batch)
            model_probs.append(batch_probs)

        model_probs = np.concatenate(model_probs)

        # Associate probabilities with segments (original index and start time)
        seg_probs.loc[data['index_and_start'].index, model_name] = model_probs

    # Save segment probabilities
    save_dataframe_multiformat(seg_probs, pdir.segment_probabilities_table)
    logging.info(f'Saved segment probabilities to {pdir.segment_probabilities_table}')


def main(pdirs: list[PatientDir] = PATHS.patient_dirs()):
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    for pdir in pdirs:
        calc_seg_probabilities(pdir)


if __name__ == '__main__': main()
