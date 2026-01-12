import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import pickle
import time

import keras
import numpy as np
import pandas as pd
from keras import layers
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import BinaryCrossentropy

from utils.utils import PeriodicalLogger

# make tensorflow only use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.metrics import Recall, AUC

from feature_extraction.extract_features import Features
from config.paths import PATHS, PatientDir
from models.load_data import subsample_shuffle_train_segs, seg_features_to_numpy
from utils.io import pickle_path

# todo move this to constants?
EPOCHS = 500  # 500
BATCH_SIZE = 256  # larger batch size, so that preictal samples are most likely in every batch
LEARNING_RATE = 0.0001  # 0.0001
ENSEMBLE_SIZE = 100  # 100


def mlp_model(n_features: int, name: str) -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        Input([n_features], name='mlp_input'),
        Dense(16, activation='relu', name='dense0'),
        BatchNormalization(name='batch_norm0'),
        Dense(8, activation='relu', name='dense1'),
        BatchNormalization(name='batch_norm1'),
        Dense(4, activation='relu', name='dense2'),
        BatchNormalization(name='batch_norm2'),
        Dense(1, activation='sigmoid', name='mlp_output')
    ], name=name)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        loss=BinaryCrossentropy(from_logits=False),
        metrics=["accuracy", Recall(name='recall'), AUC(name='AUC')]
    )

    return model


def calc_class_weights(y_train: np.ndarray) -> dict:
    total = len(y_train)  # number of training samples
    counts = np.bincount(y_train)
    n_classes = len(counts)
    class_weights = {
        0: total / (n_classes * counts[0]),
        1: total / (n_classes * counts[1]),
    }
    return class_weights


def create_ensemble(train_segs: DataFrame,
                    ensemble_size: int = ENSEMBLE_SIZE, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    input_layer = Input([Features.N_FEATURES], name='ensemble_input')
    models = []
    for i in range(ensemble_size):
        name = f"FB-MLP_{i:02}"
        logging.info(f'Creating model {name}')
        start = time.perf_counter()
        # Select train data for this model
        # Due to subsampling, every model gets different interictal segs
        train_segs = subsample_shuffle_train_segs(train_segs)
        x_train, y_train = seg_features_to_numpy(train_segs, Features.ORDERED_NAMES)

        class_weights = calc_class_weights(y_train)
        model = mlp_model(Features.N_FEATURES, name)
        # Train individual model
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weights,
                  verbose=0,
                  callbacks=[PeriodicalLogger(name, interval=100)],
                  )
        # Make all models share the same input layer
        y = model(input_layer)
        models.append(y)

        logging.info(f'Finished model {name} in {time.perf_counter() - start:.3f} sec.\n')

    # The ensemble output averages the outputs of the individual models
    output_layer = layers.average(models, name='ensemble_average')
    ensemble = keras.Model(inputs=input_layer, outputs=output_layer, name='ensemble')
    return ensemble


def create_ptnt_mlp_ensemble(ptnt_dir: PatientDir):
    # Load Data
    segs = pd.read_pickle(pickle_path(ptnt_dir.segments_table))
    esegs = segs[segs['exists']]
    split_idx = pd.read_pickle(pickle_path(ptnt_dir.train_test_split)).segment_index

    # Perform z-score normalization on the features
    scaler = StandardScaler()
    train_segs = esegs.loc[:split_idx - 1]
    train_features = train_segs[Features.ORDERED_NAMES].values
    scaler.fit(train_features)
    # Transform
    train_segs.loc[:, Features.ORDERED_NAMES] = scaler.transform(train_features)

    # Create ensemble
    # noinspection PyTypeChecker
    ensemble = create_ensemble(train_segs)
    return ensemble, scaler


def create_ptnt_ensemble_and_save(ptnt_dir: PatientDir):
    logging.info(f'Creating ensemble for {ptnt_dir.name}')
    start = time.perf_counter()

    ensemble, scaler = create_ptnt_mlp_ensemble(ptnt_dir)
    # Save
    ptnt_dir.models_dir.mkdir(exist_ok=True, parents=True)
    ensemble.save(ptnt_dir.ensemble_model)
    with open(ptnt_dir.feature_scaler, 'wb') as f:
        # noinspection PyTypeChecker
        pickle.dump(scaler, f)

    logging.info(f'Finished ensemble creation for {ptnt_dir.name} in {time.perf_counter() - start:.3f} sec.')


def create_ptnt_mlp_ensembles(ptnt_dirs: list[PatientDir], serial_processing: bool = False):
    if serial_processing:
        for ptnt_dir in ptnt_dirs:
            create_ptnt_ensemble_and_save(ptnt_dir)
    else:
        max_workers = len(tf.config.list_physical_devices('GPU'))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            pool.map(create_ptnt_ensemble_and_save, ptnt_dirs)



if __name__ == '__main__':
    date_str = datetime.now().strftime('%Y-%m-%d')
    log_path = PATHS / f'mlp_creation_log_{date_str}.txt'
    logging.basicConfig(filename=log_path, level='INFO', format='[%(levelname)s] %(message)s')
    st = time.perf_counter()

    create_ptnt_mlp_ensembles(PATHS.patient_dirs())

    elapsed_time = time.perf_counter() - st
    logging.info(f'Finished ensemble creation in {elapsed_time / 3600:.2f} hours')
