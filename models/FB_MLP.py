import logging
import os
import time

import keras
import numpy as np
import pandas as pd
from keras import layers
from tensorflow.keras.losses import BinaryCrossentropy

# make tensorflow only use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.metrics import Recall, AUC

from feature_extraction.extract_features import Features
from config.paths import PATHS, PatientDir
from models.load_data import load_features_and_labels
from utils.io import pickle_path

EPOCHS = 500
BATCH_SIZE = 256  # larger batch size, so that preictal samples are most likely in every batch
LEARNING_RATE = 0.0001
ENSEMBLE_SIZE = 100


def create_mlp(n_features: int, name: str) -> tf.keras.models.Sequential:
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


def create_ensemble(x_train: np.ndarray, y_train: np.ndarray,
                    ensemble_size: int = ENSEMBLE_SIZE, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    class_weights = calc_class_weights(y_train)
    input_layer = Input([Features.N_FEATURES], name='ensemble_input')

    models = []
    for i in range(ensemble_size):
        logging.info(f'Creating FB-MLP_{i:02}')
        start = time.perf_counter()

        model = create_mlp(Features.N_FEATURES, f"FB-MLP_{i:02}")
        # Train individual model
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weights,
                  # verbose=0,
                  )
        # Make all models share the same input layer
        y = model(input_layer)
        models.append(y)

        logging.info(f'Finished individual model {i:02} in {time.perf_counter() - start:.3f} sec.')

    # The ensemble output averages the outputs of the individual models
    output_layer = layers.average(models, name='ensemble_average')
    ensemble = keras.Model(inputs=input_layer, outputs=output_layer, name='ensemble')
    return ensemble


def create_ptnt_mlp_ensemble(ptnt_dir: PatientDir):
    # Load Data
    logging.info(f'Creating ensemble for {ptnt_dir.name}')
    start = time.perf_counter()

    segs = pd.read_pickle(pickle_path(ptnt_dir.segments_table))
    split = pd.read_pickle(pickle_path(ptnt_dir.train_test_split))
    x_train, y_train, x_test, y_test = load_features_and_labels(segs, split, Features.ORDERED_NAMES)
    # Create ensemble
    ensemble = create_ensemble(x_train, y_train)

    logging.info(f'Finished ensemble creation for {ptnt_dir.name} in {time.perf_counter() - start:.3f} sec.')
    return ensemble


def create_ensemble_models(ptnt_dirs: list[PatientDir]):
    for ptnt_dir in ptnt_dirs:
        ensemble = create_ptnt_mlp_ensemble(ptnt_dir)
        # Save
        ptnt_dir.models_dir.mkdir(exist_ok=True, parents=True)
        ensemble.save(ptnt_dir.ensemble_model)


if __name__ == '__main__':
    log_path = PATHS / 'mlp_creation_log.txt'
    logging.basicConfig(filename=log_path, level='INFO', format='[%(levelname)s] %(message)s')
    st = time.perf_counter()
    create_ensemble_models(PATHS.patient_dirs())

    elapsed_time = time.time() - st
    logging.info(f'Finished ensemble creation in {elapsed_time / 3600:.2f} hours')
