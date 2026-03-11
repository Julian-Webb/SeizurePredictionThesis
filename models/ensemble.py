import pickle
import time
from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Recall, AUC

from config.constants import RANDOM_STATE_FOR_TRAIN_DATA
from config import PatientDir, PATHS
from feature_extraction.extract_features import FeatureNames
from models.load_data import load_data
from utils.io import pickle_path, save_dataframe_multiformat
from utils.tensorflow_utils import PeriodicalLogger
from utils.utils import timeit

EPOCHS = 200  # 500
BATCH_SIZE = 256  # larger batch size, so that preictal samples are most likely in every batch
LEARNING_RATE = 0.0001  # 0.0001
ENSEMBLE_SIZE = 50  # 100


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


def create_ensemble(
        x_train: np.ndarray,
        y_train: np.ndarray,
        n_features: int,
        logging_info: str = '[unknown patient]',
        ensemble_size: int = ENSEMBLE_SIZE,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE
):
    class_weights = calc_class_weights(y_train)
    models = []
    history_frames: list[pd.DataFrame] = []
    input_layer = Input([n_features], name='ensemble_input')

    # Train individual models in loop
    for i in range(ensemble_size):
        start = time.perf_counter()
        name = f"FB-MLP_{i:02}"
        model = mlp_model(n_features, name)
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            verbose=0,
            callbacks=[PeriodicalLogger(f'{logging_info} - {name}', interval=100)],
        )

        model_history = pd.DataFrame(history.history)
        model_history.insert(0, 'epoch', np.arange(1, len(model_history) + 1))
        model_history.insert(0, 'member_idx', i)
        model_history.insert(0, 'member_name', name)
        history_frames.append(model_history)

        y = model(input_layer)  # Make all models share the same input layer
        models.append(y)

        print(f'{logging_info} Finished model {name} in {time.perf_counter() - start:.3f} sec.')

    # The ensemble output averages the outputs of the individual models
    output_layer = tf.keras.layers.average(models, name='ensemble_average')
    ensemble = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='ensemble')

    histories = pd.concat(history_frames, ignore_index=True)
    histories.set_index(['member_idx', 'epoch'], inplace=True)
    histories.sort_index(inplace=True)
    return ensemble, histories


def create_ptnt_mlp_ensemble(
        pdir: PatientDir,
        feature_names: list[str] = FeatureNames.ENSEMBLE,
):
    segs = pd.read_pickle(pickle_path(pdir.segments_table))
    split_idx = pd.read_pickle(pickle_path(pdir.train_test_split)).segment_index

    load_data_partial = partial(load_data,
                                segs=segs,
                                type_='features',
                                train=True,
                                split_idx=split_idx,
                                feature_names=feature_names)

    all_train_data = load_data_partial(subsample_shuffle_and_subselect_types=False)

    # Fit z-score normalizer on the features of the entire train set
    scaler = StandardScaler()
    scaler.fit(all_train_data['x'])

    # Subselect segments and train models
    sub_train_data = load_data_partial(subsample_shuffle_and_subselect_types=True,
                                       random_state=RANDOM_STATE_FOR_TRAIN_DATA)
    x_train = scaler.transform(sub_train_data['x'], copy=False)

    ensemble, histories = create_ensemble(
        x_train,
        sub_train_data['y'],
        n_features=len(feature_names),
        logging_info=f'[{pdir.name}]'
    )
    return ensemble, scaler, histories


@timeit
def create_ptnt_ensemble_and_save(pdir: PatientDir):
    print(f'Creating ensemble for {pdir.name}')
    start = time.perf_counter()
    ensemble, scaler, histories = create_ptnt_mlp_ensemble(pdir)

    # Save
    pdir.ensemble_model.parent.mkdir(exist_ok=True, parents=True)
    ensemble.save(pdir.ensemble_model)
    with open(pdir.feature_scaler, 'wb') as f:
        # noinspection PyTypeChecker
        pickle.dump(scaler, f)
    save_dataframe_multiformat(histories, pdir.mlp_history, csv_index=True)

    print(f'Finished ensemble creation for {pdir.name} in {time.perf_counter() - start:.3f} sec.')


def create_mlp_ensembles(pdirs: list[PatientDir]):
    st = time.perf_counter()

    for pdir in pdirs:
        create_ptnt_ensemble_and_save(pdir)

    elapsed_time = time.perf_counter() - st
    print(f'Finished ensemble creation in {elapsed_time / 3600:.2f} hours')


if __name__ == '__main__': create_mlp_ensembles(PATHS.patient_dirs())
