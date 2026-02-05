import time

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense, \
    BatchNormalization, LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Recall, AUC
import pandas as pd

from config.constants import N_CHANNELS
from config.intervals import SEGMENT
from models.load_data import load_eeg_train_data
from models.FB_MLP import calc_class_weights
from config.paths import PatientDir, PATHS
from utils.io import pickle_path
from utils.tensorflow_utils import PeriodicalLogger

EPOCHS = 50  # 50
BATCH_SIZE = 256  # larger batch size, so that preictal samples are most likely in every batch
LEARNING_RATE = 0.001

LEAKY_RELU_NEGATIVE_SLOPE = 0.3
CONV2D_KWARGS = {
    'use_bias': False,  # no bias because we use BatchNorm afterward
    'activation': None,  # activation is applied after BatchNorm
    'kernel_regularizer': tf.keras.regularizers.l1_l2(1e-9, 1e-9),
}


def block_after_conv(layer_idx: int, pool_size: int, pool_padding: str = 'same'):
    return [
        BatchNormalization(name=f'batch_norm{layer_idx}'),
        LeakyReLU(negative_slope=LEAKY_RELU_NEGATIVE_SLOPE, name=f'leaky_relu{layer_idx}'),
        MaxPool2D([pool_size, 1], padding=pool_padding, name=f'max_pool{layer_idx}'),
        Dropout(0.2, name=f'dropout{layer_idx}')
    ]


def conv_block1(layer_idx: int, kernel_size: int, n_filters: int, pool_size: int):
    return [
        Conv2D(n_filters, [kernel_size, 1], padding='same', **CONV2D_KWARGS, name=f'conv{layer_idx}'),
        *block_after_conv(layer_idx, pool_size)
    ]


def conv_block2(layer_idx: int, kernel_size: int, n_filters1: int, n_filters2: int, pool_size: int,
                pool_padding: str = 'same'):
    return [
        Conv2D(n_filters1, [kernel_size, 1], padding='valid', **CONV2D_KWARGS, name=f'conv{layer_idx}.1'),
        Conv2D(n_filters2, [kernel_size, 1], padding='valid', **CONV2D_KWARGS, name=f'conv{layer_idx}.2'),
        *block_after_conv(layer_idx, pool_size, pool_padding)
    ]


def cnn_model(n_samples: int, n_channels: int) -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        Input([n_samples, n_channels, 1]),
        BatchNormalization(name='batch_norm0'),

        *conv_block1(1, kernel_size=5, n_filters=32, pool_size=5),
        *conv_block1(2, kernel_size=5, n_filters=64, pool_size=3),
        *conv_block1(3, kernel_size=3, n_filters=96, pool_size=2),
        *conv_block1(4, kernel_size=3, n_filters=128, pool_size=2),

        *conv_block2(5, kernel_size=4, n_filters1=128, n_filters2=96, pool_size=2),
        # !!! changed padding to valid !!!
        *conv_block2(6, kernel_size=4, n_filters1=64, n_filters2=32, pool_size=2, pool_padding='valid'),
        *conv_block2(7, kernel_size=4, n_filters1=32, n_filters2=32, pool_size=2),

        Flatten(name='flatten8'),
        Dropout(0.5, name='dropout8'),

        # !!! changed to 8 nodes, instead of 64, because #nodes gets reduced by a factor of 8
        Dense(8, activation=None, name='dense9'),
        LeakyReLU(negative_slope=LEAKY_RELU_NEGATIVE_SLOPE, name='leaky_relu9'),

        Dense(1, activation='sigmoid', name='output')
    ], name='CNN')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=BinaryCrossentropy(from_logits=False),
        metrics=["accuracy", Recall(name='recall'), AUC(name='AUC')]
    )

    return model


def create_ptnt_cnn(pdir: PatientDir):
    cnn = cnn_model(SEGMENT.n_samples, N_CHANNELS)

    # load data
    start = time.perf_counter()
    print(f'[{pdir.name}] Loading EEG data for CNN training')
    segs = pd.read_pickle(pickle_path(pdir.segments_table))
    esegs = segs[segs['exists']]
    split_idx = pd.read_pickle(pickle_path(pdir.train_test_split)).segment_index
    x_train, y_train = load_eeg_train_data(esegs, split_idx, pdir.edf_dir)
    print(f'[{pdir.name}] Finished loading data in {time.perf_counter() - start:.3f} sec.')

    # train model
    start = time.perf_counter()
    print(f'[{pdir.name}] Training CNN')
    class_weights = calc_class_weights(y_train)
    history = cnn.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weights,
                      verbose=0,
                      callbacks=[PeriodicalLogger(f'[{pdir.name}] CNN', interval=10)],
                      )
    print(f'[{pdir.name}] Finished training CNN in {time.perf_counter() - start:.3f} sec.')

    # Save
    pdir.cnn_model.parent.mkdir(parents=True, exist_ok=True)
    cnn.save(pdir.cnn_model)
    pd.DataFrame.from_dict(history.history).to_csv(pdir.cnn_history)


def create_ptnt_cnns(ptnt_dirs: list[PatientDir]):
    for ptnt_dir in ptnt_dirs:
        create_ptnt_cnn(ptnt_dir)


if __name__ == '__main__':
    pdirs = PATHS.patient_dirs()
    create_ptnt_cnns(pdirs)
