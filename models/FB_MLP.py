import os
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.losses import BinaryCrossentropy

# make tensorflow only use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.metrics import Recall, AUC, Accuracy

from feature_extraction.extract_features import Features
from config.paths import PATHS, PatientDir
from models.load_data import load_features_and_labels
from utils.io import pickle_path

# EPOCHS = 500 # todo uncomment
EPOCHS = 10  # todo delete
BATCH_SIZE = 256  # larger batch size, so that preictal samples are most likely in every batch
LEARNING_RATE = 0.0001
ENSEMBLE_SIZE = 100


def create_mlp(n_features: int, name: str) -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        Input([n_features]),
        Dense(16, activation='relu', name='dense0'),
        BatchNormalization(name='batch_norm0'),
        Dense(8, activation='relu', name='dense1'),
        BatchNormalization(name='batch_norm1'),
        Dense(4, activation='relu', name='dense2'),
        BatchNormalization(name='batch_norm2'),
        Dense(1, activation='sigmoid', name='output')
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


class Ensemble:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray,
                        ensemble_size: int = ENSEMBLE_SIZE, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):

        class_weights = calc_class_weights(y_train)
        self.models = []
        for i in range(ensemble_size):
            model = create_mlp(Features.N_FEATURES, f"FB-MLP_{i:02}")
            model.fit(x_train, y_train,
                      epochs=epochs,
                      batch_size=batch_size,
                      class_weight=class_weights
                      )
            self.models.append(model)


    def predict_probs(self, x_test: np.ndarray):
        """Predict the probabilities for x_test"""
        all_preds = np.array([m.predict(x_test) for m in self.models])
        # Average the probabilities
        ensemble_preds = all_preds.mean(axis=0)
        return ensemble_preds

    def predict_class(self, x_test: np.ndarray, threshold: float):
        """Predict the class labels (0/1) for x_test based on the threshold"""
        probs = self.predict_probs(x_test)
        class_preds = np.array([prob > threshold for prob in probs]).astype(int)
        return class_preds

    def save_model(self, folder: Path):
        for model in self.models:
            path = (folder / model.name).with_suffix('.keras')
            model.save(path)




def create_ptnt_mlp_ensemble(ptnt_dir: PatientDir):
    segs = pd.read_pickle(pickle_path(ptnt_dir.segments_table))
    split = pd.read_pickle(pickle_path(ptnt_dir.train_test_split))
    x_train, y_train, x_test, y_test = load_features_and_labels(segs, split, Features.ORDERED_FEATURE_NAMES)
    ensemble = Ensemble(x_train, y_train)

