import logging

import tensorflow as tf


class PeriodicalLogger(tf.keras.callbacks.Callback):
    def __init__(self, model_name, interval=100):
        super().__init__()
        self.model_name = model_name
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if (epoch == 1) or (epoch % self.interval == 0):
            # logs is a dict containing the metrics defined in model.compile
            msg = f"[{self.model_name}] Epoch {epoch}/{self.params['epochs']}"
            for metric, value in logs.items():
                msg += f" - {metric}: {value:.4f}"
            logging.info(msg)
