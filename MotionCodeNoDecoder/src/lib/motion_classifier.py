import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers.core import Dense, Activation, Dropout
from tqdm import tqdm


class Classifier(tf.keras.Model):
    def __init__(
        self,
        name,
        ouput_class_distn,
        num_gru_layers=8,
        gru_units=32,
        dense_size=[128, 128],
        use_weighted_loss=True,
        weighted_loss_multiplier=0.15,
        num_iters=6000,
    ):
        super(Classifier, self).__init__()
        self.namer = name
        self.ouput_class_distn = ouput_class_distn
        self.output_dimension = len(ouput_class_distn)
        self.get_multiplier()
        self.num_gru_layers = num_gru_layers
        self.gru_units = gru_units
        self.dense_size = dense_size
        self.use_weighted_loss = use_weighted_loss
        self.weighted_loss_multiplier = weighted_loss_multiplier
        self.num_iters = num_iters
        self.data_dim = 3
        self.build_model()

    def get_multiplier(self):
        all_sum = sum([v for _, v in self.ouput_class_distn.items()])
        self.multiplier_dct = {
            k: all_sum / v for k, v in self.ouput_class_distn.items()
        }

    def build_model(self):
        self.model = keras.Sequential()
        if self.num_gru_layers >= 1:
            idx = 0
            for idx in range(self.num_gru_layers - 1):
                self.model.add(
                    layers.GRU(
                        self.gru_units,
                        activation="tanh",
                        return_sequences=True,
                        name=f"adv_classifier_{self.namer}_gru_{idx}",
                    )
                )
            self.model.add(
                layers.GRU(
                    self.gru_units,
                    activation="tanh",
                    name=f"adv_classifier_{self.namer}_gru_{idx+1}",
                )
            )
        else:
            self.model.add(
                tf.keras.layers.Conv1D(
                    64,
                    kernel_size=3,
                    padding="same",
                    dtype=tf.float32,
                    name=f"adv_classifier_{self.namer}_cnn",
                )
            )
            self.model.add(layers.Flatten())
        for idx, ds in enumerate(self.dense_size):
            self.model.add(Dense(ds, name=f"adv_classifier_{self.namer}_dense_{idx}"))
        self.model.add(
            Dense(self.output_dimension, name=f"adv_classifier_{self.namer}_op")
        )
        self.model.add(Activation("softmax"))

    def calculate_loss(self, y_pred, y_gt, use_weight=False):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_gt, y_pred)
        if self.use_weighted_loss:
            multiplier = tf.constant(
                [self.multiplier_dct[x] for x in y_gt.numpy()], dtype=tf.float32
            )
            loss = loss * (1 + self.weighted_loss_multiplier * multiplier)
        return tf.math.reduce_mean(loss)
