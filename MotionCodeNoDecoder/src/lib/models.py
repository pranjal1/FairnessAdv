"""
TensorFlow models for use in this project.
"""
import tensorflow as tf

from .utils import *
from .nn_utils import *
from .motion_classifier import Classifier

# Encoder
class Encoder(tf.keras.Model):
    def __init__(
        self,
        z_size,
        hidden_sizes=(64, 64),
        window_size=3,
        transpose=False,
        data_type=None,
        **kwargs
    ):
        """Encoder with 1d-convolutional network and factorized Normal posterior
        Used by joint VAE and HI-VAE with Standard Normal prior or GP-VAE with factorized Normal posterior
        :param z_size: latent space dimensionality
        :param hidden_sizes: tuple of hidden layer sizes.
                             The tuple length sets the number of hidden layers.
        :param window_size: kernel size for Conv1D layer
        :param transpose: True for GP prior | False for Standard Normal prior
        """
        super(Encoder, self).__init__()
        self.z_size = int(z_size)
        self.net = make_cnn(z_size, hidden_sizes, window_size)
        self.transpose = transpose
        self.data_type = data_type

    def __call__(self, x):
        mapped = self.net(x)
        if self.data_type in ["physionet", "hirid", "synth"]:
            if self.transpose:
                num_dim = len(x.shape.as_list())
                perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
                mapped = tf.transpose(mapped, perm=perm)
                return mapped
            return mapped
        else:
            if self.transpose:
                num_dim = len(x.shape.as_list())
                perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
                mapped = tf.transpose(mapped, perm=perm)
                return mapped
            return mapped


# Trainer Model
class TrainerModel(tf.keras.Model):
    def __init__(
        self,
        latent_dim,
        data_dim,
        time_length,
        encoder_sizes=(64, 64),
        encoder=Encoder,
        **kwargs
    ):
        """Basic Variational Autoencoder with Standard Normal prior
        :param latent_dim: latent space dimensionality
        :param data_dim: original data dimensionality
        :param time_length: time series duration
        :param encoder_sizes: layer sizes for the encoder network
        :param encoder: encoder model class {Diagonal, Joint, BandedJoint}Encoder
        """
        super(TrainerModel, self).__init__()
        self.latent_dim = latent_dim
        assert self.latent_dim > 1, "Latent dimension has to be greater than 1"
        self.data_dim = data_dim
        self.time_length = time_length

        self.encoder = encoder(latent_dim, encoder_sizes, **kwargs)
        self.initialize_adv_nws(**kwargs)

    def initialize_adv_nws(self, **kwargs):
        self.adv_classifier_ts = Classifier(
            name="ts",
            ouput_class_distn=kwargs.get("gender_distn"),
            num_gru_layers=kwargs.get("num_gru_layers_sens"),
            gru_units=kwargs.get("gru_units_sens"),
            dense_size=kwargs.get("dense_size_sens"),
            use_weighted_loss=kwargs.get("use_weighted_loss_sens"),
            weighted_loss_multiplier=kwargs.get("weighted_loss_multiplier_adv_sens"),
        )
        self.adv_classifier_tt = Classifier(
            name="tt",
            ouput_class_distn=kwargs.get("tgt_distn"),
            num_gru_layers=kwargs.get("num_gru_layers_tgt"),
            gru_units=kwargs.get("gru_units_tgt"),
            dense_size=kwargs.get("dense_size_tgt"),
            use_weighted_loss=kwargs.get("use_weighted_loss_tgt"),
            weighted_loss_multiplier=kwargs.get("weighted_loss_multiplier_adv_tgt"),
        )

    def encode(self, x):
        x = tf.identity(x)  # in case x is not a Tensor already...
        return self.encoder(x)

    def compute_loss(
        self,
        x,
        target_label=None,
        target_sensitive=None,
    ):
        assert (
            len(x.shape) == 3
        ), "Input should have shape: [batch_size, time_length, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        z = self.encode(x)

        target_predicts_target = self.adv_classifier_tt.model(z)

        loss_tt = self.adv_classifier_tt.calculate_loss(
            target_predicts_target, target_label
        )

        target_predicts_sensitive = self.adv_classifier_ts.model(z)

        loss_ts = self.adv_classifier_ts.calculate_loss(
            target_predicts_sensitive, target_sensitive
        )
        return loss_tt, loss_ts

    def get_trainable_vars(self):
        self.compute_loss(
            x=tf.random.normal(
                shape=(1, self.time_length, self.data_dim), dtype=tf.float32
            ),
            target_label=tf.constant(
                [0],
                dtype=tf.float32,
            ),
            target_sensitive=tf.constant(
                [0],
                dtype=tf.float32,
            ),
        )
        return self.trainable_variables
