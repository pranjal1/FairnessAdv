import tensorflow as tf
import numpy as np


""" TF utils """


def dyn_data_reshape(data, new_len):
    """
    Reshapes data to be able to train on variable time series length.

    """
    N = data["x_train_full"].shape[0]
    orig_len = data["x_train_full"].shape[1]
    data_dim = data["x_train_full"].shape[2]
    # Don't do anything if length is already correct
    if orig_len == new_len:
        return data

    assert (
        orig_len % new_len == 0
    ), f"original length {orig_len} must be integer multiple of new length {new_len}"
    assert orig_len > new_len, "New length must be less than original length"

    c = orig_len // new_len

    data_reshape = {}
    for file in data.files:
        if len(data[file].shape) > 2:  # Only reshape if there are enough dimensions
            N_new = data[file].shape[0] * c
            data_reshape[file] = np.reshape(data[file], (N_new, new_len, data_dim))

    return data_reshape


def reduce_logmeanexp(x, axis, eps=1e-5):
    """Numerically-stable (?) implementation of log-mean-exp.
    Args:
        x: The tensor to reduce. Should have numeric type.
        axis: The dimensions to reduce. If `None` (the default),
              reduces all dimensions. Must be in the range
              `[-rank(input_tensor), rank(input_tensor)]`.
        eps: Floating point scalar to avoid log-underflow.
    Returns:
        log_mean_exp: A `Tensor` representing `log(Avg{exp(x): x})`.
    """
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    return (
        tf.math.log(tf.reduce_mean(tf.exp(x - x_max), axis=axis, keepdims=True) + eps)
        + x_max
    )
