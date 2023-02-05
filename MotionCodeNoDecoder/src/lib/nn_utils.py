import tensorflow as tf


""" NN utils """


def make_nn(output_size, hidden_sizes, nw_name):
    """Creates fully connected neural network
    :param output_size: output dimensionality
    :param hidden_sizes: tuple of hidden layer sizes.
                         The tuple length sets the number of hidden layers.
    """
    layers = [
        tf.keras.layers.Dense(
            h, activation=tf.nn.relu, dtype=tf.float32, name=f"{nw_name}_hidden_{i}"
        )
        for i, h in enumerate(hidden_sizes)
    ]
    layers.append(
        tf.keras.layers.Dense(output_size, dtype=tf.float32, name=f"{nw_name}_op")
    )
    return tf.keras.Sequential(layers)


def make_cnn(output_size, hidden_sizes, kernel_size=3, nw_name="dgp_enc"):
    """Construct neural network consisting of
      one 1d-convolutional layer that utilizes temporal dependences,
      fully connected network

    :param output_size: output dimensionality
    :param hidden_sizes: tuple of hidden layer sizes.
                         The tuple length sets the number of hidden layers.
    :param kernel_size: kernel size for convolutional layer
    """
    cnn_layer = [
        tf.keras.layers.Conv1D(
            hidden_sizes[0],
            kernel_size=kernel_size,
            padding="same",
            dtype=tf.float32,
            name=f"{nw_name}_cnn",
        )
    ]
    layers = [
        tf.keras.layers.Dense(
            h, activation=tf.nn.relu, dtype=tf.float32, name=f"{nw_name}_hidden_{i}"
        )
        for i, h in enumerate(hidden_sizes[1:])
    ]
    layers.append(
        tf.keras.layers.Dense(output_size, dtype=tf.float32, name=f"{nw_name}_op")
    )
    return tf.keras.Sequential(cnn_layer + layers)
