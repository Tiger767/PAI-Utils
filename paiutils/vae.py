"""
Author: Travis Hammond
Version: 11_19_2020
"""


import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json

try:
    from paiutils.neural_network import (
        Trainer, Predictor, dense, conv2d
    )
except ImportError:
    from neural_network import (
        Trainer, Predictor, dense, conv2d
    )


class VAETrainer(Trainer):
    """VAETrainer is used for loading, saving,
       and training keras variational autoencoder models.
    """

    def __init__(self, encoder_model,
                 decoder_model, train_data):
        """Initializes train, validation, and test data.
        params:
            model: A compiled full keras model
            encoder_model: The encoder part of the full model
            decoder_model: The decoder part of the full model
            train_data: A dictionary, numpy ndarray containg train data,
                        or a list with x and y ndarrays.
        """
        if not isinstance(train_data, (dict, np.ndarray, list)):
            raise TypeError(
                'data must be either a dictionary, ndarray, or list'
            )
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.input_shape = self.encoder_model.layers[0].input_shape[0][1:]
        self.optimizer = self.encoder_model.optimizer
        self.metric = tf.keras.metrics.Mean(name='loss')
        self.train_data = train_data

        if not isinstance(train_data, np.ndarray):
            if isinstance(train_data, dict):
                if 'train_x' in train_data:
                    self.train_data = train_data['train_x']
                elif 'train' in train_data:
                    self.train_data = train_data['train']
                else:
                    raise Exception('There must be train data')
            else:
                raise ValueError('Invalid train_data')
        self.train_data = self.train_data.astype(
            tf.keras.backend.floatx()
        )

    @tf.function
    def _train_step(self, x):
        """Trains the VAE 1 epoch.
        params:
            x: A Tensor
        """
        def log_normal_pdf(sample, mean, logvar, raxis=1):
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_sum(-.5 *
                                 ((sample - mean) ** 2. *
                                  tf.exp(-logvar) + logvar + log2pi),
                                 axis=raxis)

        with tf.GradientTape() as tape:
            mean, logvar = tf.split(
                self.encoder_model(x), num_or_size_splits=2, axis=1
            )
            eps = tf.random.normal(shape=mean.shape)
            z = eps * tf.exp(logvar * .5) + mean
            x_logit = self.decoder_model(z)
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit, labels=x
            )
            logpx_z = -tf.reduce_sum(entropy, axis=[1, 2, 3])
            logpz = log_normal_pdf(z, .0, .0)
            logqz_x = log_normal_pdf(z, mean, logvar)
            loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)
        tvs = (self.encoder_model.trainable_variables +
               self.decoder_model.trainable_variables)
        grads = tape.gradient(loss, tvs)
        self.optimizer.apply_gradients(zip(grads, tvs))
        self.metric(loss)

    def train(self, epochs, batch_size=None, verbose=True):
        """Trains the keras model.
        params:
            epochs: An integer, which is the number of complete
                    iterations to train
            batch_size: An integer, which is the number of samples
                        per graident update
            verbose: A boolean, which determines the verbositiy level
        """
        length = self.train_data.shape[0]
        batches = tf.data.Dataset.from_tensor_slices(
            self.train_data
        ).shuffle(length).batch(batch_size)
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f'Epoch {epoch}/{epochs}')
            count = 0
            for batch in batches:
                self._train_step(batch)
                count += np.minimum(batch_size, length - count)
                if verbose:
                    print(f'{count}/{length}', end='\r')
            if verbose:
                print(f'{count}/{length} - '
                      f'loss: {self.metric.result()}')
            self.metric.reset_states()

    def load(self, path, custom_objects=None):
        """Loads a encoder and decoder model and weights from a file.
           (overrides the inital provided model)
        params:
            path: A string, which is the path to a folder
                  containing model.json, weights.h5, and note.txt
            optimizer: A string or optimizer instance, which will be
                       the optimizer for the loaded generator model
            custom_objects: A dictionary mapping to custom classes
                            or functions for loading the model
        """
        with open(os.path.join(path, 'encoder_model.json'), 'r') as file:
            self.encoder_model = model_from_json(
                file.read(), custom_objects=custom_objects
            )
            self.encoder_model.compile(optimizer=self.optimizer, loss='mae')
        with open(os.path.join(path, 'decoder_model.json'), 'r') as file:
            self.decoder_model = model_from_json(
                file.read(), custom_objects=custom_objects
            )
            self.decoder_model.compile(optimizer=self.optimizer, loss='mae')
        self.encoder_model.load_weights(
            os.path.join(path, 'encoder_weights.h5')
        )
        self.decoder_model.load_weights(
            os.path.join(path, 'decoder_weights.h5')
        )
        with open(os.path.join(path, 'note.txt'), 'r') as file:
            print(file.read(), end='')

    def save(self, path, note=None):
        """Saves the generator and discriminator model and weights to a file.
        params:
            path: A string, which is the path to create a folder in
                  containing model.json, weights.h5, note.txt,
                  dis_model.json, and dis_weights.h5
            note: A string, which is a note to save in the folder
        return: A string, which is the given path + folder name
        """
        time = datetime.datetime.now()
        path = os.path.join(path, time.strftime(r'%Y%m%d_%H%M%S_%f'))
        os.mkdir(path)
        self.encoder_model.save_weights(
            os.path.join(path, 'encoder_weights.h5')
        )
        self.decoder_model.save_weights(
            os.path.join(path, 'decoder_weights.h5')
        )
        with open(os.path.join(path, 'encoder_model.json'), 'w') as file:
            file.write(self.encoder_model.to_json())
        with open(os.path.join(path, 'decoder_model.json'), 'w') as file:
            file.write(self.decoder_model.to_json())
        with open(os.path.join(path, 'note.txt'), 'w') as file:
            if note is None:
                self.encoder_model.summary(
                    print_fn=lambda line: file.write(line+'\n')
                )
                self.decoder_model.summary(
                    print_fn=lambda line: file.write(line+'\n')
                )
            else:
                file.write(note)
        return path


if __name__ == '__main__':
    import tensorflow_probability as tfp
    import matplotlib.pyplot as plt
    from autoencoder import AutoencoderPredictor

    training = True
    path = None
    latent_dim = 2

    (train_data, _), _ = tf.keras.datasets.mnist.load_data()
    train_data = train_data.reshape((train_data.shape[0], 28, 28, 1)) / 255
    train_data = np.where(train_data > .5, 1.0, 0.0).astype('float32')
    print(len(train_data))

    if training:
        inputs = keras.layers.Input(shape=(28, 28, 1))
        x = inputs
        x = conv2d(32, 3, 2)(x)
        x = conv2d(64, 3, 2)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(latent_dim * 2)(x)
        encoder_model = keras.Model(inputs=inputs, outputs=x)
        encoder_model.compile(
            optimizer=keras.optimizers.Adam(lr=.001, amsgrad=True),
            loss='mae'
        )

        inputs = keras.layers.Input(shape=(latent_dim,))
        x = inputs
        x = dense(7 * 7 * 64)(x)
        x = keras.layers.Reshape((7, 7, 64))(x)
        x = conv2d(64, 3, 2, transpose=True)(x)
        x = conv2d(32, 3, 2, transpose=True)(x)
        x = conv2d(1, 3, transpose=True, activation='linear',
                   batch_norm=False)(x)
        decoder_model = keras.Model(inputs=inputs, outputs=x)
        decoder_model.compile(
            optimizer=keras.optimizers.Adam(lr=.001, amsgrad=True),
            loss='mae'
        )

        trainer = VAETrainer(encoder_model, decoder_model, train_data)
        if path is not None:
            trainer.load(path)
        trainer.train(50, 512)
        path = trainer.save('')
        trainer.load(path)
        del trainer

    predictor = AutoencoderPredictor(path, uses_decoder_model=True)

    n = 20
    norm = tfp.distributions.Normal(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
    image_width = 28 * n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            x_decoded = predictor.predict(np.array([xi, yi]))
            x_decoded = tf.math.sigmoid(x_decoded)
            digit = tf.reshape(x_decoded, (28, 28))
            image[i * 28: (i + 1) * 28,
                  j * 28: (j + 1) * 28] = digit.numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')
    plt.show()
