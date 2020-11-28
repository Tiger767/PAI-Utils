"""
Author: Travis Hammond
Version: 11_28_2020
"""


import os
import datetime
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json

try:
    from paiutils.neural_network import (
        Trainer, Predictor, dense, conv2d, conv1d
    )
except ImportError:
    from neural_network import (
        Trainer, Predictor, dense, conv2d, conv1d
    )

class AutoencoderTrainer(Trainer):
    """AutoencoderTrainer is used for loading, saving,
       and training keras autoencoder models.
    """

    def __init__(self, model, data, encoder_model=None, decoder_model=None):
        """Initializes train, validation, and test data.
        params:
            model: A compiled full keras model
            data: A dictionary containg train data
                  and optionally validation and test data.
                  If the 'train' key is present, the value will
                  be used as a generator and 'train_x' & 'train_y'
                  will be ignored.
                  Ex. {'train_x': [...], 'train_y: [...]}
                  Ex. {'train': generator()}
            encoder_model: The encoder part of the full model
            decoder_model: The decoder part of the full model
        """
        if not isinstance(data, dict):
            raise TypeError(
                'data must be a dictionary'
            )
        self.model = model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.train_data = None
        self.validation_data = None
        self.test_data = None

        if 'train_x' in data:
            self.train_data = data['train_x']
            self.train_data = (self.train_data, self.train_data)
        elif 'train' in data:
            self.train_data = data['train']
        else:
            raise Exception('There must be train data')
        if 'validation_x' in data:
            self.validation_data = data['validation_x']
            self.validation_data = (self.validation_data,
                                    self.validation_data)
        if 'test_x' in data:
            self.test_data = data['test_x']
            self.test_data = (self.test_data, self.test_data)

    def train(self, epochs, batch_size=None, verbose=True, **kwargs):
        """Trains the keras model.
        params:
            epochs: An integer, which is the number of complete
                    iterations to train
            batch_size: An integer, which is the number of samples
                        per graident update
            verbose: A boolean, which determines the verbositiy level
        """
        super().train(epochs, batch_size=batch_size,
                      verbose=verbose, **kwargs)
        weights = self.model.get_weights()
        if self.encoder_model is not None:
            encoder_num_weights = len(self.encoder_model.get_weights())
            self.encoder_model.set_weights(weights[:encoder_num_weights])
        if self.decoder_model is not None:
            decoder_num_weights = len(self.decoder_model.get_weights())
            self.decoder_model.set_weights(weights[-decoder_num_weights:])

    def load(self, path, optimizer, loss, metrics=None, custom_objects=None):
        """Loads a model and weights from a file.
           (overrides the inital provided model)
        params:
            path: A string, which is the path to a folder
                  containing model.json, weights.h5, note.txt
                  and maybe encoder/decoder parts
            optimizer: A string or optimizer instance, which will be
                       the optimizer for the loaded model
            loss: A string or loss instance, which will be
                  the loss function for the loaded model
            metrics: A list of metrics, which will be used
                     by the loaded model
            custom_objects: A dictionary mapping to custom classes
                            or functions for loading the model
        """
        super().load(path, optimizer, loss, metrics=metrics,
                     custom_objects=custom_objects)
        if 'encoder_model.json' in os.listdir(path):
            with open(os.path.join(path, 'encoder_model.json'), 'r') as file:
                self.encoder_model = model_from_json(
                    file.read(), custom_objects=custom_objects
                )
                self.encoder_model.compile(optimizer=optimizer, loss=loss,
                                           metrics=metrics)
            self.encoder_model.load_weights(
                os.path.join(path, 'encoder_weights.h5')
            )
        if 'decoder_model.json' in os.listdir(path):
            with open(os.path.join(path, 'decoder_model.json'), 'r') as file:
                self.decoder_model = model_from_json(
                    file.read(), custom_objects=custom_objects
                )
                self.decoder_model.compile(optimizer=optimizer, loss=loss)
            self.decoder_model.load_weights(
                os.path.join(path, 'decoder_weights.h5')
            )

    def save(self, path, note=None):
        """Saves the model and weights to a file.
        params:
            path: A string, which is the path to create a folder in
                  containing model.json, weights.h5, note.txt, and
                  maybe encoder/decoder parts
            note: A string, which is a note to save in the folder
        return: A string, which is the given path + folder name
        """
        time = datetime.datetime.now()
        path = os.path.join(path, time.strftime(r'%Y%m%d_%H%M%S_%f'))
        os.mkdir(path)
        self.model.save_weights(os.path.join(path, 'weights.h5'))
        with open(os.path.join(path, 'model.json'), 'w') as file:
            file.write(self.model.to_json())
        if self.encoder_model is not None:
            self.encoder_model.save_weights(
                os.path.join(path, 'encoder_weights.h5')
            )
            with open(os.path.join(path, 'encoder_model.json'), 'w') as file:
                file.write(self.encoder_model.to_json())
        if self.decoder_model is not None:
            self.decoder_model.save_weights(
                os.path.join(path, 'decoder_weights.h5')
            )
            with open(os.path.join(path, 'decoder_model.json'), 'w') as file:
                file.write(self.decoder_model.to_json())
        with open(os.path.join(path, 'note.txt'), 'w') as file:
            if note is None:
                self.model.summary(
                    print_fn=lambda line: file.write(line+'\n')
                )
            else:
                file.write(note)
        return path


class AutoencoderPredictor(Predictor):
    """AutoenocderPredictor is used for loading and predicting keras models."""

    def __init__(self, path, uses_encoder_model=False,
                 uses_decoder_model=False):
        """Initializes the model and weights.
        params:
            path: A string, which is the path to a folder containing
                  model.json, weights.h5, note.txt, and maybe encoder/decoder
                  parts
            uses_encoder_model: A boolean, which determines if encoder model
                                should be used for predictions
                                (cannot also enable uses_decoder_model)
            uses_decoder_model: A boolean, which determines if decoder model
                                should be used for predictions
                                (cannot also enable uses_encoder_model)
        """
        if uses_encoder_model:
            super().__init__(path, 'encoder_weights.h5',
                             'encoder_model.json')
        elif uses_decoder_model:
            super().__init__(path, 'decoder_weights.h5',
                             'decoder_model.json')
        else:
            super().__init__(path)


class AutoencoderExtraDecoderTrainer(Trainer):
    """Autoencoder with an Extra Decoder Trainer is an Autoencoder Trainer
       with a extra decoder that can be trained to y-data.
    """

    def __init__(self, model, encoder_model, decoder_model,
                 extra_decoder_model, data, include_y_data=True):
        """Initializes train, validation, and test data.
        params:
            model: A compiled full keras model
            encoder_model: The encoder part of the full model
            decoder_model: The decoder part of the full model
            extra_decoder_model: The extra decoder is trained
                                 to map the encoder to a
                                 different output
                                 (not part of the full model)
            data: A dictionary containg train data
                  and optionally validation and test data.
            include_y_data: A boolean, which determines if y-data should
                            be appened with the x-data for training the
                            autoencoder
        """
        if not isinstance(data, dict):
            raise TypeError(
                'data must be a dictionary'
            )
        self.model = model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.extra_decoder_model = extra_decoder_model
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.train_data2 = None
        self.validation_data2 = None
        self.test_data2 = None

        if 'train_x' in data and 'train_y' in data:
            if include_y_data:
                self.train_data = np.vstack([data['train_x'],
                                             data['train_y']])
            else:
                self.train_data = data['train_x']
            self.train_data = (self.train_data, self.train_data)
            self.train_data2 = [data['train_x'], data['train_y']]
        else:
            raise Exception('There must be train data')
        if 'validation_x' in data and 'validation_y' in data:
            if include_y_data:
                self.validation_data = np.vstack([data['validation_x'],
                                                data['validation_y']])
            else:
                self.validation_data = data['validation_x']
            self.validation_data = (self.validation_data,
                                    self.validation_data)
            self.validation_data2 = [data['validation_x'],
                                        data['validation_y']]
        if 'test_x' in data and 'test_y' in data:
            if include_y_data:
                self.test_data = np.vstack([data['test_x'],
                                            data['test_y']])
            else:
                self.test_data = data['test_x']
            self.test_data = (self.test_data, self.test_data)
            self.test_data2 = [data['test_x'], data['test_y']]

    def train_extra_decoder(self, epochs, batch_size=None,
                            verbose=True, **kwargs):
        """Trains the extra decoder keras model on the outputs
           of the assumingly trained encoder.
        params:
            epochs: An integer, which is the number of complete
                    iterations to train
            batch_size: An integer, which is the number of samples
                        per graident update
            callbacks: A list of keras Callback instances,
                       which are called during training and validation
            verbose: A boolean, which determines the verbositiy level
        """
        train_data2_x = self.encoder_model.predict(
            self.train_data2[0], batch_size=batch_size
        )
        validation_data2 = None
        if self.validation_data2 is not None:
            validation_data2_x = self.encoder_model.predict(
                self.validation_data2[0], batch_size=batch_size
            )
            validation_data2 = (validation_data2_x,
                                self.validation_data2[1])
        self.extra_decoder_model.fit(train_data2_x, self.train_data2[1],
                                     validation_data=validation_data2,
                                     batch_size=batch_size, epochs=epochs,
                                     verbose=1 if verbose else 0,
                                     **kwargs)
        if verbose:
            print('Extra Decoder Train Data Evaluation: ', end='')
            print(self.extra_decoder_model.evaluate(train_data2_x,
                                                    self.train_data2[1],
                                                    batch_size=batch_size,
                                                    verbose=0))
            if self.validation_data2 is not None:
                print('Extra Decoder Validation Data Evaluation: ', end='')
                print(self.extra_decoder_model.evaluate(validation_data2[0],
                                                        validation_data2[1],
                                                        batch_size=batch_size,
                                                        verbose=0))
            if self.test_data2 is not None:
                test_data2_x = self.encoder_model.predict(
                    self.test_data2[0], batch_size=batch_size
                )
                print('Extra Decoder Test Data Evaluation: ', end='')
                print(self.extra_decoder_model.evaluate(test_data2_x,
                                                        self.test_data2[1],
                                                        batch_size=batch_size,
                                                        verbose=0))

    def load(self, path, optimizer, loss, metrics=None, custom_objects=None):
        """Loads a model and weights from a file.
           (overrides the inital provided model)
        params:
            path: A string, which is the path to a folder
                  containing model.json, weights.h5, note.txt
                  and maybe encoder/decoder parts
            optimizer: A string or optimizer instance, which will be
                       the optimizer for the loaded model
            loss: A string or loss instance, which will be
                  the loss function for the loaded model
            metrics: A list of metrics, which will be used
                     by the loaded model
            custom_objects: A dictionary mapping to custom classes
                            or functions for loading the model
        """
        super().load(path, optimizer, loss, metrics=metrics,
                     custom_objects=custom_objects)
        if 'extra_decoder_model.json' in os.listdir(path):
            edm_path = os.path.join(path, 'extra_decoder_model.json')
            with open(edm_path, 'r') as file:
                self.extra_decoder_model = model_from_json(
                    file.read(), custom_objects=custom_objects
                )
                self.extra_decoder_model.compile(optimizer=optimizer,
                                                 loss=loss,
                                                 metrics=metrics)
            self.extra_decoder_model.load_weights(
                os.path.join(path, 'extra_decoder_weights.h5')
            )

    def save(self, path, note=None):
        """Saves the model and weights to a file.
        params:
            path: A string, which is the path to create a folder in
                  containing model.json, weights.h5, note.txt, and
                  maybe encoder/decoder parts
            note: A string, which is a note to save in the folder
        return: A string, which is the given path + folder name
        """
        path = super().save(path, note=note)
        self.extra_decoder_model.save_weights(
            os.path.join(path, 'extra_decoder_weights.h5')
        )
        edm_path = os.path.join(path, 'extra_decoder_model.json')
        with open(edm_path, 'w') as file:
            file.write(self.extra_decoder_model.to_json())
        return path


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


def create_basic_dense_model(input_shape, units_list,
                             activation='relu',
                             output_activation='sigmoid',
                             dropout=0, batch_norm=True,
                             encoder_output_activation=None,
                             encoder_output_batch_norm=None):
    """Creates a basic dense model for mainly testing purposes.
    params:
        input_shape: A tuple of integers, which is the expected input shape
        units_list: A list of integers, which are the dimensionality of the
                    output space
        activation: A string or keras/TF activation function
        output_activation: A string or keras/TF activation function
        dropout: A float, which is the dropout rate between inputs and
                 first dense layer
        batch_norm: A boolean, which determines if batch
                    normalization is enabled
        encoder_output_activation: A string or keras/TF activation function
        encoder_output_batch_norm: A boolean, which determines if batch
                                   normalization is enabled for the encoder
                                   output
    return: A tuple of the full model, the encoder model, and decoder model
            uncompiled
    """
    if encoder_output_activation is None:
        encoder_output_activation = activation

    if encoder_output_batch_norm is None:
        encoder_output_batch_norm = batch_norm

    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    if len(input_shape) > 1:
        x = keras.layers.Flatten()(x)
    if dropout > 0:
        x = keras.layers.Dropout(dropout)(x)
    for units in units_list[:-1]:
        x = dense(units, activation=activation, batch_norm=batch_norm)(x)
    x = dense(units_list[-1], activation=encoder_output_activation,
              batch_norm=encoder_output_batch_norm)(x)
    encoder_model = keras.Model(inputs=inputs, outputs=x)
    decoder_inputs = keras.layers.Input(shape=(units_list[-1],))
    decoder_start_ndx = len(encoder_model.layers)
    for units in reversed(units_list):
        x = dense(units, activation=activation, batch_norm=batch_norm)(x)
    if len(input_shape) > 1:
        outputs = dense(np.prod(input_shape), activation=output_activation,
                        batch_norm=False)(x)
        outputs = keras.layers.Reshape(input_shape)(outputs)
    else:
        outputs = dense(input_shape[0], activation=output_activation,
                        batch_norm=False)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    x = decoder_inputs
    for layer in model.layers[decoder_start_ndx:]:
        x = layer(x)
    decoder_model = keras.Model(inputs=decoder_inputs, outputs=x)
    return model, encoder_model, decoder_model


def create_basic_conv2d_model(input_shape, filters_list, kernel_sizes,
                              max_pools, activation='relu',
                              output_activation='sigmoid',
                              dropout=0, batch_norm=True,
                              encoder_output_activation=None,
                              encoder_output_batch_norm=None):
    """Creates a basic 2D convolution model for mainly testing purposes.
    params:
        input_shape: A tuple of integers, which is the expected input shape
        filters_list: A list of integers, which are the dimensionality of the
                      output space
        kernel_sizes: An integer or tuple of 2 integers, which is the size of
                     the convoluition kernel
        max_pools: A list of integers or tuples, which are the size of the
                   pooling windows
        activation: A string or keras/TF activation function
        output_activation: A string or keras/TF activation function
        dropout: A float, which is the dropout rate between inputs and
                 first dense layer
        batch_norm: A boolean, which determines if batch
                    normalization is enabled
        encoder_output_activation: A string or keras/TF activation function
        encoder_output_batch_norm: A boolean, which determines if batch
                                   normalization is enabled for the encoder
                                   output
    return: A tuple of the full model, the encoder model, and decoder model
            uncompiled
    """
    if encoder_output_activation is None:
        encoder_output_activation = activation
    if encoder_output_batch_norm is None:
        encoder_output_batch_norm = batch_norm
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    if dropout > 0:
        x = keras.layers.Dropout(dropout)(x)
    for ndx in range(len(filters_list)-1):
        x = conv2d(filters_list[ndx], kernel_sizes[ndx], activation=activation,
                   max_pool_size=max_pools[ndx], batch_norm=batch_norm)(x)
    x = conv2d(filters_list[-1], kernel_sizes[-1],
               activation=encoder_output_activation,
               max_pool_size=max_pools[-1],
               batch_norm=encoder_output_batch_norm)(x)
    encoder_model = keras.Model(inputs=inputs, outputs=x)
    decoder_inputs = keras.layers.Input(shape=[y for y in x.shape[1:]])
    decoder_start_ndx = len(encoder_model.layers)
    for ndx in reversed(range(len(filters_list))):
        x = conv2d(filters_list[ndx], kernel_sizes[ndx], activation=activation,
                   batch_norm=batch_norm, upsampling_size=max_pools[ndx])(x)
    outputs = conv2d(input_shape[-1], activation=output_activation,
                     batch_norm=False)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    x = decoder_inputs
    for layer in model.layers[decoder_start_ndx:]:
        x = layer(x)
    decoder_model = keras.Model(inputs=decoder_inputs, outputs=x)
    return model, encoder_model, decoder_model


def create_basic_conv1d_model(input_shape, filters_list, kernel_sizes,
                              max_pools, activation='relu',
                              output_activation='sigmoid',
                              dropout=0, batch_norm=True,
                              encoder_output_activation=None,
                              encoder_output_batch_norm=None):
    """Creates a basic 1D convolution model for mainly testing purposes.
    params:
        input_shape: A tuple of integers, which is the expected input shape
        filters_list: A list of integers, which are the dimensionality of the
                      output space
        kernel_sizes: An integer or tuple of 1 integers, which is the size of
                     the convoluition kernel
        max_pools: A list of integers or tuples, which are the size of the
                   pooling windows
        activation: A string or keras/TF activation function
        output_activation: A string or keras/TF activation function
        dropout: A float, which is the dropout rate between inputs and
                 first dense layer
        batch_norm: A boolean, which determines if batch
                    normalization is enabled
        encoder_output_activation: A string or keras/TF activation function
        encoder_output_batch_norm: A boolean, which determines if batch
                                   normalization is enabled for the encoder
                                   output
    return: A tuple of the full model, the encoder model, and decoder model
            uncompiled
    """
    if encoder_output_activation is None:
        encoder_output_activation = activation
    if encoder_output_batch_norm is None:
        encoder_output_batch_norm = batch_norm
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    if dropout > 0:
        x = keras.layers.Dropout(dropout)(x)
    for ndx in range(len(filters_list)-1):
        x = conv1d(filters_list[ndx], kernel_sizes[ndx], activation=activation,
                   max_pool_size=max_pools[ndx], batch_norm=batch_norm)(x)
    x = conv1d(filters_list[-1], kernel_sizes[-1],
               activation=encoder_output_activation,
               max_pool_size=max_pools[-1],
               batch_norm=encoder_output_batch_norm)(x)
    encoder_model = keras.Model(inputs=inputs, outputs=x)
    decoder_inputs = keras.layers.Input(shape=[y for y in x.shape[1:]])
    decoder_start_ndx = len(encoder_model.layers)
    for ndx in reversed(range(len(filters_list))):
        x = conv1d(filters_list[ndx], kernel_sizes[ndx], activation=activation,
                   batch_norm=batch_norm, upsampling_size=max_pools[ndx])(x)
    outputs = conv1d(input_shape[-1], activation=output_activation,
                     batch_norm=False)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    x = decoder_inputs
    for layer in model.layers[decoder_start_ndx:]:
        x = layer(x)
    decoder_model = keras.Model(inputs=decoder_inputs, outputs=x)
    return model, encoder_model, decoder_model


if __name__ == '__main__':
    import image as img
    from time import sleep

    training = True
    uses_encoder_model = False
    uses_decoder_model = True
    uses_conv_model = True

    (tx, _), (vx, _) = keras.datasets.fashion_mnist.load_data()
    tx2, vx2 = [], []
    # Enlarge so that after downsamping and upsampling
    # the shape will be same as the input shape
    for x in tx:
        tx2.append(img.resize(x, (32, 32)))
    for x in vx:
        vx2.append(img.resize(x, (32, 32)))
    del tx, vx
    tx2 = np.expand_dims(np.array(tx2), axis=-1)
    vx2 = np.expand_dims(np.array(vx2), axis=-1)
    dataset = {'train_x': tx2, 'validation_x': vx2}

    path = 'conv_weights' if uses_conv_model else 'dense_weights'

    if training:
        if uses_conv_model:
            models = create_basic_conv2d_model(
                (32, 32, 1), [256, 128, 64, 32, 16],
                [3, 3, 3, 3, 3], [2, 2, 2, 2, 2],
                activation='relu', output_activation='linear',
                dropout=0, batch_norm=True,
                encoder_output_activation='sigmoid',
                encoder_output_batch_norm=False
            )
        else:
            models = create_basic_dense_model(
                (32, 32, 1), [512, 256, 128, 64, 32, 16],
                activation='relu', output_activation='linear',
                dropout=0, batch_norm=True,
                encoder_output_activation='sigmoid',
                encoder_output_batch_norm=False
            )
        model, encoder, decoder = models
        model.compile(optimizer=keras.optimizers.Adam(lr=.01, amsgrad=True),
                      loss='mse')
        model.summary()

        trainer = AutoencoderTrainer(model, dataset, encoder_model=encoder,
                                     decoder_model=decoder)
        trainer.train(20, batch_size=32)
        path = trainer.save('')

    predictor = AutoencoderPredictor(
        path, uses_encoder_model=uses_encoder_model,
        uses_decoder_model=uses_decoder_model
    )

    ws = img.Windows()
    ws.start()
    w = ws.add('Image')

    if uses_encoder_model:
        for vx in dataset['validation_x']:
            out = predictor.predict(vx)
            real_img = np.squeeze(vx).astype(np.uint8)
            ws.set(w, real_img)
            print(out.round(3), out.shape)
            sleep(.01)
    elif uses_decoder_model:
        while True:
            shape = predictor.model.layers[0].input_shape[0][1:]
            inputs = np.random.random(shape)
            print(inputs.round(3))
            out = predictor.predict(inputs)
            out_img = np.clip(np.squeeze(out), 0, 255).astype(np.uint8)
            ws.set(w, out_img)
            sleep(.01)
    else:
        for vx in dataset['validation_x']:
            out = predictor.predict(vx)
            out_img = np.clip(np.squeeze(out), 0, 255).astype(np.uint8)
            real_img = np.squeeze(vx).astype(np.uint8)
            ws.set(w, np.hstack((out_img, real_img)))
            sleep(.01)

    ws.stop()
