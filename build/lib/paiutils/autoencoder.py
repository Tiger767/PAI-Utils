"""
Author: Travis Hammond
Version: 12_9_2020
"""


import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json

from paiutils.neural_network import (
    Trainer, Predictor, dense, conv2d, conv1d
)


class AutoencoderTrainer(Trainer):
    """AutoencoderTrainer is used for loading, saving,
       and training keras autoencoder models.
    """

    def __init__(self, encoder_model, decoder_model, data):
        """Initializes train, validation, and test data.
        params:
            encoder_model: The encoder model (full model shares optimizer
                           and other attributes with this model)
            decoder_model: The decoder model
            data: A dictionary containg train data
                  and optionally validation and test data.
                  If the 'train' key is present, the value will
                  be used as a generator and 'train_x' & 'train_y'
                  will be ignored.
                  Ex. {'train_x': [...], 'train_y: [...]}
                  Ex. {'train': generator()}
        """
        if not isinstance(data, dict):
            raise TypeError(
                'data must be a dictionary'
            )
        x0 = keras.layers.Input(shape=encoder_model.input_shape[1:])
        x1 = encoder_model(x0)
        x2 = decoder_model(x1)
        self.model = keras.Model(inputs=x0, outputs=x2)
        self.model.compile(optimizer=encoder_model.optimizer,
                           loss=encoder_model.loss,
                           metrics=encoder_model.compiled_metrics._metrics)
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
            raise ValueError('Invalid data. There must be train data.')
        if 'validation_x' in data:
            self.validation_data = data['validation_x']
            self.validation_data = (self.validation_data,
                                    self.validation_data)
        if 'test_x' in data:
            self.test_data = data['test_x']
            self.test_data = (self.test_data, self.test_data)

    def load(self, path, custom_objects=None):
        """Loads a model and weights from a file.
           (overrides the inital provided model)
        params:
            path: A string, which is the path to a folder
                  containing model.json, weights.h5, note.txt
                  and maybe encoder/decoder parts
            custom_objects: A dictionary mapping to custom classes
                            or functions for loading the model
        """
        optimizer = self.model.optimizer
        loss = self.model.loss
        metrics = self.model.compiled_metrics._metrics
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
            self.decoder_model.compile(optimizer=optimizer, loss=loss,
                                       metrics=metrics)
            self.decoder_model.load_weights(
                os.path.join(path, 'decoder_weights.h5')
            )
        x0 = keras.layers.Input(shape=self.encoder_model.input_shape[1:])
        x1 = self.encoder_model(x0)
        x2 = self.decoder_model(x1)
        self.model = keras.Model(inputs=x0, outputs=x2)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        with open(os.path.join(path, 'note.txt'), 'r') as file:
            print(file.read(), end='')

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


class AutoencoderExtraDecoderTrainer(AutoencoderTrainer):
    """Autoencoder with an Extra Decoder Trainer is an Autoencoder Trainer
       with a extra decoder that can be trained to y-data.
    """

    def __init__(self, encoder_model, decoder_model,
                 extra_decoder_model, data, include_y_data=True):
        """Initializes train, validation, and test data.
        params:
            encoder_model: The encoder model (full model shares optimizer
                           and other attributes with this model)
            decoder_model: The decoder model
            extra_decoder_model: The extra decoder is trained
                                 to map the encoder to a
                                 different output
                                 (not part of the full model)
            data: A dictionary containg train data
                  and optionally validation and test data.
                  Ex. {'train_x': [...], 'train_y: [...]}
            include_y_data: A boolean, which determines if y-data should
                            be appened with the x-data for training the
                            autoencoder
        """
        if not isinstance(data, dict):
            raise TypeError(
                'data must be a dictionary'
            )
        x0 = keras.layers.Input(shape=encoder_model.input_shape[1:])
        x1 = encoder_model(x0)
        x2 = decoder_model(x1)
        self.model = keras.Model(inputs=x0, outputs=x2)
        self.model.compile(optimizer=encoder_model.optimizer,
                           loss=encoder_model.loss,
                           metrics=encoder_model.compiled_metrics._metrics)
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
            raise ValueError('Invalid data. There must be train data.')
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

    def load(self, path, custom_objects=None):
        """Loads a model and weights from a file.
           (overrides the initally provided models)
        params:
            path: A string, which is the path to a folder
                  containing model.json, weights.h5, note.txt,
                  encoder/decoder parts, etc.
            custom_objects: A dictionary mapping to custom classes
                            or functions for loading the model
        """
        super().load(path, custom_objects=custom_objects)
        if 'extra_decoder_model.json' in os.listdir(path):
            optimizer = self.extra_decoder_model.optimizer
            loss = self.extra_decoder_model.loss
            metrics = self.extra_decoder_model.compiled_metrics._metrics
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
                    raise ValueError('Invalid train_data. '
                                     'There must be train data.')
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
            logpx_z = -tf.reduce_sum(
                entropy, axis=tf.range(1, entropy.get_shape().ndims)
            )
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
