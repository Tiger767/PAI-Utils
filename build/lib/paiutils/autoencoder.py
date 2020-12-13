"""
Author: Travis Hammond
Version: 12_13_2020
"""


import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json

from paiutils.neural_network import (
    Trainer, Predictor
)


class AutoencoderTrainer(Trainer):
    """AutoencoderTrainer is used for loading, saving,
       and training keras autoencoder models.
    """

    def __init__(self, encoder_model, decoder_model, data):
        """Initializes train, validation, and test data.
        params:
            encoder_model: The encoder model
            decoder_model: The decoder model (full model shares optimizer
                           and other attributes with this model)
            data: A dictionary containg train data
                  and optionally validation and test data.
                  If the train/validation/test key is present without
                  the _x the value will be used as a
                  generator/Keras-Sequence/TF-Dataset and
                  keys with _x will be ignored.
                  Ex. {'train_x': [...], 'train_y: [...]}
                  Ex. {'train': generator()}
                  Ex. {'train': tf.data.Dataset(), 'test': generator()}
        """
        if not isinstance(data, dict):
            raise TypeError(
                'data must be a dictionary'
            )
        x0 = keras.layers.Input(shape=encoder_model.input_shape[1:])
        x1 = encoder_model(x0)
        x2 = decoder_model(x1)
        self.model = keras.Model(inputs=x0, outputs=x2)
        self.model.compile(optimizer=decoder_model.optimizer,
                           loss=decoder_model.loss,
                           metrics=decoder_model.compiled_metrics._metrics)
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
        return: A string of note.txt
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
            note = file.read()
        return note

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
                file.write('encoder_model\n')
                self.encoder_model.summary(
                    print_fn=lambda line: file.write(line+'\n')
                )
                file.write('\ndecoder_model\n')
                self.decoder_model.summary(
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
            encoder_model: The encoder model
            decoder_model: The decoder model (full model shares optimizer
                           and other attributes with this model)
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
        self.model.compile(optimizer=decoder_model.optimizer,
                           loss=decoder_model.loss,
                           metrics=decoder_model.compiled_metrics._metrics)
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
        train_data2_0 = self.encoder_model.predict(
            self.train_data2[0], batch_size=batch_size
        )
        validation_data2 = None
        if self.validation_data2 is not None:
            validation_data2_0 = self.encoder_model.predict(
                self.validation_data2[0], batch_size=batch_size
            )
            validation_data2 = (validation_data2_0,
                                self.validation_data2[1])
        self.extra_decoder_model.fit(train_data2_0, self.train_data2[1],
                                     validation_data=validation_data2,
                                     batch_size=batch_size, epochs=epochs,
                                     verbose=1 if verbose else 0,
                                     **kwargs)

    def eval_extra_decoder(self, train_data=True, validation_data=True,
                           test_data=True, batch_size=None,
                           verbose=True, **kwargs):
        """Evaluates the extra decoder model with the
           train/validation/test data.
        params:
            train_data: A boolean, which determines if
                        train_data should be evaluated
            validation_data: A boolean, which determines if
                             validation_data should be evaluated
            test_data: A boolean, which determines if
                       test_data should be evaluated
            batch_size: An integer, which is the number of samples
                        per graident update
            verbose: A boolean, which determines the verbositiy level
        return: A dictionary of the results
        """
        verbose = 1 if verbose else 0

        to_eval = []
        if train_data:
            to_eval.append(('Train', self.train_data2))
        if validation_data:
            to_eval.append(('Validation', self.validation_data2))
        if test_data:
            to_eval.append(('Test', self.test_data2))

        results = {}
        for name, data in to_eval:
            if verbose == 1:
                print(f'{name} Data Evaluation: ')
            if data is not None:
                data_0 = self.encoder_model.predict(
                    data[0], batch_size=batch_size
                )
                results[name] = self.extra_decoder_model.evaluate(
                    data_0, data[1], batch_size=batch_size,
                    verbose=verbose, **kwargs
                )
        return results

    def load(self, path, custom_objects=None):
        """Loads a model and weights from a file.
           (overrides the initally provided models)
        params:
            path: A string, which is the path to a folder
                  containing model.json, weights.h5, note.txt,
                  encoder/decoder parts, etc.
            custom_objects: A dictionary mapping to custom classes
                            or functions for loading the model
        return: A string of note.txt
        """
        note = super().load(path, custom_objects=custom_objects)
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
        return note

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
        if note is None:
            with open(os.path.join(path, 'note.txt'), 'a') as file:
                file.write('\nextra_decoder_model\n')
                self.extra_decoder_model.summary(
                    print_fn=lambda line: file.write(line+'\n')
                )
        return path


class VAETrainer(AutoencoderTrainer):
    """VAETrainer is used for loading, saving,
       and training keras variational autoencoder models.
    """

    class VAEModel(keras.Model):
        def __init__(self, encoder, decoder, use_logits=True, **kwargs):
            """VAE Keras Model that has a modified train_step.
            params:
                encoder_model: The encoder model
                decoder_model: The decoder model (full model shares optimizer
                               and other attributes with this model)
                use_logits: A boolean that determines if binary crossentropy
                            should be used with logit inputs (decoder_model
                            loss will be ignored for training)
            """
            super(VAETrainer.VAEModel, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.use_logits = use_logits
            self.decoder.compiled_loss.build(
                tf.zeros(self.decoder.output_shape[1:])
            )
            self.rloss = self.decoder.compiled_loss._losses[0]

        def train_step(self, x):
            """Trains the model 1 step.
            params:
                x: A tensor or tuple
            """
            if isinstance(x, tuple):
                x = x[0]

            with tf.GradientTape() as tape:
                z_mean, z_log_var = self.encoder(x, training=True)
                eps = tf.random.normal(shape=tf.shape(z_mean))
                z = eps * tf.exp(z_log_var * .5) + z_mean
                if self.use_logits:
                    reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(  # noqa
                        logits=self.decoder(z, training=True), labels=x
                    )
                else:
                    reconstruction_loss = self.rloss.fn(
                        x, self.decoder(z, training=True)
                    )
                reconstruction_loss = tf.reduce_sum(
                    reconstruction_loss,
                    axis=tf.range(1, reconstruction_loss.get_shape().ndims)
                )
                reconstruction_loss = tf.reduce_mean(
                    reconstruction_loss
                )
                log2pi = tf.math.log(2. * np.pi)
                logpz = tf.reduce_sum(.5 * (z ** 2. + log2pi), axis=1)
                logqz_x = tf.reduce_sum(
                    -.5 * ((z - z_mean) ** 2. *
                           tf.exp(-z_log_var) + z_log_var + log2pi),
                    axis=1
                )
                divergence_loss = tf.reduce_mean(
                    logqz_x + logpz
                )
                loss = reconstruction_loss + divergence_loss
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return {
                'loss': loss,
                'reconstruction_loss': reconstruction_loss,
                'divergence_loss': divergence_loss
            }

        def call(self, x, training=False):
            """Calls the model on new inputs.
            params:
                inputs: A tensor or list of tensors
                training: A boolean or boolean scalar tensor, indicating
                          whether to run the `Network` in training mode
                          or inference mode
            return: A tensor if there is a single output, or a list of
                    tensors if there are more than one outputs.
            """
            z_mean, z_log_var = self.encoder(x, training=training)
            eps = tf.random.normal(shape=tf.shape(z_mean))
            z = eps * tf.exp(z_log_var * .5) + z_mean
            y = self.decoder(z, training=training)
            if self.use_logits:
                y = tf.math.sigmoid(y)
            return y

    def __init__(self, encoder_model, decoder_model, data, use_logits=True):
        """Initializes train, validation, and test data.
        params:
            encoder_model: The encoder model
            decoder_model: The decoder model (full model shares optimizer
                           and other attributes with this model)
            data: A dictionary containg train data
                  and optionally validation and test data.
                  If the train/validation/test key is present without
                  the _x the value will be used as a
                  generator/Keras-Sequence/TF-Dataset and
                  keys with _x will be ignored.
                  Ex. {'train_x': [...], 'train_y: [...]}
                  Ex. {'train': generator()}
                  Ex. {'train': tf.data.Dataset(), 'test': generator()}
            use_logits: A boolean that determines if binary crossentropy
                        should be used with logit inputs (decoder_model
                        loss will be ignored for training)
        """
        if not isinstance(data, dict):
            raise TypeError(
                'data must be a dictionary'
            )
        self.model = VAETrainer.VAEModel(
            encoder_model, decoder_model, use_logits=use_logits
        )
        self.model.compile(optimizer=decoder_model.optimizer,
                           loss=decoder_model.loss,
                           metrics=decoder_model.compiled_metrics._metrics)
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
        """Loads a encoder and decoder model and weights from a file.
           (overrides the inital provided model)
        params:
            path: A string, which is the path to a folder
                  containing model.json, weights.h5, note.txt, etc.
            custom_objects: A dictionary mapping to custom classes
                            or functions for loading the model
        return: A string of note.txt
        """
        optimizer = self.model.optimizer
        loss = self.model.loss
        metrics = self.model.compiled_metrics._metrics
        if 'encoder_model.json' in os.listdir(path):
            with open(os.path.join(path, 'encoder_model.json'), 'r') as file:
                self.encoder_model = model_from_json(
                    file.read(), custom_objects=custom_objects
                )
            self.encoder_model.load_weights(
                os.path.join(path, 'encoder_weights.h5')
            )
        if 'decoder_model.json' in os.listdir(path):
            with open(os.path.join(path, 'decoder_model.json'), 'r') as file:
                self.decoder_model = model_from_json(
                    file.read(), custom_objects=custom_objects
                )
            self.decoder_model.load_weights(
                os.path.join(path, 'decoder_weights.h5')
            )
            self.decoder_model.compile(loss=loss)
        self.model = VAETrainer.VAEModel(self.encoder_model,
                                         self.decoder_model,
                                         use_logits=self.model.use_logits)
        self.model.compile(optimizer=optimizer, loss=loss,
                           metrics=metrics)

        with open(os.path.join(path, 'note.txt'), 'r') as file:
            note = file.read()
        return note

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

        x0 = keras.layers.Input(shape=self.model.encoder.input_shape[1:])
        z_mean, z_log_var = self.model.encoder(x0)
        eps = tf.random.normal(shape=tf.shape(z_mean))
        z = eps * tf.exp(z_log_var * .5) + z_mean
        x1 = self.model.decoder(z)
        if self.model.use_logits:
            x1 = tf.math.sigmoid(x1)
        model = keras.Model(inputs=x0, outputs=x1)

        model.save_weights(os.path.join(path, 'weights.h5'))
        with open(os.path.join(path, 'model.json'), 'w') as file:
            file.write(model.to_json())
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
                file.write('encoder_model\n')
                self.encoder_model.summary(
                    print_fn=lambda line: file.write(line+'\n')
                )
                file.write('\ndecoder_model\n')
                self.decoder_model.summary(
                    print_fn=lambda line: file.write(line+'\n')
                )
            else:
                file.write(note)
        return path
