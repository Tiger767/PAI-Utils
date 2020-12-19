"""
Author: Travis Hammond
Version: 12_19_2020
"""

from types import GeneratorType
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.regularizers import l1_l2


class Trainer:
    """Trainer is used for loading, saving, and training keras models."""
    GEN_DATA_TYPES = (
        GeneratorType, keras.utils.Sequence, tf.data.Dataset
    )

    def __init__(self, model, data):
        """Initializes the model and the train/validation/test data.
        params:
            model: A compiled keras model
            data: A dictionary containg train data
                  and optionally validation and test data.
                  If the train/validation/test key is present without
                  the _x/_y the value will be used as a
                  generator/Keras-Sequence/TF-Dataset and
                  keys with _x/_y will be ignored.
                  Ex. {'train_x': [...], 'train_y: [...]}
                  Ex. {'train': generator(), 'test': [...]}
                  Ex. {'train': tf.data.Dataset(), 'test': generator()}
        """
        self.model = model
        self.model_names = ['model']
        self.set_data(data)

    def set_data(self, data):
        """Sets train, validation, and test data from data.
        params:
            data: A dictionary containg train data
                  and optionally validation and test data.
                  If the train/validation/test key is present without
                  the _x/_y the value will be used as a
                  generator/Keras-Sequence/TF-Dataset and
                  keys with _x/_y will be ignored.
                  Ex. {'train_x': [...], 'train_y: [...]}
                  Ex. {'train': generator(), 'test': [...]}
                  Ex. {'train': tf.data.Dataset(), 'test': generator()}
        """
        if not isinstance(data, dict):
            raise TypeError(
                'data must be a dictionary'
            )
        self.train_data = None
        self.validation_data = None
        self.test_data = None

        if 'train_x' in data and 'train_y' in data:
            self.train_data = (data['train_x'], data['train_y'])
        elif 'train' in data:
            if isinstance(data['train'], Trainer.GEN_DATA_TYPES):
                self.train_data = data['train']
            else:
                raise ValueError(
                    f'train data must be of type {Trainer.GEN_DATA_TYPES}. '
                    f'Use train_x/_y for keys if using ndarrays.'
                )
        else:
            raise ValueError('Invalid data. There must be train data.')
        if 'validation_x' in data and 'validation_y' in data:
            self.validation_data = (data['validation_x'],
                                    data['validation_y'])
        elif 'validation' in data:
            if isinstance(data['validation'], Trainer.GEN_DATA_TYPES):
                self.validation_data = data['validation']
            else:
                raise ValueError(
                    f'validation data must be of type {Trainer.GEN_DATA_TYPES}'
                    f'. Use validation_x/_y for keys if using ndarrays.'
                )
        if 'test_x' in data and 'test_y' in data:
            self.test_data = (data['test_x'], data['test_y'])
        elif 'test' in data:
            if isinstance(data['test'], Trainer.GEN_DATA_TYPES):
                self.test_data = data['test']
            else:
                raise ValueError(
                    f'test data must be of type {Trainer.GEN_DATA_TYPES}. '
                    f'Use test_x/_y for keys if using ndarrays.'
                )

    def train(self, epochs, batch_size=None, verbose=True, **kwargs):
        """Trains the keras model.
        params:
            epochs: An integer, which is the number of complete
                    iterations to train
            batch_size: An integer, which is the number of samples
                        per graident update
            verbose: A boolean, which determines the verbositiy level
        """
        verbose = 1 if verbose else 0
        if isinstance(self.train_data, Trainer.GEN_DATA_TYPES):
            if 'steps_per_epoch' not in kwargs and batch_size is not None:
                kwargs['steps_per_epoch'] = batch_size
            self.model.fit(self.train_data,
                           validation_data=self.validation_data,
                           epochs=epochs,
                           verbose=verbose, **kwargs)
        else:
            self.model.fit(self.train_data[0], self.train_data[1],
                           validation_data=self.validation_data,
                           batch_size=batch_size, epochs=epochs,
                           verbose=verbose, **kwargs)

    def eval(self, train_data=True, validation_data=True,
             test_data=True, batch_size=None, verbose=True, **kwargs):
        """Evaluates the model with the train/validation/test data.
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
            to_eval.append(('Train', self.train_data))
        if validation_data:
            to_eval.append(('Validation', self.validation_data))
        if test_data:
            to_eval.append(('Test', self.test_data))

        results = {}
        for name, data in to_eval:
            if data is not None:
                if verbose == 1:
                    print(f'{name} Data Evaluation: ')
                if isinstance(data, Trainer.GEN_DATA_TYPES):
                    if 'steps' not in kwargs and batch_size is not None:
                        kwargs['steps'] = batch_size
                    results[name] = self.model.evaluate(
                        data, verbose=verbose, **kwargs
                    )
                else:
                    results[name] = self.model.evaluate(
                        data[0], data[1], batch_size=batch_size,
                        verbose=verbose, **kwargs
                    )
        return results

    def load(self, path, custom_objects=None):
        """Loads models and weights from a folder.
           (overrides the inital provided model)
        params:
            path: A string, which is the path to a folder
                  containing model.json, model_weights.h5, note.txt, etc.
            custom_objects: A dictionary mapping to custom classes
                            or functions for loading the model
        return: A string of note.txt
        """
        for name in self.model_names:
            optimizer = self.__dict__[name].optimizer
            loss = self.__dict__[name].loss
            metrics = self.__dict__[name].compiled_metrics._metrics
            with open(os.path.join(path, f'{name}.json'), 'r') as file:
                self.__dict__[name] = model_from_json(
                    file.read(), custom_objects=custom_objects
                )
            self.__dict__[name].compile(
                optimizer=optimizer, loss=loss, metrics=metrics
            )
            self.__dict__[name].load_weights(
                os.path.join(path, f'{name}_weights.h5')
            )
        with open(os.path.join(path, 'note.txt'), 'r') as file:
            note = file.read()
        return note

    def save(self, path, note=None):
        """Saves the models and weights to a new folder.
        params:
            path: A string, which is the path to create a folder in
                  containing model.json, model_weights.h5, note.txt, etc.
            note: A string, which is a note to save in the folder
        return: A string, which is the given path + created folder
        """
        time = datetime.datetime.now()
        path = os.path.join(path, time.strftime(r'%Y%m%d_%H%M%S_%f'))
        os.mkdir(path)
        for name in self.model_names:
            self.__dict__[name].save_weights(
                os.path.join(path, f'{name}_weights.h5')
            )

            with open(os.path.join(path, f'{name}.json'), 'w') as file:
                file.write(self.__dict__[name].to_json())

        with open(os.path.join(path, 'note.txt'), 'w') as file:
            if note is None:
                for name in self.model_names:
                    file.write(f'{name}\n')
                    self.__dict__[name].summary(
                        print_fn=lambda line: file.write(line+'\n')
                    )
                    file.write('\n')
            else:
                file.write(note)
        return path


class Predictor:
    """Predictor is used for loading and predicting keras models."""

    def __init__(self, path, weights_name='model_weights.h5',
                 model_name='model.json', custom_objects=None):
        """Initializes the model and weights.
        params:
            path: A string, which is the path to a folder containing
                  model.json, weights.h5, and maybe note.txt
            weights_name: A string, which is the name of the weights to load
            model_name: A string, which is the name of the model to load
            custom_objects: A dictionary mapping to custom classes
                            or functions for loading the model
        """
        with open(os.path.join(path, model_name), 'r') as file:
            self.model = model_from_json(
                file.read(), custom_objects=custom_objects
            )
        self.model.load_weights(os.path.join(path, weights_name))
        note_path = os.path.join(path, 'note.txt')
        if os.path.exists(note_path):
            with open(note_path, 'r') as file:
                self.note = file.read()

    def predict(self, x):
        """Predicts on a single sample.
        params:
            x: A ndarray or list/tuple/dict of ndarrays
        return: A result from the model output
        """
        if isinstance(x, (list, tuple)):
            x = [np.expand_dims(y, axis=0) for y in x]
        elif isinstance(x, dict):
            x = {key: np.expand_dims(x[key], axis=0) for key in x}
        else:
            x = np.expand_dims(x, axis=0)
        return self.model.predict(x)[0]

    def predict_all(self, x, batch_size=None):
        """Predicts on many samples.
        params:
            x: A ndarray of model inputs
        return: A result from the model output
        """
        return self.model.predict(x, batch_size=batch_size)


def dense(units, activation='relu', l1=0, l2=0, batch_norm=True,
          momentum=0.99, epsilon=1e-5, name=None):
    """Creates a dense layer function.
    params:
        units: An integer, which is the dimensionality of the output space
        activation: A string or keras/TF activation function
        l1: A float, which is the amount of L1 regularization
        l2: A float, which is the amount of L2 regularization
        batch_norm: A boolean, which determines if batch
                    normalization is enabled
        momentum: A float, which is the momentum for the moving
                  mean and variance
        epsilon: A float, which adds variance to avoid dividing by zero
        name: A string, which is the name of the dense layer
    return: A function, which takes a layer as input and returns a dense(layer)
    """
    if activation == 'relu':
        kernel_initializer = 'he_normal'
    else:
        kernel_initializer = 'glorot_uniform'
    if l1 == l2 == 0:
        dl = keras.layers.Dense(units, activation=activation, name=name,
                                kernel_initializer=kernel_initializer)
    else:
        dl = keras.layers.Dense(units, activation=activation,
                                kernel_regularizer=l1_l2(l1, l2), name=name,
                                kernel_initializer=kernel_initializer)
    if batch_norm:
        bn_name = name + '_batchnorm' if name is not None else None
        bnl = keras.layers.BatchNormalization(epsilon=epsilon,
                                              momentum=momentum,
                                              name=bn_name)

    def layer(x):
        """Applies dense layer to input layer.
        params:
            x: A Tensor
        return: A Tensor
        """
        x = dl(x)
        if batch_norm:
            x = bnl(x)
        return x
    return layer


def conv1d(filters, kernel_size, strides=1, activation='relu',
           padding='same', max_pool_size=None, max_pool_strides=None,
           upsampling_size=None, l1=0, l2=0, batch_norm=True,
           momentum=0.99, epsilon=1e-5, transpose=False, name=None):
    """Creates a 1D convolution layer function.
    params:
        filters: An integer, which is the dimensionality of the output space
        kernel_size: An integer or tuple of 1 integers, which is the size of
                     the convoluition kernel
        strides: An integer or tuple of 1 integers, which is stride length
                 of the windows
        activation: A string or keras/TF activation function
        padding: A string ('same', 'valid')
        max_pool_size: An integer, which is the size of the pooling windows
        max_pool_strides: An integer, which is the factor to downscale by
        upsampling_size: An integer, which is the factor to upsample by
        l1: A float, which is the amount of L1 regularization
        l2: A float, which is the amount of L2 regularization
        batch_norm: A boolean, which determines if batch
                    normalization is enabled
        momentum: A float, which is the momentum for the moving
                  mean and variance
        epsilon: A float, which adds variance to avoid dividing by zero
        transpose: A boolean, which determines if the convolution layer
                   should be a deconvolution layer
        name: A string, which is the name of the dense layer
    return: A function, which takes a layer as input and returns
            a conv1d(layer)
    """
    if activation == 'relu':
        kernel_initializer = 'he_normal'
    else:
        kernel_initializer = 'glorot_uniform'
    if transpose:
        kl_conv1d = keras.layers.Conv1DTranspose
    else:
        kl_conv1d = keras.layers.Conv1D
    if l1 == l2 == 0:
        cl = kl_conv1d(filters, kernel_size,
                       activation=activation,
                       strides=strides, padding=padding,
                       name=name,
                       kernel_initializer=kernel_initializer)
    else:
        cl = kl_conv1d(filters, kernel_size,
                       activation=activation,
                       strides=strides, padding=padding,
                       kernel_regularizer=l1_l2(l1, l2),
                       name=name,
                       kernel_initializer=kernel_initializer)
    if batch_norm:
        bn_name = name + '_batchnorm' if name is not None else None
        bnl = keras.layers.BatchNormalization(epsilon=epsilon,
                                              momentum=momentum,
                                              name=bn_name)
    if (max_pool_size is not None or max_pool_strides is not None):
        mp_name = name + '_maxpool' if name is not None else None
        mpl = keras.layers.MaxPooling1D(pool_size=max_pool_size,
                                        strides=max_pool_strides,
                                        padding=padding,
                                        name=mp_name)
    if upsampling_size is not None:
        us_name = name + '_upsample' if name is not None else None
        usl = keras.layers.UpSampling1D(upsampling_size, name=us_name)

    def layer(x):
        """Applies 1D convolution layer to layer x.
        params:
            x: A Tensor
        return: A Tensor
        """
        x = cl(x)
        if batch_norm:
            x = bnl(x)
        if (max_pool_size is not None or max_pool_strides is not None):
            x = mpl(x)
        if upsampling_size is not None:
            x = usl(x)
        return x
    return layer


def conv2d(filters, kernel_size=3, strides=1, activation='relu',
           padding='same', max_pool_size=None, max_pool_strides=None,
           l1=0, l2=0, batch_norm=True, momentum=0.99, epsilon=1e-5,
           upsampling_size=None, transpose=False, name=None):
    """Creates a 2D convolution layer function.
    params:
        filters: An integer, which is the dimensionality of the output space
        kernel_size: An integer or tuple of 2 integers, which is the size of
                     the convoluition kernel
        strides: An integer or tuple of 2 integers, which is stride length
                 of the windows
        activation: A string or keras/TF activation function
        padding: A string ('same', 'valid')
        max_pool_size: An integer or tuple of 2 integers, which is the size
                       of the pooling windows
        max_pool_strides: An integer or tuple of 2 integers, which is the
                          factor to downscale by
        l1: A float, which is the amount of L1 regularization
        l2: A float, which is the amount of L2 regularization
        batch_norm: A boolean, which determines if batch
                    normalization is enabled
        momentum: A float, which is the momentum for the moving
                  mean and variance
        epsilon: A float, which adds variance to avoid dividing by zero
        upsampling_size: An integer, which is the factor to upsample by
        transpose: A boolean, which determines if the convolution layer
                   should be a deconvolution layer
        name: A string, which is the name of the dense layer
    return: A function, which takes a layer as input and returns
            a conv2d(layer)
    """
    if activation == 'relu':
        kernel_initializer = 'he_normal'
    else:
        kernel_initializer = 'glorot_uniform'
    if transpose:
        kl_conv2d = keras.layers.Conv2DTranspose
    else:
        kl_conv2d = keras.layers.Conv2D
    if l1 == l2 == 0:
        cl = kl_conv2d(filters, kernel_size,
                       activation=activation,
                       strides=strides, padding=padding,
                       name=name,
                       kernel_initializer=kernel_initializer)
    else:
        cl = kl_conv2d(filters, kernel_size,
                       activation=activation,
                       strides=strides, padding=padding,
                       kernel_regularizer=l1_l2(l1, l2),
                       name=name,
                       kernel_initializer=kernel_initializer)
    if batch_norm:
        bn_name = name + '_batchnorm' if name is not None else None
        bnl = keras.layers.BatchNormalization(epsilon=epsilon,
                                              momentum=momentum,
                                              name=bn_name)
    if (max_pool_size is not None or max_pool_strides is not None):
        mp_name = name + '_maxpool' if name is not None else None
        mpl = keras.layers.MaxPooling2D(pool_size=max_pool_size,
                                        strides=max_pool_strides,
                                        padding=padding,
                                        name=mp_name)
    if upsampling_size is not None:
        us_name = name + '_upsample' if name is not None else None
        usl = keras.layers.UpSampling2D(upsampling_size, name=us_name)

    def layer(x):
        """Applies 2D convolution layer to layer x.
        params:
            x: A Tensor
        return: A Tensor
        """
        x = cl(x)
        if batch_norm:
            x = bnl(x)
        if (max_pool_size is not None or max_pool_strides is not None):
            x = mpl(x)
        if upsampling_size is not None:
            x = usl(x)
        return x
    return layer


def inception(inceptions):
    """Creates an inception network.
    params:
        inceptions: A list of functions that apply layers or Tensors
    return: A function, which takes a layer and returns inception(layer)
    """
    def layer(x):
        """Builds and applies an inception architecture.
        params:
            x: A Tensor
        return: A Tensor
        """
        branches = []
        for branch in inceptions:
            y = branch[0](x)
            for layer in branch[1:]:
                y = layer(y)
            branches.append(y)
        return branches
    return layer
