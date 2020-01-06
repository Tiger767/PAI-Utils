"""
Author: Travis Hammond
Version: 1_3_2020
"""


import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json

try:
    from paiutils.neural_network import (
        Trainner, Predictor, dense, conv2d
    )
    from paiutils.util_funcs import load_directory_dataset, load_h5py
except ImportError:
    from neural_network import (
        Trainner, Predictor, dense, conv2d
    )
    from util_funcs import load_directory_dataset, load_h5py


class GANTrainner(Trainner):
    """Generative Adversarial Network Trainner is used for loading, saving,
       and training keras GAN models.
    """

    def __init__(self, model, dis_model, train_data, file_loader=None,
                 conditional=False, normal_distribution=False):
        """Initializes train, validation, and test data.
        params:
            model: A compiled keras model, which is the generator
                   (loss function does not matter)
            dis_model: A compiled keras model, which is the discriminator
                       (loss function does not matter)
            train_data: A dictionary, numpy ndarray, string/path
                        containg train data, or a list with x
                        and y ndarrays (Ex. {'train_x': [...]})
            file_loader: A function for loading each file
            conditional: A boolean, which determines if the GAN is a
                         conditional GAN and neededs y data
            normal_distribution: A boolean, which determines if the
                                 model should be trained with normal
                                 or uniform random values
        """
        assert isinstance(train_data, (str, dict, np.ndarray, list)), (
            'train_data must be a dictionary, a file/folder path, a ndarray, '
            'or a list with two ndarrays'
        )
        self.model = model
        self.input_shape = self.model.layers[0].input_shape[0][1:]
        self.optimizer = model.optimizer
        self.dis_model = dis_model
        self.dis_optimizer = dis_model.optimizer
        self.metric = tf.keras.metrics.Mean(name='loss')
        self.dis_metric = tf.keras.metrics.Mean(name='dis_loss')
        self.train_data = train_data
        self.conditional = conditional
        self.normal_distribution = normal_distribution

        if (not isinstance(train_data, np.ndarray) and
                (self.conditional and not
                 isinstance(train_data[0], np.ndarray))):
            if isinstance(train_data, str):
                if os.path.isdir(train_data):
                    assert file_loader is not None
                    if self.conditional:
                        data = load_directory_dataset(
                            train_data, file_loader
                        )
                        train_data = [data['train_x'], data['train_y']]
                    else:
                        train_data = load_directory_dataset(
                            train_data, file_loader
                        )['train_x']
                else:
                    assert train_data.split('.')[1] == 'h5'
                    train_data = load_h5py(train_data)
            if isinstance(train_data, dict):
                if 'train_x' in train_data:
                    if self.conditional:
                        self.train_data = [train_data['train_x'],
                                           train_data['train_y']]
                    else:
                        self.train_data = train_data['train_x']
                else:
                    raise Exception('There must be a train dataset')
            else:
                raise ValueError('Invalid train_data')
        if self.conditional:
            self.train_data[0] = self.train_data[0].astype(
                tf.keras.backend.floatx()
            )
            self.train_data[1] = self.train_data[1].astype(
                tf.keras.backend.floatx()
            )
        else:
            self.train_data = self.train_data.astype(
                tf.keras.backend.floatx()
            )

    @tf.function
    def _train_step(self, x):
        """Trains the GAN 1 epoch.
        params:
            x: A Tensor
            y: A Tensor
        """
        if self.conditional:
            length = x[0].shape[0]
        else:
            length = x.shape[0]
        if self.normal_distribution:
            inputs = tf.random.normal([length,
                                       *self.input_shape])
        else:
            inputs = tf.random.uniform([length,
                                        *self.input_shape])
        if self.conditional:
            inputs = [inputs, x[1]]
        with tf.GradientTape() as tape, tf.GradientTape() as dis_tape:
            preds = self.model(inputs, training=True)
            if len(self.model.losses) > 0:
                reg_loss = tf.math.add_n(self.model.losses)
            else:
                reg_loss = 0
            if self.conditional:
                preds = [preds, x[1]]
            dis_preds = self.dis_model(preds, training=True)
            dis_real_preds = self.dis_model(x, training=True)
            if len(self.dis_model.losses) > 0:
                dis_reg_loss = tf.math.add_n(self.dis_model.losses)
            else:
                dis_reg_loss = 0
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.ones_like(dis_preds), dis_preds
            ) + reg_loss
            dis_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.zeros_like(dis_preds), dis_preds
            )
            dis_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.ones_like(dis_real_preds), dis_real_preds
            )
            total_dis_loss = dis_loss + dis_real_loss + dis_reg_loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        dis_grads = dis_tape.gradient(total_dis_loss,
                                      self.dis_model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        self.dis_optimizer.apply_gradients(
            zip(dis_grads, self.dis_model.trainable_variables)
        )

        self.metric(loss)
        self.dis_metric(total_dis_loss)

    def train(self, epochs, batch_size=None, verbose=True):
        """Trains the keras model.
        params:
            epochs: An integer, which is the number of complete
                    iterations to train
            batch_size: An integer, which is the number of samples
                        per graident update
            verbose: A boolean, which determines the verbositiy level
        """

        if self.conditional:
            length = self.train_data[0].shape[0]
            batches = tf.data.Dataset.from_tensor_slices(
                (self.train_data[0],
                 self.train_data[1])
            ).shuffle(length).batch(batch_size)
        else:
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
                print(f'{count}/{length}', end='\r')
            if verbose:
                print(f'{count}/{length} - '
                      f'loss: {self.metric.result()} - '
                      f'dis_loss: {self.dis_metric.result()}')
            self.metric.reset_states()
            self.dis_metric.reset_states()

    def load(self, path, optimizer='sgd', dis_optimizer='sgd'):
        """Loads a generator and discriminator model and weights from a file.
           (overrides the inital provided model)
        params:
            path: A string, which is the path to a folder
                  containing model.json, weights.h5, and note.txt
            optimizer: A string or optimizer instance, which will be
                       the optimizer for the loaded generator model
            dis_optimizer: A string or optimizer instance, which will be
                           the optimizer for the loaded discriminator model
        """
        with open(os.path.join(path, 'model.json'), 'r') as file:
            self.model = model_from_json(file.read())
            self.model.optimizer = optimizer
        with open(os.path.join(path, 'dis_model.json'), 'r') as file:
            self.dis_model = model_from_json(file.read())
            self.dis_model.optimizer = dis_optimizer
        self.model.load_weights(os.path.join(path, 'weights.h5'))
        self.dis_model.load_weights(os.path.join(path, 'dis_weights.h5'))
        with open(os.path.join(path, 'note.txt'), 'r') as file:
            print(file.read(), end='')

    def save(self, path, note=None):
        """Saves the generator and discriminator model and weights to a file.
        params:
            path: A string, which is the path to create a folder in
                  containing model.json, weights.h5, note.txt,
                  dis_model.json, and dis_weights.h5
        return: A string, which is the given path + folder name
        """
        time = datetime.datetime.now()
        path = os.path.join(path, time.strftime(r'%Y%m%d_%H%M%S_%f'))
        os.mkdir(path)
        self.model.save_weights(os.path.join(path, 'weights.h5'))
        self.dis_model.save_weights(os.path.join(path, 'dis_weights.h5'))
        with open(os.path.join(path, 'model.json'), 'w') as file:
            file.write(self.model.to_json())
        with open(os.path.join(path, 'dis_model.json'), 'w') as file:
            file.write(self.dis_model.to_json())
        with open(os.path.join(path, 'note.txt'), 'w') as file:
            if note is None:
                self.model.summary(print_fn=lambda line: file.write(line+'\n'))
            else:
                file.write(note)
        return path


class GANPredictor(Predictor):
    """Generative Adversarial Network Predictor is used for
       loading and predicting keras GAN models.
    """

    def predict(self, x, y=None):
        """Predicts on a single sample.
        params:
            x: A single model input
            y: A single model conditional input
        return: A result from the model output
        """
        if y is None:
            return self.model.predict(np.expand_dims(x, axis=0))[0]
        return self.model.predict([np.expand_dims(x, axis=0),
                                   np.expand_dims(y, axis=0)])[0]

    def predict_all(self, x, y=None, batch_size=None):
        """Predicts on many samples.
        params:
            x: A ndarray of model inputs
            y: A ndarray of model conditional inputs
        return: A result from the model output
        """
        if y is None:
            return self.model.predict(x, batch_size=batch_size)
        return self.model.predict([x, y], batch_size=batch_size)

    def random_normal_predict(self, y=None):
        """Predicts an output with a random normal distribution.
        params:
            y: A single model conditional input
        return: A result from the model output
        """
        input_shape = self.model.layers[0].input_shape[0][1:]
        normal = tf.random.normal([1, *input_shape])
        if y is None:
            return self.model.predict(normal)[0]
        return self.model.predict([normal,
                                   np.expand_dims(y, axis=0)])[0]

    def random_uniform_predict(self, y=None):
        """Predicts an output with a random uniform distribution.
        params:
            y: A single model conditional input
        return: A result from the model output
        """
        input_shape = self.model.layers[0].input_shape[0][1:]
        uniform = tf.random.uniform([1, *input_shape])
        if y is None:
            return self.model.predict(uniform)[0]
        return self.model.predict([uniform,
                                   np.expand_dims(y, axis=0)])[0]


if __name__ == '__main__':
    import image as img
    from time import sleep

    training = False
    conditional = False
    path = 'trained_conditional' if conditional else 'trained'

    (tx, ty), _ = keras.datasets.fashion_mnist.load_data()
    tx = np.expand_dims((tx - 127.5) / 127.5, axis=-1)
    if conditional:
        labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress',
                  'Coat', 'Sandal', 'Shirt', 'Sneaker',
                  'Bag', 'Ankle boot']
        ty = np.identity(len(labels))[ty]
        tx = [tx, ty]

    if training:
        if conditional:
            # Generator Model
            inputs = keras.layers.Input(shape=(100))
            x1 = dense(512)(inputs)
            cond_inputs = keras.layers.Input(shape=(len(labels)))
            x2 = dense(512)(cond_inputs)
            x = keras.layers.Concatenate()([x1, x2])
            x = dense(7*7*32)(x)
            x = keras.layers.Reshape((7, 7, 32))(x)
            x = conv2d(128, 3, strides=1)(x)
            x = conv2d(64, 3, strides=2, transpose=True)(x)
            outputs = conv2d(1, 3, strides=2,
                             activation='tanh', batch_norm=False,
                             transpose=True)(x)
            model = keras.Model(inputs=[inputs, cond_inputs],
                                outputs=outputs)
            model.summary()
            optimizer = tf.keras.optimizers.Adam(.0002, .5)
            model.optimizer = optimizer

            # Discriminator Model
            inputs = keras.layers.Input(shape=(28, 28, 1))
            cond_inputs = keras.layers.Input(shape=(len(labels)))
            x = conv2d(64, 3, strides=2, activation=None,
                       batch_norm=False)(inputs)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
            x = conv2d(128, 3, strides=2, activation=None)(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
            x = conv2d(256, 3, strides=2, activation=None)(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
            x = conv2d(512, 3, strides=2, activation=None)(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
            x = keras.layers.Flatten()(x)
            x2 = dense(1024, activation=None)(cond_inputs)
            x2 = keras.layers.LeakyReLU(alpha=0.2)(x2)
            x = keras.layers.Concatenate()([x, x2])
            x = dense(1024, activation=None)(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
            outputs = dense(1, activation=None, batch_norm=False)(x)
            dis_model = keras.Model(inputs=[inputs, cond_inputs],
                                    outputs=outputs)
            dis_model.summary()
            dis_optimizer = tf.keras.optimizers.Adam(.0002, .5)
            dis_model.optimizer = dis_optimizer
        else:
            # Generator Model
            inputs = keras.layers.Input(shape=(100))
            x = dense(7*7*32)(inputs)
            x = keras.layers.Reshape((7, 7, 32))(x)
            x = conv2d(128, 3, strides=1)(x)
            x = conv2d(64, 3, strides=2, transpose=True)(x)
            outputs = conv2d(1, 3, strides=2,
                             activation='tanh', batch_norm=False,
                             transpose=True)(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.summary()
            optimizer = tf.keras.optimizers.Adam(.0002, .5)
            # model.compile(optimizer=optimizer, loss='mse')
            model.optimizer = optimizer

            # Discriminator Model
            inputs = keras.layers.Input(shape=(28, 28, 1))
            x = conv2d(64, 3, strides=2, activation=None,
                       batch_norm=False)(inputs)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
            x = conv2d(128, 3, strides=2, activation=None)(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
            x = conv2d(256, 3, strides=2, activation=None)(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
            x = conv2d(512, 3, strides=2, activation=None)(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
            x = keras.layers.Flatten()(x)
            outputs = dense(1, activation=None, batch_norm=False)(x)
            dis_model = keras.Model(inputs=inputs, outputs=outputs)
            dis_model.summary()
            dis_optimizer = tf.keras.optimizers.Adam(.0002, .5)
            dis_model.optimizer = dis_optimizer

        gant = GANTrainner(model, dis_model, tx,
                           conditional=conditional)
        if path is not None:
            gant.load(path)
        gant.train(50, 512)
        path = gant.save('')
        gant.load(path, optimizer=optimizer, dis_optimizer=dis_optimizer)

        del gant

    ganp = GANPredictor(path)

    ws = img.Windows()
    w = ws.add('Image')
    ws.start()

    while True:
        if conditional:
            identity = np.identity(len(labels))
            for ndx in range(len(labels)):
                preds = ganp.random_uniform_predict(identity[ndx])
                preds = np.squeeze(preds * 127.5 + 127.5).astype(np.uint8)
                ws.set(w, preds)
                print(labels[ndx])
                sleep(2)
        else:
            preds = ganp.random_uniform_predict() * 127.5 + 127.5
            preds = np.squeeze(preds).astype(np.uint8)
            ws.set(w, preds)
            sleep(1)

    ws.stop()
