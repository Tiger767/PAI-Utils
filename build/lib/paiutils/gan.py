"""
Author: Travis Hammond
Version: 12_8_2020
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


class GANTrainer(Trainer):
    """Generative Adversarial Network Trainer is used for loading, saving,
       and training keras GAN models.
    """

    def __init__(self, model, dis_model, train_data,
                 conditional=False, normal_distribution=False,
                 idt_loss_func=None):
        """Initializes data, optimizers, metrics, and models.
        params:
            model: A compiled keras model, which is the generator
                   (loss function does not matter)
            dis_model: A compiled keras model, which is the discriminator
                       (loss function does not matter)
            train_data: A dictionary, numpy ndarray containg train data,
                        or a list with x and y ndarrays.
            conditional: A boolean, which determines if the GAN is a
                         conditional GAN and neededs x and y data
            normal_distribution: A boolean, which determines if the
                                 model should be trained with normal
                                 or uniform random values
            idt_loss_func: A function for calculating the loss
                           between real x and predict x
                           (default: Mean Abs Error)
        """
        if not isinstance(train_data, (dict, np.ndarray, list)):
            raise TypeError(
                'train_data must be either a dictionary, ndarray, or list'
            )
        if conditional:
            names = model.input_names
            if len(names) == 2:
                if 'x' not in names or 'noise' not in names:
                    raise ValueError('model input names are invalid')
            else:
                raise ValueError('dis_model should have two inputs')
            noise_ndx = names.index('noise')
            self.input_shape = model.input_shape[noise_ndx][1:]
            names = dis_model.input_names
            if len(names) == 2:
                if 'x' not in names or 'y' not in names:
                    raise ValueError('dis_model input names are invalid')
            else:
                raise ValueError('model should have two inputs')
        else:
            self.input_shape = model.input_shape[1:]
            if len(model.input_names) != 1:
                raise ValueError('model should only have one input')
            if len(dis_model.input_names) != 1:
                raise ValueError('dis_model should only have one input')
        self.model = model
        self.optimizer = model.optimizer
        if idt_loss_func is None:
            idt_loss_func = keras.losses.MeanAbsoluteError()
        self.idt_loss_func = idt_loss_func
        self.dis_model = dis_model
        self.dis_optimizer = dis_model.optimizer
        self.gen_adv_metric = keras.metrics.Mean(name='gen_adv_loss')
        self.gen_idt_metric = keras.metrics.Mean(name='gen_idt_loss')
        self.dis_metric = keras.metrics.Mean(name='dis_loss')
        self.train_data = train_data
        self.conditional = conditional
        self.normal_distribution = normal_distribution
        self.idt_loss_coef = 1

        if isinstance(train_data, dict):
            if ('train_x' in train_data and 'train_y' in train_data):
                if not self.conditional:
                    raise ValueError('Not conditional but x '
                                     'and y data provided')
                self.train_data = [train_data['train_x'],
                                   train_data['train_y']]
            elif 'train_x' in train_data:
                if self.conditional:
                    raise ValueError('Conditional but x and '
                                     'y data not provided')
                self.train_data = train_data['train_x']
            elif 'train_y' in train_data:
                if self.conditional:
                    raise ValueError('Conditional but x and '
                                     'y data not provided')
                self.train_data = train_data['train_y']
            else:
                raise ValueError('Invalid train_data')
        elif isinstance(train_data, list):
            if len(train_data) == 2:
                if not self.conditional:
                    raise ValueError('Not conditional but x '
                                     'and y data provided')
            elif len(train_data) == 1:
                if self.conditional:
                    raise ValueError('Conditional but x and '
                                     'y data not provided')
                self.train_data = train_data[0]
            else:
                raise ValueError('Invalid train_data')
        if self.conditional:
            self.train_data[0] = self.train_data[0].astype(
                keras.backend.floatx()
            )
            self.train_data[1] = self.train_data[1].astype(
                keras.backend.floatx()
            )
        else:
            self.train_data = self.train_data.astype(
                keras.backend.floatx()
            )

    @tf.function
    def _train_step(self, batch):
        """Trains the GAN 1 epoch.
        params:
            batch: A Tensor
        """
        if self.conditional:
            x, y = batch
        else:
            y = batch
        length = y.shape[0]
        if self.normal_distribution:
            noise = tf.random.normal([length, *self.input_shape])
        else:
            noise = tf.random.uniform([length, *self.input_shape])
        with tf.GradientTape() as tape, tf.GradientTape() as dis_tape:
            if self.conditional:
                fake_y = self.model({'x': x, 'noise': noise}, training=True)

                dis_fake_y = self.dis_model({'x': x, 'y': fake_y},
                                            training=True)
                dis_real_y = self.dis_model({'x': x, 'y': y}, training=True)
            else:
                fake_y = self.model(noise, training=True)

                dis_fake_y = self.dis_model(fake_y, training=True)
                dis_real_y = self.dis_model(y, training=True)

            adv_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.ones_like(dis_fake_y), dis_fake_y
            )
            idt_loss = (self.idt_loss_coef *
                        self.idt_loss_func(y, fake_y))
            gen_loss = adv_loss + idt_loss

            dis_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.zeros_like(dis_fake_y), dis_fake_y
            )
            dis_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.ones_like(dis_real_y), dis_real_y
            )
            dis_loss = dis_fake_loss + dis_real_loss

        grads = tape.gradient(gen_loss, self.model.trainable_variables)
        dis_grads = dis_tape.gradient(dis_loss,
                                      self.dis_model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        self.dis_optimizer.apply_gradients(
            zip(dis_grads, self.dis_model.trainable_variables)
        )

        self.gen_adv_metric(adv_loss)
        self.gen_idt_metric(idt_loss)
        self.dis_metric(dis_loss)

    def train(self, epochs, batch_size=None, idt_loss_coef=0, verbose=True):
        """Trains the keras model.
        params:
            epochs: An integer, which is the number of complete
                    iterations to train
            batch_size: An integer, which is the number of samples
                        per graident update
            idt_loss_coef: A float, which is the amount of the identity
                           loss to be added to the gen model loss
            verbose: A boolean, which determines the verbositiy level
        """
        self.idt_loss_coef = idt_loss_coef
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
                if verbose:
                    print(f'{count}/{length}', end='\r')
            if verbose:
                print(f'{count}/{length} - '
                      f'gen_adv_loss: {self.gen_adv_metric.result()} - '
                      f'gen_idt_loss: {self.gen_idt_metric.result()} - '
                      f'dis_loss: {self.dis_metric.result()}')
            self.gen_adv_metric.reset_states()
            self.gen_idt_metric.reset_states()
            self.dis_metric.reset_states()

    def load(self, path, custom_objects=None):
        """Loads a generator and discriminator model and weights from a file.
           (overrides the inital provided model)
        params:
            path: A string, which is the path to a folder
                  containing model.json, weights.h5, and note.txt
            custom_objects: A dictionary mapping to custom classes
                            or functions for loading the model
        """
        with open(os.path.join(path, 'model.json'), 'r') as file:
            self.model = model_from_json(
                file.read(), custom_objects=custom_objects
            )
        with open(os.path.join(path, 'dis_model.json'), 'r') as file:
            self.dis_model = model_from_json(
                file.read(), custom_objects=custom_objects
            )
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
            note: A string, which is a note to save in the folder
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
                file.write('model1\n')
                self.model.summary(print_fn=lambda line: file.write(line+'\n'))
                file.write('\ndis_model1\n')
                self.dis_model.summary(
                    print_fn=lambda line: file.write(line+'\n')
                )
            else:
                file.write(note)
        return path


class GANPredictor(Predictor):
    """Generative Adversarial Network Predictor is used for
       loading and predicting keras GAN models.
    """

    def __init__(self, path,  weights_name='weights.h5',
                 model_name='model.json', custom_objects=None):
        Predictor.__init__(self, path, weights_name=weights_name,
                           model_name=model_name,
                           custom_objects=custom_objects)
        if isinstance(self.model.input_shape, list):
            noise_ndx = self.model.input_names.index('noise')
            self.input_shape = self.model.input_shape[noise_ndx][1:]
        else:
            self.input_shape = self.model.input_shape[1:]

    def predict(self, noise, x=None):
        """Predicts on a single sample.
        params:
            noise: A single model noise input
            x: A single model conditional input
        return: A result from the model output
        """
        if x is None:
            return self.model.predict(np.expand_dims(noise, axis=0))[0]
        return self.model.predict({'x': np.expand_dims(x, axis=0),
                                   'noise': np.expand_dims(noise, axis=0)})[0]

    def predict_all(self, noise, x=None, batch_size=None):
        """Predicts on many samples.
        params:
            noise: A ndarray of model noise input
            x: A ndarray of model conditional input
        return: A result from the model output
        """
        if x is None:
            return self.model.predict(noise, batch_size=batch_size)
        return self.model.predict({'x': x, 'noise': noise},
                                  batch_size=batch_size)

    def random_normal_predict(self, x=None):
        """Predicts an output with a random normal distribution.
        params:
            x: A single model conditional input
        return: A result from the model output
        """
        noise = tf.random.normal([1, *self.input_shape])
        if x is None:
            return self.model.predict(noise)[0]
        return self.model.predict({'x': np.expand_dims(x, axis=0),
                                   'noise': noise})[0]

    def random_uniform_predict(self, x=None):
        """Predicts an output with a random uniform distribution.
        params:
            x: A single model conditional input
        return: A result from the model output
        """
        noise = tf.random.uniform([1, *self.input_shape])
        if x is None:
            return self.model.predict(noise)[0]
        return self.model.predict({'x': np.expand_dims(x, axis=0),
                                   'noise': noise})[0]


class GANITrainer(GANTrainer):
    """Generative Adversarial Network with provided Inputs
       Trainer is used for loading, saving, and training
       keras GAN models that do not have random inputs.
    """

    def __init__(self, model, dis_model, train_data, idt_loss_func=None):
        """Initializes data, optimizers, metrics, and models.
        params:
            model: A compiled keras model, which is the generator
                   (loss function does not matter)
            dis_model: A compiled keras model, which is the discriminator
                       (loss function does not matter)
            train_data: A dictionary containg train data or a
                        list with x and y ndarrays
                        Ex. {'train_x': [...], 'train_y: [...]}
            idt_loss_func: A function for calculating the loss
                           between real x and predict x
                           (default: Mean Abs Error)
        """
        if not isinstance(train_data, (dict, list)):
            raise TypeError(
                'train_data must be either a dictionary or list'
            )
        names = dis_model.input_names
        if len(names) == 2:
            if 'x' not in names or 'y' not in names:
                raise ValueError('dis_model input names are invalid')
        else:
            raise ValueError('dis_model should have two inputs')
        if len(model.input_names) != 1:
            raise ValueError('model should only have one input')
        self.model = model
        self.optimizer = model.optimizer
        if idt_loss_func is None:
            idt_loss_func = keras.losses.MeanAbsoluteError()
        self.idt_loss_func = idt_loss_func
        self.dis_model = dis_model
        self.dis_optimizer = dis_model.optimizer
        self.gen_adv_metric = keras.metrics.Mean(name='gen_adv_loss')
        self.gen_idt_metric = keras.metrics.Mean(name='gen_idt_loss')
        self.dis_metric = keras.metrics.Mean(name='dis_loss')
        self.conditional = True
        self.train_data = train_data
        self.idt_loss_coef = 1

        if isinstance(train_data, dict):
            if 'train_x' in train_data and 'train_y' in train_data:
                self.train_data = [train_data['train_x'],
                                   train_data['train_y']]
            else:
                raise ValueError('train_data missing x or y data')
        elif isinstance(train_data, list) and len(train_data) != 2:
            raise ValueError('train_data should only have a x and y element')
        self.train_data[0] = self.train_data[0].astype(
            keras.backend.floatx()
        )
        self.train_data[1] = self.train_data[1].astype(
            keras.backend.floatx()
        )

    @tf.function
    def _train_step(self, batch):
        """Trains the GAN 1 epoch.
        params:
            batch: A tensor
        """
        x, y = batch
        with tf.GradientTape() as tape, tf.GradientTape() as dis_tape:
            fake_y = self.model(x, training=True)

            dis_fake_y = self.dis_model({'x': x, 'y': fake_y}, training=True)
            dis_real_y = self.dis_model({'x': x, 'y': y}, training=True)

            adv_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.ones_like(dis_fake_y), dis_fake_y
            )
            idt_loss = (self.idt_loss_coef *
                        self.idt_loss_func(y, fake_y))
            gen_loss = adv_loss + idt_loss

            dis_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.zeros_like(dis_fake_y), dis_fake_y
            )
            dis_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.ones_like(dis_real_y), dis_real_y
            )
            dis_loss = dis_fake_loss + dis_real_loss

        grads = tape.gradient(gen_loss, self.model.trainable_variables)
        dis_grads = dis_tape.gradient(dis_loss,
                                      self.dis_model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        self.dis_optimizer.apply_gradients(
            zip(dis_grads, self.dis_model.trainable_variables)
        )

        self.gen_adv_metric(adv_loss)
        self.gen_idt_metric(idt_loss)
        self.dis_metric(dis_loss)


class CycleGANTrainer(Trainer):
    """Cyle Generative Adversarial Network Trainer is used for loading, saving,
       and training keras GAN models.
    """

    def __init__(self, model, dis_model, train_data, idt_loss_func=None):
        """Initializes data, optimizers, metrics, and models.
        params:
            model: A compiled keras model, which is the generator
                   (loss function does not matter)
            dis_model: A compiled keras model, which is the discriminator
                       (loss function does not matter)
            train_data: A dictionary, numpy ndarray containg train data,
                        or a list with x and y ndarrays.
            idt_loss_func: A function for calculating the loss
                           between real x and predict x
                           (default: Mean Abs Error)
        """
        if not isinstance(train_data, (dict, list)):
            raise TypeError(
                'train_data must be either a dictionary or list'
            )
        if len(dis_model.input_names) != 1:
            raise ValueError('dis_model should only have one input')
        if len(model.input_names) != 1:
            raise ValueError('model should only have one input')
        self.model = model
        self.model2 = keras.models.clone_model(model)
        self.optimizer = model.optimizer
        self.optimizer2 = keras.optimizers.get(model.optimizer) # share params?
        if idt_loss_func is None:
            idt_loss_func = keras.losses.MeanAbsoluteError()
        self.idt_loss_func = idt_loss_func
        self.dis_model = dis_model
        self.dis_model2 = keras.models.clone_model(dis_model)
        self.dis_optimizer = dis_model.optimizer
        self.dis_optimizer2 = keras.optimizers.get(dis_model.optimizer)
        self.gen_adv_metric1 = keras.metrics.Mean(name='gen_adv_loss')
        self.gen_idt_metric1 = keras.metrics.Mean(name='gen_idt_loss')
        self.dis_metric1 = keras.metrics.Mean(name='dis_loss')
        self.gen_adv_metric2 = keras.metrics.Mean(name='gen_adv_loss')
        self.gen_idt_metric2 = keras.metrics.Mean(name='gen_idt_loss')
        self.dis_metric2 = keras.metrics.Mean(name='dis_loss')
        self.cycle_metric = keras.metrics.Mean(name='cycle_loss')
        self.train_data = train_data
        self.cycle_loss_coef = 1
        self.idt_loss_coef = 1

        if isinstance(train_data, dict):
            if 'train_x' in train_data and 'train_y' in train_data:
                self.train_data = [train_data['train_x'],
                                    train_data['train_y']]
            else:
                raise ValueError('There must be x and y train data')
        elif len(train_data) != 2:
            raise ValueError('Invalid train_data')
        self.train_data[0] = self.train_data[0].astype(
            keras.backend.floatx()
        )
        self.train_data[1] = self.train_data[1].astype(
            keras.backend.floatx()
        )

    @tf.function
    def _train_step(self, x, y):
        """Trains the GAN 1 epoch.
        params:
            x: A Tensor
            y: A Tensor
        """
        with tf.GradientTape(persistent=True) as tape:
            fake_y = self.model(x, training=True)
            cycled_x = self.model2(fake_y, training=True)
            fake_x = self.model2(y, training=True)
            cycled_y = self.model(fake_x, training=True)

            same_x = self.model2(x, training=True)
            same_y = self.model(y, training=True)

            dis_real_x = self.dis_model2(x, training=True)
            dis_real_y = self.dis_model(y, training=True)
            dis_fake_x = self.dis_model2(fake_x, training=True)
            dis_fake_y = self.dis_model(fake_y, training=True)

            adv_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.ones_like(dis_fake_y), dis_fake_y
            )
            adv_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.ones_like(dis_fake_x), dis_fake_x
            )
            idt_loss1 = (self.idt_loss_func(y, same_y) *
                         self.idt_loss_coef)
            idt_loss2 = (self.idt_loss_func(x, same_x) *
                         self.idt_loss_coef)
            cycle_loss = ((self.idt_loss_func(x, cycled_x) +
                           self.idt_loss_func(y, cycled_y)) *
                          self.cycle_loss_coef)
            gen_loss1 = cycle_loss + idt_loss1
            gen_loss2 = cycle_loss + idt_loss2

            dis_fake_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.zeros_like(dis_fake_x), dis_fake_x
            )
            dis_real_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.ones_like(dis_real_x), dis_real_x
            )
            dis_loss2 = dis_fake_loss2 + dis_real_loss2

            dis_fake_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.zeros_like(dis_fake_y), dis_fake_y
            )
            dis_real_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(
                tf.ones_like(dis_real_y), dis_real_y
            )
            dis_loss1 = dis_fake_loss1 + dis_real_loss1

        grads = tape.gradient(gen_loss1, self.model.trainable_variables)
        grads2 = tape.gradient(gen_loss2, self.model2.trainable_variables)

        dis_grads = tape.gradient(dis_loss1,
                                  self.dis_model.trainable_variables)
        dis_grads2 = tape.gradient(dis_loss2,
                                   self.dis_model2.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        self.optimizer2.apply_gradients(
            zip(grads2, self.model2.trainable_variables)
        )

        self.dis_optimizer.apply_gradients(
            zip(dis_grads, self.dis_model.trainable_variables)
        )
        self.dis_optimizer2.apply_gradients(
            zip(dis_grads2, self.dis_model2.trainable_variables)
        )

        self.gen_adv_metric1(adv_loss1)
        self.gen_idt_metric1(idt_loss1)
        self.dis_metric1(dis_loss1)
        self.gen_adv_metric2(adv_loss2)
        self.gen_idt_metric2(idt_loss2)
        self.dis_metric2(dis_loss2)
        self.cycle_metric(cycle_loss)

    def train(self, epochs, batch_size=None, idt_loss_coef=10,
              cycle_loss_coef=10, verbose=True):
        """Trains the keras model.
        params:
            epochs: An integer, which is the number of complete
                    iterations to train
            batch_size: An integer, which is the number of samples
                        per graident update
            idt_loss_coef: A float, which is the amount of the identity
                          loss to be added to the gen model loss
            cycle_loss_coef: A float, which is the amount of the cycle
                             loss to be added to the gen model loss
            verbose: A boolean, which determines the verbositiy level
        """
        self.idt_loss_coef = idt_loss_coef
        self.cycle_loss_coef = cycle_loss_coef
        length = self.train_data[0].shape[0]
        batches = tf.data.Dataset.from_tensor_slices(
            (self.train_data[0],
             self.train_data[1])
        ).shuffle(length).batch(batch_size)
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f'Epoch {epoch}/{epochs}')
            count = 0
            for batch in batches:
                self._train_step(*batch)
                count += np.minimum(batch_size, length - count)
                if verbose:
                    print(f'{count}/{length}', end='\r')
            if verbose:
                print(f'{count}/{length} - '
                      f'gen_adv_loss1: {self.gen_adv_metric1.result()} - '
                      f'gen_idt_loss1: {self.gen_idt_metric1.result()} - '
                      f'dis_loss1: {self.dis_metric1.result()}'
                      f'gen_adv_loss2: {self.gen_adv_metric2.result()} - '
                      f'gen_idt_loss2: {self.gen_idt_metric2.result()} - '
                      f'dis_loss2: {self.dis_metric2.result()} - '
                      f'cycle_loss: {self.cycle_metric.result()}')
            self.gen_adv_metric1.reset_states()
            self.gen_idt_metric1.reset_states()
            self.dis_metric1.reset_states()
            self.gen_adv_metric2.reset_states()
            self.gen_idt_metric2.reset_states()
            self.dis_metric2.reset_states()
            self.cycle_metric.reset_states()

    def load(self, path, custom_objects=None):
        """Loads a generator and discriminator model and weights from a file.
           (overrides the inital provided model)
        params:
            path: A string, which is the path to a folder
                  containing model.json, weights.h5, and note.txt
            custom_objects: A dictionary mapping to custom classes
                            or functions for loading the model
        """
        with open(os.path.join(path, 'model.json'), 'r') as file:
            self.model = model_from_json(
                file.read(), custom_objects=custom_objects
            )
        with open(os.path.join(path, 'model2.json'), 'r') as file:
            self.model2 = model_from_json(
                file.read(), custom_objects=custom_objects
            )
        with open(os.path.join(path, 'dis_model.json'), 'r') as file:
            self.dis_model = model_from_json(
                file.read(), custom_objects=custom_objects
            )
        with open(os.path.join(path, 'dis_model2.json'), 'r') as file:
            self.dis_model2 = model_from_json(
                file.read(), custom_objects=custom_objects
            )
        self.model.load_weights(os.path.join(path, 'weights.h5'))
        self.model2.load_weights(os.path.join(path, 'weights2.h5'))
        self.dis_model.load_weights(os.path.join(path, 'dis_weights.h5'))
        self.dis_model2.load_weights(os.path.join(path, 'dis_weights2.h5'))
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
        self.model.save_weights(os.path.join(path, 'weights.h5'))
        self.model2.save_weights(os.path.join(path, 'weights2.h5'))
        self.dis_model.save_weights(os.path.join(path, 'dis_weights.h5'))
        self.dis_model2.save_weights(os.path.join(path, 'dis_weights2.h5'))
        with open(os.path.join(path, 'model.json'), 'w') as file:
            file.write(self.model.to_json())
        with open(os.path.join(path, 'model2.json'), 'w') as file:
            file.write(self.model2.to_json())
        with open(os.path.join(path, 'dis_model.json'), 'w') as file:
            file.write(self.dis_model.to_json())
        with open(os.path.join(path, 'dis_model2.json'), 'w') as file:
            file.write(self.dis_model2.to_json())
        with open(os.path.join(path, 'note.txt'), 'w') as file:
            if note is None:
                file.write('model1\n')
                self.model.summary(print_fn=lambda line: file.write(line+'\n'))
                file.write('\ndis_model1\n')
                self.dis_model.summary(
                    print_fn=lambda line: file.write(line+'\n')
                )
                file.write('\n\nmodel2\n')
                self.model2.summary(print_fn=lambda line: file.write(line+'\n'))
                file.write('\ndis_model2\n')
                self.dis_model2.summary(
                    print_fn=lambda line: file.write(line+'\n')
                )
            else:
                file.write(note)
        return path
