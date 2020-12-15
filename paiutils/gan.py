"""
Author: Travis Hammond
Version: 12_15_2020
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras

from paiutils.neural_network import (
    Trainer, Predictor
)


class GANTrainer(Trainer):
    """Generative Adversarial Network Trainer is used for loading, saving,
       and training keras GAN models.
    """

    class GANModel(keras.Model):
        def __init__(self, generator, discriminator, conditional=False,
                     noise_fn=None, idt_loss_coef=0, **kwargs):
            """GAN Keras Model that has a modified train_step.
            params:
                generator: The generative model
                discriminator: The discriminative model
                conditional: A boolean, which determines if the GAN is a
                             conditional GAN
                noise_fn: A TF function that takes a shape and returns
                          noise (Default: normal noise)
                idt_loss_coef: A float, which is the amount of the identity
                               loss (generator model's loss function)
                               to be added to the generator loss
            """
            super().__init__(**kwargs)
            self.generator = generator
            self.discriminator = discriminator
            self.generator.compiled_loss.build(
                tf.zeros(self.generator.output_shape[1:])
            )
            self.idt_loss_fn = self.generator.compiled_loss._losses[0]
            self.conditional = conditional
            if noise_fn is None:
                noise_fn = tf.random.normal
            self.noise_fn = noise_fn
            self.idt_loss_coef = idt_loss_coef
            if conditional:
                noise_ndx = generator.input_names.index('noise')
                self.noise_shape = generator.input_shape[noise_ndx][1:]
            else:
                self.noise_shape = generator.input_shape[1:]
            self.noise_shape = tf.convert_to_tensor(self.noise_shape)

        def train_step(self, batch):
            """Trains the model 1 step.
            params:
                batch: A tensor, tuple, or list
            """
            if self.conditional:
                cond, real_y = batch
            else:
                if isinstance(batch, (tuple, list)):
                    real_y = batch[0]
                else:
                    real_y = batch

            length = [tf.shape(real_y)[0]]
            noise = tf.random.normal(
                shape=tf.concat([length, self.noise_shape], 0)
            )
            with tf.GradientTape(persistent=True) as tape:
                if self.conditional:
                    fake_y = self.generator(
                        {'condition': cond, 'noise': noise}, training=True
                    )
                    dis_fake_y = self.discriminator(
                        {'condition': cond, 'y': fake_y}, training=True
                    )
                    dis_real_y = self.discriminator(
                        {'condition': cond, 'y': real_y}, training=True
                    )
                else:
                    fake_y = self.generator(noise, training=True)
                    dis_fake_y = self.discriminator(fake_y, training=True)
                    dis_real_y = self.discriminator(real_y, training=True)

                adv_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.ones_like(dis_fake_y), dis_fake_y
                )
                adv_loss = tf.reduce_mean(adv_loss)
                idt_loss = self.idt_loss_fn(real_y, fake_y)
                idt_loss = tf.reduce_mean(idt_loss)
                gen_loss = adv_loss + idt_loss * self.idt_loss_coef

                dis_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.zeros_like(dis_fake_y), dis_fake_y
                )
                dis_fake_loss = tf.reduce_mean(dis_fake_loss)
                dis_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.ones_like(dis_real_y), dis_real_y
                )
                dis_real_loss = tf.reduce_mean(dis_real_loss)
                dis_loss = dis_fake_loss + dis_real_loss
            gen_grads = tape.gradient(
                gen_loss, self.generator.trainable_variables
            )
            dis_grads = tape.gradient(
                dis_loss, self.discriminator.trainable_variables
            )
            self.generator.optimizer.apply_gradients(
                zip(gen_grads, self.generator.trainable_variables)
            )
            self.discriminator.optimizer.apply_gradients(
                zip(dis_grads, self.discriminator.trainable_variables)
            )
            return {
                'gen_loss': gen_loss,
                'adversarial_loss': adv_loss,
                'identity_loss': idt_loss,
                'discriminator_loss': dis_loss,
                'dis_fake_input_loss': dis_fake_loss,
                'dis_real_input_loss': dis_real_loss
            }

        def call(self, inputs, training=False):
            """Calls the discriminator model on new inputs.
            params:
                inputs: A tensor or list of tensors
                training: A boolean or boolean scalar tensor, indicating
                          whether to run the `Network` in training mode
                          or inference mode
            return: A tensor if there is a single output, or a list of
                    tensors if there are more than one outputs.
            """
            if self.conditional:
                cond, real_y = inputs
                dis_real_y = self.discriminator(
                    {'condition': cond, 'y': real_y}, training=training
                )
            else:
                if isinstance(inputs, (tuple, list)):
                    real_y = inputs[0]
                else:
                    real_y = inputs
                dis_real_y = self.discriminator(real_y, training=training)
            return dis_real_y

    def __init__(self, gen_model, dis_model, data, conditional=False,
                 noise_fn=None, idt_loss_coef=0):
        """Initializes data and GANModel.
        params:
            gen_model: A compiled keras model, which is the generator
            dis_model: A compiled keras model, which is the discriminator
            data: A dictionary containg train data
                  and optionally validation and test data.
                  If the train/validation/test key is present without
                  the _x/_y the value will be used as a
                  generator/Keras-Sequence/TF-Dataset and
                  keys with _x/_y will be ignored.
                  Ex. {'train_x': [...], 'train_y: [...]}
                  Ex. {'train': generator()}
                  Ex. {'train': tf.data.Dataset(), 'test': generator()}
            conditional: A boolean, which determines if the GAN is a
                         conditional GAN
            noise_fn: A TF function that takes a shape and returns
                      noise (Default: normal noise)
            idt_loss_coef: A float, which is the amount of the identity
                           loss (generator model's loss function)
                           to be added to the generator loss
        """
        if conditional:
            names = gen_model.input_names
            if len(names) == 2:
                if 'condition' not in names or 'noise' not in names:
                    raise ValueError('gen_model input names are invalid')
            else:
                raise ValueError('gen_model should have two inputs')
            names = dis_model.input_names
            if len(names) == 2:
                if 'condition' not in names or 'y' not in names:
                    raise ValueError('dis_model input names are invalid')
            else:
                raise ValueError('dis_model should have two inputs')
        else:
            if len(gen_model.input_names) != 1:
                raise ValueError('gen_model should only have one input')
            if len(dis_model.input_names) != 1:
                raise ValueError('dis_model should only have one input')

        self.model = GANTrainer.GANModel(
            gen_model, dis_model, conditional=conditional,
            noise_fn=noise_fn, idt_loss_coef=idt_loss_coef
        )
        self.model.compile(optimizer=dis_model.optimizer,
                           loss=dis_model.loss,
                           metrics=dis_model.compiled_metrics._metrics)
        self.gen_model = gen_model
        self.dis_model = dis_model
        self.model_names = ['gen_model', 'dis_model']
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
            if not self.model.conditional:
                raise ValueError('Not conditional but x '
                                 'and y data provided')
            self.train_data = (data['train_x'], data['train_y'])
        elif 'train_x' in data:
            if self.model.conditional:
                raise ValueError('Conditional but x and '
                                 'y data not provided')
            self.train_data = (data['train_x'], None)
        elif 'train_y' in data:
            if self.model.conditional:
                raise ValueError('Conditional but x and '
                                 'y data not provided')
            self.train_data = (data['train_y'], None)
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
        elif 'validation_x' in data:
            self.validation_data = (data['validation_x'],
                                    np.ones(data['validation_x'].shape[0]))
        elif 'validation_y' in data:
            self.validation_data = (data['validation_y'],
                                    np.ones(data['validation_y'].shape[0]))
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
        elif 'test_x' in data:
            self.test_data = (data['test_x'],
                              np.ones(data['test_x'].shape[0]))
        elif 'test_y' in data:
            self.test_data = (data['test_y'],
                              np.ones(data['test_y'].shape[0]))
        elif 'test' in data:
            if isinstance(data['test'], Trainer.GEN_DATA_TYPES):
                self.test_data = data['test']
            else:
                raise ValueError(
                    f'test data must be of type {Trainer.GEN_DATA_TYPES}. '
                    f'Use test_x/_y for keys if using ndarrays.'
                )

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
        note = Trainer.load(self, path, custom_objects=custom_objects)
        optimizer = self.model.optimizer
        loss = self.model.loss
        metrics = self.model.compiled_metrics._metrics
        conditional = self.model.conditional
        noise_fn = self.model.noise_fn
        idt_loss_coef = self.model.idt_loss_coef
        self.model = GANTrainer.GANModel(
            self.gen_model, self.dis_model, conditional=conditional,
            noise_fn=noise_fn, idt_loss_coef=idt_loss_coef
        )
        self.model.compile(optimizer=optimizer, loss=loss,
                           metrics=metrics)
        return note


class GANPredictor(Predictor):
    """Generative Adversarial Network Predictor is used for
       loading and predicting keras GAN models.
    """

    def __init__(self, path,  weights_name='gen_model_weights.h5',
                 model_name='gen_model.json', custom_objects=None):
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


class GANITrainer(Trainer):
    """Generative Adversarial Network with provided Inputs
       Trainer is used for loading, saving, and training
       keras GAN models that do not have random inputs.
    """

    class GANModel(keras.Model):
        def __init__(self, generator, discriminator,
                     idt_loss_coef=0, **kwargs):
            """GAN Keras Model that has a modified train_step.
            params:
                generator: The generative model
                discriminator: The discriminative model
                idt_loss_coef: A float, which is the amount of the identity
                               loss (generator model's loss function)
                               to be added to the generator loss
            """
            super().__init__(**kwargs)
            self.generator = generator
            self.discriminator = discriminator
            self.generator.compiled_loss.build(
                tf.zeros(self.generator.output_shape[1:])
            )
            self.idt_loss_fn = self.generator.compiled_loss._losses[0]
            self.idt_loss_coef = idt_loss_coef

        def train_step(self, batch):
            """Trains the model 1 step.
            params:
                batch: A tuple/list of 2 tensors
            """
            x, real_y = batch
            with tf.GradientTape(persistent=True) as tape:
                fake_y = self.generator(x, training=True)
                dis_fake_y = self.discriminator({'x': x, 'y': fake_y},
                                                training=True)
                dis_real_y = self.discriminator({'x': x, 'y': real_y},
                                                training=True)

                adv_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.ones_like(dis_fake_y), dis_fake_y
                )
                adv_loss = tf.reduce_mean(adv_loss)
                idt_loss = self.idt_loss_fn(real_y, fake_y)
                idt_loss = tf.reduce_mean(idt_loss)
                gen_loss = adv_loss + idt_loss * self.idt_loss_coef

                dis_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.zeros_like(dis_fake_y), dis_fake_y
                )
                dis_fake_loss = tf.reduce_mean(dis_fake_loss)
                dis_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.ones_like(dis_real_y), dis_real_y
                )
                dis_real_loss = tf.reduce_mean(dis_real_loss)
                dis_loss = dis_fake_loss + dis_real_loss
            gen_grads = tape.gradient(
                gen_loss, self.generator.trainable_variables
            )
            dis_grads = tape.gradient(
                dis_loss, self.discriminator.trainable_variables
            )
            self.generator.optimizer.apply_gradients(
                zip(gen_grads, self.generator.trainable_variables)
            )
            self.discriminator.optimizer.apply_gradients(
                zip(dis_grads, self.discriminator.trainable_variables)
            )
            return {
                'gen_loss': gen_loss,
                'adversarial_loss': adv_loss,
                'identity_loss': idt_loss,
                'discriminator_loss': dis_loss,
                'dis_fake_input_loss': dis_fake_loss,
                'dis_real_input_loss': dis_real_loss
            }

        def call(self, inputs, training=False):
            """Calls the discriminator model on new inputs.
            params:
                inputs: A tensor or list of tensors
                training: A boolean or boolean scalar tensor, indicating
                          whether to run the `Network` in training mode
                          or inference mode
            return: A tensor if there is a single output, or a list of
                    tensors if there are more than one outputs.
            """
            x, real_y = inputs
            dis_real_y = self.discriminator({'x': x, 'y': real_y},
                                            training=training)
            return dis_real_y

    def __init__(self, gen_model, dis_model, data, idt_loss_coef=0):
        """Initializes data and GANModel.
        params:
            gen_model: A compiled keras model, which is the generator
            dis_model: A compiled keras model, which is the discriminator
            data: A dictionary containg train data
                  and optionally validation and test data.
                  If the train/validation/test key is present without
                  the _x/_y the value will be used as a
                  generator/Keras-Sequence/TF-Dataset and
                  keys with _x/_y will be ignored.
                  Ex. {'train_x': [...], 'train_y: [...]}
                  Ex. {'train': generator()}
                  Ex. {'train': tf.data.Dataset(), 'test': generator()}
            idt_loss_coef: A float, which is the amount of the identity
                           loss (generator model's loss function)
                           to be added to the generator loss
        """
        names = dis_model.input_names
        if len(names) == 2:
            if 'x' not in names or 'y' not in names:
                raise ValueError('dis_model input names are invalid')
        else:
            raise ValueError('dis_model should have two inputs')
        if len(gen_model.input_names) != 1:
            raise ValueError('gen_model should only have one input')

        self.model = GANITrainer.GANModel(
            gen_model, dis_model, idt_loss_coef=idt_loss_coef
        )
        self.model.compile(optimizer=dis_model.optimizer,
                           loss=dis_model.loss,
                           metrics=dis_model.compiled_metrics._metrics)
        self.gen_model = gen_model
        self.dis_model = dis_model
        self.model_names = ['gen_model', 'dis_model']
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
        elif 'train_x' in data or 'train_y' in data:
            raise ValueError('Train x and y data must be provided')
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
        elif 'validation_x' in data or 'validation_y' in data:
            raise ValueError('Validation x and y data must be provided')
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
        elif 'test_x' in data or 'test_y' in data:
            raise ValueError('Test x and y data must be provided')
        elif 'test' in data:
            if isinstance(data['test'], Trainer.GEN_DATA_TYPES):
                self.test_data = data['test']
            else:
                raise ValueError(
                    f'test data must be of type {Trainer.GEN_DATA_TYPES}. '
                    f'Use test_x/_y for keys if using ndarrays.'
                )

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
        note = Trainer.load(self, path, custom_objects=custom_objects)
        optimizer = self.model.optimizer
        loss = self.model.loss
        metrics = self.model.compiled_metrics._metrics
        idt_loss_coef = self.model.idt_loss_coef
        self.model = GANITrainer.GANModel(
            self.gen_model, self.dis_model, idt_loss_coef=idt_loss_coef
        )
        self.model.compile(optimizer=optimizer, loss=loss,
                           metrics=metrics)
        return note


class CycleGANTrainer(Trainer):
    """Cycle Generative Adversarial Network Trainer is used for
       loading, saving, and training keras GAN models.
    """

    class GANModel(keras.Model):
        def __init__(self, y_generator, y_discriminator,
                     x_generator, x_discriminator,
                     idt_loss_coef=0, cycle_loss_coef=10, **kwargs):
            """GAN Keras Model that has a modified train_step.
            params:
                y_generator: The generative model that produces y
                y_discriminator: The discriminative model for y inputs
                x_generator: The generative model that produces x
                x_discriminator: The discriminative model for x inputs
                idt_loss_coef: A float, which is the amount of the identity
                               loss (generator models' loss function)
                               to be added to the generator loss
                cycle_loss_coef: A float, which is the amount of the cycle
                                 loss to be added to the gen model loss
            """
            super().__init__(**kwargs)
            self.y_generator = y_generator
            self.y_discriminator = y_discriminator
            self.x_generator = x_generator
            self.x_discriminator = x_discriminator
            self.y_generator.compiled_loss.build(
                tf.zeros(self.y_generator.output_shape[1:])
            )
            self.idt_loss_fn = self.y_generator.compiled_loss._losses[0]
            self.idt_loss_coef = idt_loss_coef
            self.cycle_loss_coef = cycle_loss_coef

        def train_step(self, batch):
            """Trains the model 1 step.
            params:
                batch: A tuple/list of 2 tensors
            """
            real_x, real_y = batch
            with tf.GradientTape(persistent=True) as tape:
                fake_y = self.y_generator(real_x, training=True)
                cycled_x = self.x_generator(fake_y, training=True)
                fake_x = self.x_generator(real_y, training=True)
                cycled_y = self.y_generator(fake_x, training=True)

                same_x = self.x_generator(real_x, training=True)
                same_y = self.y_generator(real_y, training=True)

                dis_real_x = self.x_discriminator(real_x, training=True)
                dis_real_y = self.y_discriminator(real_y, training=True)
                dis_fake_x = self.x_discriminator(fake_x, training=True)
                dis_fake_y = self.y_discriminator(fake_y, training=True)

                y_adv_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.ones_like(dis_fake_y), dis_fake_y
                )
                y_adv_loss = tf.reduce_mean(y_adv_loss)
                x_adv_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.ones_like(dis_fake_x), dis_fake_x
                )
                x_adv_loss = tf.reduce_mean(x_adv_loss)
                y_idt_loss = self.idt_loss_fn(real_y, same_y)
                y_idt_loss = tf.reduce_mean(y_idt_loss)
                x_idt_loss = self.idt_loss_fn(real_x, same_x)
                x_idt_loss = tf.reduce_mean(x_idt_loss)
                cycle_loss = (self.idt_loss_fn(real_x, cycled_x) +
                              self.idt_loss_fn(real_y, cycled_y))
                cycle_loss = tf.reduce_mean(cycle_loss)
                y_gen_loss = (cycle_loss * self.cycle_loss_coef +
                              y_idt_loss * self.idt_loss_coef +
                              y_adv_loss)
                x_gen_loss = (cycle_loss * self.cycle_loss_coef +
                              x_idt_loss * self.idt_loss_coef +
                              x_adv_loss)

                x_dis_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.zeros_like(dis_fake_x), dis_fake_x
                )
                x_dis_fake_loss = tf.reduce_mean(x_dis_fake_loss)
                x_dis_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.ones_like(dis_real_x), dis_real_x
                )
                x_dis_real_loss = tf.reduce_mean(x_dis_real_loss)
                x_dis_loss = x_dis_fake_loss + x_dis_real_loss

                y_dis_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.zeros_like(dis_fake_y), dis_fake_y
                )
                y_dis_fake_loss = tf.reduce_mean(y_dis_fake_loss)
                y_dis_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.ones_like(dis_real_y), dis_real_y
                )
                y_dis_real_loss = tf.reduce_mean(y_dis_real_loss)
                y_dis_loss = y_dis_fake_loss + y_dis_real_loss

            y_gen_grads = tape.gradient(y_gen_loss,
                                        self.y_generator.trainable_variables)
            x_gen_grads = tape.gradient(x_gen_loss,
                                        self.x_generator.trainable_variables)

            y_dis_grads = tape.gradient(
                y_dis_loss, self.y_discriminator.trainable_variables
            )
            x_dis_grads = tape.gradient(
                x_dis_loss, self.x_discriminator.trainable_variables
            )

            self.y_generator.optimizer.apply_gradients(
                zip(y_gen_grads, self.y_generator.trainable_variables)
            )
            self.x_generator.optimizer.apply_gradients(
                zip(x_gen_grads, self.x_generator.trainable_variables)
            )

            self.y_discriminator.optimizer.apply_gradients(
                zip(y_dis_grads, self.y_discriminator.trainable_variables)
            )
            self.x_discriminator.optimizer.apply_gradients(
                zip(x_dis_grads, self.x_discriminator.trainable_variables)
            )

            return {
                'y_gen_loss': y_gen_loss,
                'y_adversarial_loss': y_adv_loss,
                'y_identity_loss': y_idt_loss,
                'y_discriminator_loss': y_dis_loss,
                'y_dis_fake_input_loss': y_dis_fake_loss,
                'y_dis_real_input_loss': y_dis_real_loss,
                'x_gen_loss': x_gen_loss,
                'x_adversarial_loss': x_adv_loss,
                'x_identity_loss': x_idt_loss,
                'x_discriminator_loss': x_dis_loss,
                'x_dis_fake_input_loss': x_dis_fake_loss,
                'x_dis_real_input_loss': x_dis_real_loss,
                'cycle_loss': cycle_loss
            }

        def call(self, inputs, training=False):
            """Calls the discriminator model on new inputs.
            params:
                inputs: A tensor or list of tensors
                training: A boolean or boolean scalar tensor, indicating
                          whether to run the `Network` in training mode
                          or inference mode
            return: A tensor if there is a single output, or a list of
                    tensors if there are more than one outputs.
            """
            real_x, real_y = inputs
            dis_real_x = self.x_discriminator(real_x,
                                              training=training)
            dis_real_y = self.y_discriminator(real_y,
                                              training=training)
            return dis_real_x, dis_real_y

    def __init__(self, gen_model, dis_model, data,
                 idt_loss_coef=0, cycle_loss_coef=10):
        """Initializes data and GANModel.
        params:
            gen_model: A compiled keras model, which is the generator
            dis_model: A compiled keras model, which is the discriminator
            data: A dictionary containg train data
                  and optionally validation and test data.
                  If the train/validation/test key is present without
                  the _x/_y the value will be used as a
                  generator/Keras-Sequence/TF-Dataset and
                  keys with _x/_y will be ignored.
                  Ex. {'train_x': [...], 'train_y: [...]}
                  Ex. {'train': generator()}
                  Ex. {'train': tf.data.Dataset(), 'test': generator()}
            idt_loss_coef: A float, which is the amount of the identity
                           loss (generator model's loss function)
                           to be added to the generator loss
            cycle_loss_coef: A float, which is the amount of the cycle
                             loss to be added to the gen model loss
        """
        if len(dis_model.input_names) != 1:
            raise ValueError('dis_model should only have one input')
        if len(gen_model.input_names) != 1:
            raise ValueError('gen_model should only have one input')
        self.y_gen_model = gen_model
        self.x_gen_model = keras.models.clone_model(gen_model)
        self.x_gen_model.compile(
            optimizer=gen_model.optimizer, loss=gen_model.loss,
            metrics=gen_model.compiled_metrics._metrics
        )
        self.y_dis_model = dis_model
        self.x_dis_model = keras.models.clone_model(dis_model)
        self.x_dis_model.compile(
            optimizer=dis_model.optimizer, loss=dis_model.loss,
            metrics=dis_model.compiled_metrics._metrics
        )
        self.model_names = ['y_gen_model', 'y_dis_model',
                            'x_gen_model', 'x_dis_model']
        self.model = CycleGANTrainer.GANModel(
            self.y_gen_model, self.y_dis_model,
            self.x_gen_model, self.x_dis_model,
            idt_loss_coef=idt_loss_coef,
            cycle_loss_coef=cycle_loss_coef
        )
        self.model.compile(optimizer=dis_model.optimizer,
                           loss=dis_model.loss,
                           metrics=dis_model.compiled_metrics._metrics)
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
        elif 'train_x' in data or 'train_y' in data:
            raise ValueError('Train x and y data must be provided')
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
        elif 'validation_x' in data or 'validation_y' in data:
            raise ValueError('Validation x and y data must be provided')
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
        elif 'test_x' in data or 'test_y' in data:
            raise ValueError('Test x and y data must be provided')
        elif 'test' in data:
            if isinstance(data['test'], Trainer.GEN_DATA_TYPES):
                self.test_data = data['test']
            else:
                raise ValueError(
                    f'test data must be of type {Trainer.GEN_DATA_TYPES}. '
                    f'Use test_x/_y for keys if using ndarrays.'
                )

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
        note = Trainer.load(self, path, custom_objects=custom_objects)
        optimizer = self.model.optimizer
        loss = self.model.loss
        metrics = self.model.compiled_metrics._metrics
        idt_loss_coef = self.model.idt_loss_coef
        cycle_loss_coef = self.model.cycle_loss_coef
        self.model = CycleGANTrainer.GANModel(
            self.y_gen_model, self.y_dis_model,
            self.x_gen_model, self.x_dis_model,
            idt_loss_coef=idt_loss_coef,
            cycle_loss_coef=cycle_loss_coef
        )
        self.model.compile(optimizer=optimizer, loss=loss,
                           metrics=metrics)
        return note
