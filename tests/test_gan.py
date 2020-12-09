"""
Author: Travis Hammond
Version: 12_8_2020
"""

import pytest

from paiutils.gan import *
from paiutils.neural_network import *


def test_gan_trainer():
    y_data = np.random.random((10, 12, 12, 1))

    noise_input = keras.layers.Input(shape=(100,), name='noise')
    x = dense(3*3*32)(noise_input)
    x = keras.layers.Reshape((3, 3, 32))(x)
    x = conv2d(64, 3, strides=2, transpose=True)(x)
    output = conv2d(1, 3, strides=2,
                    activation='tanh', batch_norm=False,
                    transpose=True)(x)
    gen_model = keras.Model(inputs=noise_input, outputs=output)
    gen_model.optimizer = keras.optimizers.Adam(.0002, .5)

    inputs = keras.layers.Input(shape=(12, 12, 1))
    x = conv2d(64, 3, strides=2, activation=None,
                batch_norm=False)(inputs)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv2d(128, 3, strides=2, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv2d(256, 3, strides=2, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Flatten()(x)
    outputs = dense(1, activation=None, batch_norm=False)(x)
    dis_model = keras.Model(inputs=inputs, outputs=outputs)
    dis_model.optimizer = keras.optimizers.Adam(.0002, .5)

    trainer = GANTrainer(gen_model, dis_model, y_data)
    trainer.train(2, 32, 0)
    trainer.train(2, 32, 10)

    trainer = GANTrainer(gen_model, dis_model, {'train_y': y_data})
    trainer.train(2, 32, 0)

    trainer = GANTrainer(gen_model, dis_model, {'train_y': y_data},
                         normal_distribution=False)
    trainer.train(2, 32, 0)

    x_data = np.random.random((10, 100))

    with pytest.raises(TypeError):
        GANTrainer(gen_model, dis_model, "hello")

    with pytest.raises(ValueError):
        GANTrainer(gen_model, dis_model,
                   {'train_x': x_data, 'train_y': y_data})

    with pytest.raises(ValueError):
        GANTrainer(gen_model, dis_model, [x_data, y_data])

    with pytest.raises(ValueError):
        GANTrainer(gen_model, dis_model,
                   [x_data, y_data, y_data])

    with pytest.raises(ValueError):
        GANTrainer(gen_model, dis_model,
                   {'train_y': y_data},
                   conditional=True)

    noise_input = keras.layers.Input(shape=(100,), name='noise')
    x_input = keras.layers.Input(shape=(100,), name='x')
    x = dense(3*3*32)(noise_input)
    x = keras.layers.Reshape((3, 3, 32))(x)
    x = conv2d(64, 3, strides=2, transpose=True)(x)
    outputs = conv2d(1, 3, strides=2,
                     activation='tanh', batch_norm=False,
                     transpose=True)(x)
    gen_model = keras.Model(inputs=[noise_input, x_input], outputs=outputs)
    gen_model.optimizer = keras.optimizers.Adam(.0002, .5)

    x_input = keras.layers.Input(shape=(100,), name='x')
    y_input = keras.layers.Input(shape=(12, 12, 1), name='y')
    x = conv2d(64, 3, strides=2, activation=None,
                batch_norm=False)(y_input)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv2d(128, 3, strides=2, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv2d(256, 3, strides=2, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Flatten()(x)
    x2 = dense(256, activation=None)(x_input)
    x2 = keras.layers.LeakyReLU(alpha=0.2)(x2)
    x = keras.layers.Concatenate()([x, x2])
    outputs = dense(1, activation=None, batch_norm=False)(x)
    dis_model = keras.Model(inputs=[x_input, y_input], outputs=outputs)
    dis_model.optimizer = keras.optimizers.Adam(.0002, .5)

    trainer = GANTrainer(gen_model, dis_model, [x_data, y_data],
                         conditional=True)
    trainer.train(2, 32, 0)
    trainer.train(2, 32, 10)

    trainer = GANTrainer(gen_model, dis_model,
                         {'train_x': x_data, 'train_y': y_data},
                         conditional=True)
    trainer.train(2, 32, 0)

    trainer = GANTrainer(gen_model, dis_model,
                         {'train_x': x_data, 'train_y': y_data},
                         conditional=True, normal_distribution=False)
    trainer.train(2, 32, 0)

    path = trainer.save('')
    trainer.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

    with pytest.raises(TypeError):
        GANTrainer(gen_model, dis_model, "hello",
                   conditional=True)

    with pytest.raises(ValueError):
        GANTrainer(gen_model, dis_model,
                   {'train_x': x_data, 'train_a': y_data},
                   conditional=True)

    with pytest.raises(ValueError):
        GANTrainer(gen_model, dis_model, [x_data],
                   conditional=True)

    with pytest.raises(ValueError):
        GANTrainer(gen_model, dis_model,
                   [x_data, y_data, y_data],
                   conditional=True)

    with pytest.raises(ValueError):
        GANTrainer(gen_model, dis_model,
                   {'train_x': x_data, 'train_y': y_data},
                   conditional=False)

def test_gan_predictor():
    y_data = np.random.random((10, 12, 12, 1))

    noise_input = keras.layers.Input(shape=(100,), name='noise')
    x = dense(3*3*32)(noise_input)
    x = keras.layers.Reshape((3, 3, 32))(x)
    x = conv2d(64, 3, strides=2, transpose=True)(x)
    output = conv2d(1, 3, strides=2,
                    activation='tanh', batch_norm=False,
                    transpose=True)(x)
    gen_model = keras.Model(inputs=noise_input, outputs=output)
    gen_model.optimizer = keras.optimizers.Adam(.0002, .5)

    inputs = keras.layers.Input(shape=(12, 12, 1))
    x = conv2d(64, 3, strides=2, activation=None,
                batch_norm=False)(inputs)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv2d(128, 3, strides=2, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv2d(256, 3, strides=2, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Flatten()(x)
    outputs = dense(1, activation=None, batch_norm=False)(x)
    dis_model = keras.Model(inputs=inputs, outputs=outputs)
    dis_model.optimizer = keras.optimizers.Adam(.0002, .5)

    trainer = GANTrainer(gen_model, dis_model, y_data)

    path = trainer.save('')
    predictor = GANPredictor(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

    assert predictor.predict(np.random.random((100,))).shape == (12, 12, 1)
    assert predictor.predict_all(np.random.random((10, 100))).shape == (10, 12, 12, 1)
    assert predictor.random_normal_predict().shape == (12, 12, 1)
    assert predictor.random_uniform_predict().shape == (12, 12, 1)

def test_gani_trainer():
    x_data = np.random.random((10, 12, 12, 1))
    y_data = np.random.random((10, 12, 12, 1))

    inputs = keras.layers.Input(shape=(12, 12, 1))
    x = conv2d(64, 3)(inputs)
    output = conv2d(1, 3, activation='tanh', batch_norm=False)(x)
    gen_model = keras.Model(inputs=inputs, outputs=output)
    gen_model.optimizer = keras.optimizers.Adam(.0002, .5)

    x_input = keras.layers.Input(shape=(12, 12, 1), name='x')
    y_input = keras.layers.Input(shape=(12, 12, 1), name='y')
    x = conv2d(64, 3, strides=2, activation=None,
                batch_norm=False)(x_input)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv2d(128, 3, strides=2, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv2d(256, 3, strides=2, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Flatten()(x)

    x2 = conv2d(64, 3, strides=2, activation=None,
                batch_norm=False)(y_input)
    x2 = keras.layers.LeakyReLU(alpha=0.2)(x2)
    x2 = conv2d(128, 3, strides=2, activation=None)(x2)
    x2 = keras.layers.LeakyReLU(alpha=0.2)(x2)
    x2 = conv2d(256, 3, strides=2, activation=None)(x2)
    x2 = keras.layers.LeakyReLU(alpha=0.2)(x2)
    x2 = keras.layers.Flatten()(x2)
    x = keras.layers.Concatenate()([x, x2])
    outputs = dense(1, activation=None, batch_norm=False)(x)
    dis_model = keras.Model(inputs=[x_input, y_input], outputs=outputs)
    dis_model.optimizer = keras.optimizers.Adam(.0002, .5)

    trainer = GANITrainer(gen_model, dis_model, [x_data, y_data])
    trainer.train(2, 32, 0)
    trainer.train(2, 32, 10)

    trainer = GANITrainer(gen_model, dis_model,
                          {'train_x': x_data, 'train_y': y_data})
    trainer.train(2, 32, 0)

    path = trainer.save('')
    trainer.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

    with pytest.raises(TypeError):
        GANITrainer(gen_model, dis_model, y_data)

    with pytest.raises(ValueError):
        GANITrainer(gen_model, dis_model,
                    {'train_x': x_data, 'train_a': y_data})

    with pytest.raises(ValueError):
        GANITrainer(gen_model, dis_model, [x_data])

    with pytest.raises(ValueError):
        GANITrainer(gen_model, dis_model,
                    [x_data, y_data, y_data])

def test_cycle_gan_trainer():
    x_data = np.random.random((10, 12, 12, 1))
    y_data = np.random.random((10, 12, 12, 1))

    inputs = keras.layers.Input(shape=(12, 12, 1))
    x = conv2d(64, 3)(inputs)
    output = conv2d(1, 3, activation='tanh', batch_norm=False)(x)
    gen_model = keras.Model(inputs=inputs, outputs=output)
    gen_model.optimizer = keras.optimizers.Adam(.0002, .5)

    inputs = keras.layers.Input(shape=(12, 12, 1))
    x = conv2d(64, 3, strides=2, activation=None,
                batch_norm=False)(inputs)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv2d(128, 3, strides=2, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv2d(256, 3, strides=2, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Flatten()(x)
    outputs = dense(1, activation=None, batch_norm=False)(x)
    dis_model = keras.Model(inputs=inputs, outputs=outputs)
    dis_model.optimizer = keras.optimizers.Adam(.0002, .5)

    trainer = CycleGANTrainer(gen_model, dis_model, [x_data, y_data])
    trainer.train(2, 32, 0)
    trainer.train(2, 32, 10)

    trainer = CycleGANTrainer(gen_model, dis_model,
                              {'train_x': x_data, 'train_y': y_data})
    trainer.train(2, 32, 0)

    path = trainer.save('')
    trainer.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

    with pytest.raises(TypeError):
        CycleGANTrainer(gen_model, dis_model, y_data)

    with pytest.raises(ValueError):
        CycleGANTrainer(gen_model, dis_model,
                        {'train_x': x_data, 'train_a': y_data})

    with pytest.raises(ValueError):
        CycleGANTrainer(gen_model, dis_model, [x_data])

    with pytest.raises(ValueError):
        CycleGANTrainer(gen_model, dis_model,
                        [x_data, y_data, y_data])
