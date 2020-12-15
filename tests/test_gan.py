"""
Author: Travis Hammond
Version: 12_14_2020
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
    gen_model.compile(optimizer=keras.optimizers.Adam(.0002, .5), loss='mse')

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
    dis_model.compile(optimizer=keras.optimizers.Adam(.0002, .5), loss='mse')

    trainer = GANTrainer(gen_model, dis_model, {'train_y': y_data})
    trainer.train(2, 32)
    trainer.train(2, 32)

    def gen():
        while True:
            yield np.random.random((10, 12, 12, 1))

    val_y_data = np.random.random((10, 12, 12, 1))

    trainer = GANTrainer(gen_model, dis_model,
                         {'train': gen(),
                          'validation_y': val_y_data},
                         noise_fn=tf.random.uniform)
    trainer.train(2, 2)

    trainer = GANTrainer(gen_model, dis_model, {'train_y': y_data},
                         noise_fn=tf.random.uniform, idt_loss_coef=10)
    trainer.train(2, 32)

    x_data = np.random.random((10, 14))

    with pytest.raises(TypeError):
        GANTrainer(gen_model, dis_model, "hello")

    with pytest.raises(ValueError):
        GANTrainer(gen_model, dis_model,
                   {'train_x': x_data, 'train_y': y_data})

    with pytest.raises(TypeError):
        GANTrainer(gen_model, dis_model, [x_data, y_data])

    with pytest.raises(ValueError):
        GANTrainer(gen_model, dis_model,
                   {'train_y': y_data},
                   conditional=True)

    cond_input = keras.layers.Input(shape=(14,), name='condition')
    noise_input = keras.layers.Input(shape=(100,), name='noise')
    x = dense(3*3*32)(noise_input)
    x = keras.layers.Reshape((3, 3, 32))(x)
    x = conv2d(64, 3, strides=2, transpose=True)(x)
    outputs = conv2d(1, 3, strides=2,
                     activation='tanh', batch_norm=False,
                     transpose=True)(x)
    gen_model = keras.Model(inputs=[cond_input, noise_input], outputs=outputs)
    gen_model.compile(optimizer=keras.optimizers.Adam(.0002, .5), loss='mse')

    cond_input = keras.layers.Input(shape=(14,), name='condition')
    y_input = keras.layers.Input(shape=(12, 12, 1), name='y')
    x = conv2d(64, 3, strides=2, activation=None,
               batch_norm=False)(y_input)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv2d(128, 3, strides=2, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv2d(256, 3, strides=2, activation=None)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Flatten()(x)
    x2 = dense(256, activation=None)(cond_input)
    x2 = keras.layers.LeakyReLU(alpha=0.2)(x2)
    x = keras.layers.Concatenate()([x, x2])
    outputs = dense(1, activation=None, batch_norm=False)(x)
    dis_model = keras.Model(inputs=[cond_input, y_input], outputs=outputs)
    dis_model.compile(optimizer=keras.optimizers.Adam(.0002, .5), loss='mse')

    trainer = GANTrainer(gen_model, dis_model,
                         {'train_x': x_data, 'train_y': y_data},
                         conditional=True)
    trainer.train(2, 32)
    trainer.train(2, 32)

    trainer = GANTrainer(gen_model, dis_model,
                         {'train_x': x_data, 'train_y': y_data},
                         conditional=True, noise_fn=tf.random.uniform,
                         idt_loss_coef=10)
    trainer.train(2, 32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (np.zeros((100, 14), dtype=np.float32),
         np.zeros((100, 12, 12, 1), dtype=np.float32))
    ).batch(10)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        ((np.zeros((10, 14), dtype=np.float32),
          np.zeros((10, 12, 12, 1), dtype=np.float32)),
         np.ones(10, dtype=np.float32))
    ).batch(10)
    trainer = GANTrainer(gen_model, dis_model,
                         {'train': dataset, 'validation': val_dataset},
                         conditional=True, idt_loss_coef=10)
    trainer.train(2, 10)

    path = trainer.save('')
    note = trainer.load(path)
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

    with pytest.raises(TypeError):
        GANTrainer(gen_model, dis_model, [x_data],
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
    gen_model.compile(optimizer=keras.optimizers.Adam(.0002, .5), loss='mse')

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
    dis_model.compile(optimizer=keras.optimizers.Adam(.0002, .5), loss='mse')

    trainer = GANTrainer(gen_model, dis_model, {'train_y': y_data})

    path = trainer.save('')
    predictor = GANPredictor(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

    assert predictor.predict(np.random.random((100,))).shape == (12, 12, 1)
    assert predictor.predict_all(np.random.random(
        (10, 100))).shape == (10, 12, 12, 1)
    assert predictor.random_normal_predict().shape == (12, 12, 1)
    assert predictor.random_uniform_predict().shape == (12, 12, 1)


def test_gani_trainer():
    x_data = np.random.random((10, 12, 12, 1))
    y_data = np.random.random((10, 12, 12, 1))

    inputs = keras.layers.Input(shape=(12, 12, 1))
    x = conv2d(64, 3)(inputs)
    output = conv2d(1, 3, activation='tanh', batch_norm=False)(x)
    gen_model = keras.Model(inputs=inputs, outputs=output)
    gen_model.compile(optimizer=keras.optimizers.Adam(.0002, .5), loss='mse')

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
    dis_model.compile(optimizer=keras.optimizers.Adam(.0002, .5), loss='mse')

    trainer = GANITrainer(gen_model, dis_model,
                          {'train_x': x_data, 'train_y': y_data})
    trainer.train(2, 32)
    trainer.train(2, 32)

    trainer = GANITrainer(gen_model, dis_model,
                          {'train_x': x_data, 'train_y': y_data},
                          idt_loss_coef=10)
    trainer.train(2, 32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (np.zeros((100, 12, 12, 1), dtype=np.float32),
         np.zeros((100, 12, 12, 1), dtype=np.float32))
    ).batch(10)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        ((np.zeros((10, 12, 12, 1), dtype=np.float32),
          np.zeros((10, 12, 12, 1), dtype=np.float32)),
         np.ones(10, dtype=np.float32))
    ).batch(10)
    trainer = GANITrainer(gen_model, dis_model,
                          {'train': dataset, 'validation': val_dataset},
                          idt_loss_coef=10)
    trainer.train(2, 2)


    path = trainer.save('')
    note = trainer.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

    with pytest.raises(TypeError):
        GANITrainer(gen_model, dis_model, y_data)

    with pytest.raises(ValueError):
        GANITrainer(gen_model, dis_model,
                    {'train_x': x_data, 'train_a': y_data})


def test_cycle_gan_trainer():
    x_data = np.random.random((10, 12, 12, 1))
    y_data = np.random.random((10, 12, 12, 1))

    inputs = keras.layers.Input(shape=(12, 12, 1))
    x = conv2d(64, 3)(inputs)
    output = conv2d(1, 3, activation='tanh', batch_norm=False)(x)
    gen_model = keras.Model(inputs=inputs, outputs=output)
    gen_model.compile(optimizer=keras.optimizers.Adam(.0002, .5), loss='mse')

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
    dis_model.compile(optimizer=keras.optimizers.Adam(.0002, .5), loss='mse')

    trainer = CycleGANTrainer(gen_model, dis_model,
                              {'train_x': x_data, 'train_y': y_data})
    trainer.train(2, 32)
    trainer.train(2, 32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (np.zeros((100, 12, 12, 1), dtype=np.float32),
         np.zeros((100, 12, 12, 1), dtype=np.float32))
    ).batch(10)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        ((np.zeros((10, 12, 12, 1), dtype=np.float32),
          np.zeros((10, 12, 12, 1), dtype=np.float32)),
         (np.ones(10, dtype=np.float32), 
          np.ones(10, dtype=np.float32)))
    ).batch(10)
    trainer = CycleGANTrainer(gen_model, dis_model,
                              {'train': dataset, 'validation': val_dataset},
                              idt_loss_coef=10)
    trainer.train(2, 32)

    path = trainer.save('')
    note = trainer.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

    with pytest.raises(TypeError):
        CycleGANTrainer(gen_model, dis_model, y_data)

    with pytest.raises(ValueError):
        CycleGANTrainer(gen_model, dis_model,
                        {'train_x': x_data, 'train_a': y_data})

    with pytest.raises(ValueError):
        CycleGANTrainer(gen_model, dis_model,
                        {'train_x': x_data, 'train_a': y_data})