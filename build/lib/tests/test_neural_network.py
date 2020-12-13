"""
Author: Travis Hammond
Version: 12_13_2020
"""

import pytest

from paiutils.neural_network import *


def test_trainer():
    x_tdata = np.random.random((10, 100))
    y_tdata = np.random.random((10, 10))
    x_vdata = np.random.random((10, 100))
    y_vdata = np.random.random((10, 10))
    x_ttdata = np.random.random((10, 100))
    y_ttdata = np.random.random((10, 10))

    x0 = keras.layers.Input(shape=(100,))
    x = dense(100)(x0)
    output = dense(10)(x)
    model = keras.Model(inputs=x0, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    trainer = Trainer(model, {'train_x': x_tdata, 'train_y': y_tdata,
                              'validation_x': x_vdata, 'validation_y': y_vdata,
                              'test_x': x_ttdata, 'test_y': y_ttdata})
    trainer.train(5, batch_size=32)

    results = trainer.eval(batch_size=32)
    assert len(results) == 3
    results = trainer.eval(validation_data=False, batch_size=32)
    assert len(results) == 2
    results = trainer.eval(validation_data=False, test_data=False,
                           batch_size=32)
    assert len(results) == 1
    results = trainer.eval(batch_size=32, verbose=False)
    assert len(results) == 3

    path = trainer.save('')
    note = trainer.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

    trainer.train(5, batch_size=32)

    with pytest.raises(TypeError):
        Trainer(model, x_tdata)

    with pytest.raises(ValueError):
        Trainer(model, {'train_x': x_tdata})

    def gen():
        while True:
            yield np.random.random((32, 100)), np.random.random((32, 10))

    trainer = Trainer(model, {'train': gen()})
    trainer.train(5, batch_size=32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (np.zeros((100, 100)), np.zeros((100, 10)))
    ).batch(10)
    trainer = Trainer(model, {'train': dataset})
    trainer.train(5, batch_size=10)

    dataset = tf.data.Dataset.from_tensor_slices(
        (np.zeros((10, 100)), np.zeros((10, 10)))
    ).batch(10)
    dataset2 = tf.data.Dataset.from_tensor_slices(
        (np.zeros((10, 100)), np.zeros((10, 10)))
    ).batch(10)
    trainer = Trainer(model, {'train': dataset,
                              'validation': dataset2})
    trainer.train(5, batch_size=10)
    results = trainer.eval(batch_size=10)
    assert len(results) == 2

def test_predictor():
    x_tdata = np.random.random((10, 100))
    y_tdata = np.random.random((10, 10))

    x0 = keras.layers.Input(shape=(100,))
    x = dense(100)(x0)
    output = dense(10)(x)
    model = keras.Model(inputs=x0, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    trainer = Trainer(model, {'train_x': x_tdata, 'train_y': y_tdata})
    path = trainer.save('')
    predictor = Predictor(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

    assert predictor.predict(np.random.random((100,))).shape == (10,)
    assert predictor.predict_all(np.random.random((10, 100))).shape == (10, 10)

def test_dense():
    x_tdata = np.random.random((10, 100))
    y_tdata = np.random.random((10, 10))

    x0 = keras.layers.Input(shape=(100,))
    x = dense(32, activation='relu')(x0)
    d = dense(32, activation='linear')
    x1 = d(x)
    x2 = d(x)
    x = keras.layers.Concatenate()([x1, x2])
    x = dense(32, l1=.5, l2=.5)(x)
    x = dense(32, batch_norm=False)(x)
    output = dense(10, momentum=.99, epsilon=1e-6)(x)
    model = keras.Model(inputs=x0, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_tdata, y_tdata, epochs=2)

def test_conv1d():
    x_tdata = np.random.random((10, 32, 100))
    y_tdata = np.random.random((10, 10))

    x0 = keras.layers.Input(shape=(32, 100))
    x = conv1d(32, 3, activation='relu')(x0)
    d = conv1d(32, 3, 2, activation='linear')
    x1 = d(x)
    x2 = d(x)
    x = keras.layers.Concatenate(axis=-1)([x1, x2])
    x = conv1d(16, 3, 2, l1=.5, l2=.5, max_pool_size=3, max_pool_strides=2)(x)
    x = conv1d(16, 3, 2, transpose=True)(x)
    x = conv1d(16, 3, 2, batch_norm=False, upsampling_size=2)(x)
    x = keras.layers.Flatten()(x)
    output = dense(10, momentum=.99, epsilon=1e-6)(x)
    model = keras.Model(inputs=x0, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_tdata, y_tdata, epochs=2)

def test_conv2d():
    x_tdata = np.random.random((10, 32, 32, 3))
    y_tdata = np.random.random((10, 10))

    x0 = keras.layers.Input(shape=(32, 32, 3))
    x = conv2d(32, 3, activation='relu')(x0)
    d = conv2d(32, 3, 2, activation='linear')
    x1 = d(x)
    x2 = d(x)
    x = keras.layers.Concatenate(axis=-1)([x1, x2])
    x = conv2d(16, 3, 2, l1=.5, l2=.5, max_pool_size=3, max_pool_strides=2)(x)
    x = conv2d(16, 3, 2, transpose=True)(x)
    x = conv2d(16, 3, 2, batch_norm=False, upsampling_size=2)(x)
    x = keras.layers.Flatten()(x)
    output = dense(10, momentum=.99, epsilon=1e-6)(x)
    model = keras.Model(inputs=x0, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_tdata, y_tdata, epochs=2)

def test_inception():
    x_tdata = np.random.random((10, 32, 32, 3))
    y_tdata = np.random.random((10, 10))

    x0 = keras.layers.Input(shape=(32, 32, 3))
    x = inception([
        [conv2d(16, 3, 2), conv2d(16, 3, 1), conv2d(16, 3, 1)],
        [conv2d(16, 3, 1, max_pool_size=3, max_pool_strides=2)],
        [conv2d(16, 3, 1), conv2d(16, 3, 2), conv2d(32, 3, 1)]
    ])(x0)
    x = keras.layers.Concatenate()(x)
    x = keras.layers.Flatten()(x)
    output = dense(10, momentum=.99, epsilon=1e-6)(x)
    model = keras.Model(inputs=x0, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_tdata, y_tdata, epochs=2)
