"""
Author: Travis Hammond
Version: 12_13_2020
"""

import pytest

from paiutils.neural_network import *
from paiutils.autoencoder import *


def test_autoencoder_trainer():
    x_tdata = np.random.random((10, 100))
    x_vdata = np.random.random((10, 100))
    x_ttdata = np.random.random((10, 100))

    x0 = keras.layers.Input(shape=(100,))
    x = dense(100)(x0)
    output = dense(10)(x)
    encoder_model = keras.Model(inputs=x0, outputs=output)
    encoder_model.compile(optimizer='adam', loss='mse')

    x0 = keras.layers.Input(shape=(10,))
    x = dense(100)(x0)
    output = dense(100)(x)
    decoder_model = keras.Model(inputs=x0, outputs=output)
    decoder_model.compile(optimizer='adam', loss='mse')

    trainer = AutoencoderTrainer(
        encoder_model, decoder_model,
        {'train_x': x_tdata,
         'validation_x': x_vdata,
         'test_x': x_ttdata}
    )
    trainer.train(10, batch_size=32)

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

    trainer.train(10, batch_size=32)

    with pytest.raises(ValueError):
        AutoencoderTrainer(
            encoder_model, decoder_model,
            {'train_a': x_tdata,
             'validation_x': x_vdata,
             'test_x': x_ttdata}
        )

    with pytest.raises(TypeError):
        AutoencoderTrainer(
            encoder_model, decoder_model, 'hello'
        )

    def gen():
        while True:
            yield np.random.random((32, 100)), np.random.random((32, 100))

    trainer = AutoencoderTrainer(
        encoder_model, decoder_model, {'train': gen()}
    )
    trainer.train(5, batch_size=32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (np.zeros((100, 100), dtype=np.float32),
         np.zeros((100, 100), dtype=np.float32))
    ).batch(10)
    trainer = AutoencoderTrainer(encoder_model, decoder_model,
                                 {'train': dataset})
    trainer.train(5, batch_size=10)


def test_autoencoder_predictor():
    x_tdata = np.random.random((10, 100))
    x_vdata = np.random.random((10, 100))
    x_ttdata = np.random.random((10, 100))

    x0 = keras.layers.Input(shape=(100,))
    x = dense(100)(x0)
    output = dense(10)(x)
    encoder_model = keras.Model(inputs=x0, outputs=output)
    encoder_model.compile(optimizer='adam', loss='mse')

    x0 = keras.layers.Input(shape=(10,))
    x = dense(100)(x0)
    output = dense(100)(x)
    decoder_model = keras.Model(inputs=x0, outputs=output)
    decoder_model.compile(optimizer='adam', loss='mse')

    trainer = AutoencoderTrainer(
        encoder_model, decoder_model,
        {'train_x': x_tdata,
         'validation_x': x_vdata,
         'test_x': x_ttdata}
    )
    trainer.train(10, batch_size=32)

    path = trainer.save('')
    predictor1 = AutoencoderPredictor(path)
    predictor2 = AutoencoderPredictor(path, uses_decoder_model=True)
    predictor3 = AutoencoderPredictor(path, uses_encoder_model=True)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

    assert predictor1.predict(np.random.random((100,))).shape == (100,)
    assert predictor1.predict_all(
        np.random.random((10, 100))).shape == (10, 100)

    assert predictor2.predict(np.random.random((10,))).shape == (100,)
    assert predictor2.predict_all(
        np.random.random((10, 10))).shape == (10, 100)

    assert predictor3.predict(np.random.random((100,))).shape == (10,)
    assert predictor3.predict_all(
        np.random.random((10, 100))).shape == (10, 10)


def test_autoencoder_extra_decoder_trainer():
    x_tdata = np.random.random((10, 100))
    y_tdata = np.random.random((10, 100))
    x_vdata = np.random.random((10, 100))
    y_vdata = np.random.random((10, 100))
    x_ttdata = np.random.random((10, 100))
    y_ttdata = np.random.random((10, 100))

    x0 = keras.layers.Input(shape=(100,))
    x = dense(100)(x0)
    output = dense(10)(x)
    encoder_model = keras.Model(inputs=x0, outputs=output)
    encoder_model.compile(optimizer='adam', loss='mse')

    x0 = keras.layers.Input(shape=(10,))
    x = dense(100)(x0)
    output = dense(100)(x)
    decoder_model = keras.Model(inputs=x0, outputs=output)
    decoder_model.compile(optimizer='adam', loss='mse')

    x0 = keras.layers.Input(shape=(10,))
    x = dense(100)(x0)
    output = dense(100)(x)
    decoder_model2 = keras.Model(inputs=x0, outputs=output)
    decoder_model2.compile(optimizer='adam', loss='mse')

    trainer = AutoencoderExtraDecoderTrainer(
        encoder_model, decoder_model, decoder_model2,
        {'train_x': x_tdata, 'train_y': y_tdata,
         'validation_x': x_vdata, 'validation_y': y_vdata,
         'test_x': x_ttdata, 'test_y': y_ttdata}
    )
    trainer.train(10, batch_size=32)
    trainer.train_extra_decoder(10, batch_size=32)

    results = trainer.eval_extra_decoder(batch_size=32)
    assert len(results) == 3
    results = trainer.eval_extra_decoder(
        validation_data=False, batch_size=32
    )
    assert len(results) == 2
    results = trainer.eval_extra_decoder(
        validation_data=False, test_data=False, batch_size=32
    )
    assert len(results) == 1
    results = trainer.eval_extra_decoder(
        batch_size=32, verbose=False
    )
    assert len(results) == 3

    path = trainer.save('')
    note = trainer.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

    trainer.train(10, batch_size=32)
    trainer.train_extra_decoder(10, batch_size=32)

    trainer = AutoencoderExtraDecoderTrainer(
        encoder_model, decoder_model, decoder_model2,
        {'train_x': x_tdata, 'train_y': y_tdata,
         'validation_x': x_vdata, 'validation_y': y_vdata,
         'test_x': x_ttdata, 'test_y': y_ttdata},
        include_y_data=False,
    )
    trainer.train(10, batch_size=32)
    trainer.train_extra_decoder(10, batch_size=32)

    path = trainer.save('')
    note = trainer.load(path)
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))
    os.rmdir(path)

    trainer.train(10, batch_size=32)
    trainer.train_extra_decoder(10, batch_size=32)

    with pytest.raises(ValueError):
        AutoencoderExtraDecoderTrainer(
            encoder_model, decoder_model, decoder_model2,
            {'train_a': x_tdata, 'train_y': y_tdata,
             'validation_x': x_vdata, 'validation_y': y_vdata,
             'test_x': x_ttdata, 'test_y': y_ttdata}
        )

    with pytest.raises(TypeError):
        AutoencoderExtraDecoderTrainer(
            encoder_model, decoder_model, decoder_model2, 'hello'
        )


def test_vae_trainer():
    x_tdata = np.random.random((10, 100))
    x_vdata = np.random.random((10, 100))
    x_ttdata = np.random.random((10, 100))

    x0 = keras.layers.Input(shape=(100,))
    x = dense(100)(x0)
    z_mean = dense(5)(x)
    z_log_var = dense(5)(x)
    encoder_model = keras.Model(inputs=x0, outputs=[z_mean, z_log_var])
    encoder_model.compile(optimizer='adam', loss='mse')

    x0 = keras.layers.Input(shape=(5,))
    x = dense(100)(x0)
    output = dense(100, activation='sigmoid')(x)
    decoder_model = keras.Model(inputs=x0, outputs=output)
    decoder_model.compile(optimizer='adam', loss='mse')

    trainer = VAETrainer(
        encoder_model, decoder_model,
        {'train_x': x_tdata,
         'validation_x': x_vdata,
         'test_x': x_ttdata}
    )
    trainer.train(10, batch_size=32)

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

    trainer.train(10, batch_size=32)

    with pytest.raises(ValueError):
        VAETrainer(
            encoder_model, decoder_model,
            {'train_a': x_tdata,
             'validation_x': x_vdata,
             'test_x': x_ttdata}
        )

    with pytest.raises(TypeError):
        VAETrainer(
            encoder_model, decoder_model, 'hello'
        )

    def gen():
        while True:
            yield np.random.random((32, 100)), np.random.random((32, 100))

    trainer = VAETrainer(
        encoder_model, decoder_model, {'train': gen()}
    )
    trainer.train(5, batch_size=32)

    dataset = tf.data.Dataset.from_tensor_slices(
        np.zeros((10, 100), dtype=np.float32)
    ).batch(10)
    trainer = VAETrainer(encoder_model, decoder_model,
                         {'train': dataset})
    trainer.train(5, batch_size=10)
