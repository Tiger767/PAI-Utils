"""
Author: Travis Hammond
Version: 12_28_2020
"""

import pytest
import numpy as np
import os

from paiutils.audio import *


def test_convert_width_to_atype():
    assert convert_width_to_atype(1) == 'int8'
    assert convert_width_to_atype(2) == 'int16'
    with pytest.raises(ValueError):
        convert_width_to_atype(4)

def test_convert_atype_to_width():
    assert convert_atype_to_width('int8') == 1
    assert convert_atype_to_width('int16') == 2
    with pytest.raises(ValueError):
        convert_atype_to_width('int32')

def test_change_rate():
    a = np.random.random(16000 * 2) * 2 - 1

    x, sr, at = change_rate(a, 16000, 8000)
    assert (len(x), sr) == (8000 * 2, 8000)
    assert at == 'int16'

    x, sr, at = change_rate(a, 16000, 8000, atype='int8')
    assert (len(x), sr) == (8000 * 2, 8000)
    assert at == 'int8'

    x, sr, _ = change_rate(a, 16000, 32000)
    assert (len(x), sr) == (32000 * 2, 32000)
    assert np.mean(np.abs(change_rate(x, 32000, 16000)[0] - a)) < .15

    x, sr, _ = change_rate(a, 16000, 44100)
    assert (len(x), sr) == (44100 * 2, 44100)

    a = np.random.random(16000 * 3) * 2 - 1

    x, sr, _ = change_rate(a, 16000, 8000)
    assert (len(x), sr) == (8000 * 3, 8000)

    x, sr, _ = change_rate(a, 16000, 32000)
    assert (len(x), sr) == (32000 * 3, 32000)

    x, sr, _ = change_rate(a, 16000, 44100)
    assert (len(x), sr) == (44100 * 3, 44100)

def test_load():
    with pytest.raises(Exception):
        load('test_load.wav')
    a = np.random.random(16000) * 2 - 1

    save('test_load.wav', a, 16000, atype='int8')

    x, sr, at = load('test_load.wav')
    assert np.mean(np.abs(x - a)) < .004
    assert (sr, at) == (16000, 'int8')

    assert load('test_load.wav', rate=8000)[1:] == (8000, 'int8')

    save('test_load.wav', a, 16000)
    x, sr, at = load('test_load.wav')
    assert np.mean(np.abs(x - a)) < .00004
    assert sr == 16000
    os.remove('test_load.wav')

def test_save():
    a = np.random.random(16000) * 2 - 1

    save('test_save.wav', a, 16000)
    assert os.path.isfile('test_save.wav')
    os.remove('test_save.wav')

    save('test_save.wav', a, 32000)
    assert os.path.isfile('test_save.wav')
    os.remove('test_save.wav')

    save('test_save.wav', a, 16000, atype='int8')
    assert os.path.isfile('test_save.wav')
    os.remove('test_save.wav')

def test_file_record():
    file_record('test_file_record.wav', 1, 16000)
    x, sr, _ = load('test_file_record.wav')
    assert len(x) == 16000
    assert sr == 16000

    file_record('test_file_record.wav', 1, 16000, atype='int8')
    x, sr, at = load('test_file_record.wav')
    assert len(x) == 16000
    assert sr == 16000
    assert at == 'int8'

    file_record('test_file_record.wav', 1, 32000)
    x, sr, _ = load('test_file_record.wav')
    assert len(x) == 32000
    assert sr == 32000

    file_record('test_file_record.wav', 2, 32000)
    x, sr, _ = load('test_file_record.wav')
    assert len(x) == 64000
    assert sr == 32000
    os.remove('test_file_record.wav')

def test_record():
    x, sr, at = record(1, 16000)
    assert len(x) == sr == 16000
    assert at == 'int16'
    assert x.min() >= -1
    assert x.max() <= 1

    x, sr, at = record(1, 16000, atype='int8')
    assert len(x) == sr == 16000
    assert at == 'int8'
    assert x.min() >= -1
    assert x.max() <= 1

    x, sr, at = record(.5, 32000)
    assert len(x) == sr == 16000
    assert at == 'int16'
    assert x.min() >= -1
    assert x.max() <= 1


def test_file_play():
    a = (np.random.random(4000) * 2 - 1) / 1000
    save('test_file_play.wav', a, 16000)
    file_play('test_file_play.wav')
    os.remove('test_file_play.wav')

def test_play():
    a = (np.random.random(4000) * 2 - 1) / 1000
    play(a, 16000)
    play(a, 8000)
    play(a, 1600, atype='int8')

def test_calc_duration():
    a = np.random.random(16000) * 2 - 1
    assert calc_duration(a, 16000) - 1 < .00001
    a = np.random.random(16000) * 2 - 1
    assert calc_duration(a, 32000) - .5 < .00001
    a = np.random.random(32000) * 2 - 1
    assert calc_duration(a, 16000) - 2 < .00001

def test_set_length():
    a = np.random.random(16000) * 2 - 1

    x = set_length(a, 32000)
    assert len(x) == 32000
    assert (x[16000:] == 0).all()

    x = set_length(a, 32000, pad_value=1)
    assert len(x) == 32000
    assert (x[16000:] == 1).all()

    x = set_length(a, 32000, mode='L')
    assert len(x) == 32000
    assert (x[:16000] == 0).all()

    x = set_length(a, 32000, mode='B')
    assert len(x) == 32000
    assert (x[:8000] == 0).all()
    assert (x[-8000:] == 0).all()

def test_set_duration():
    a = np.random.random(16000) * 2 - 1

    x = set_duration(a, 16000, 2)
    assert len(x) == 32000
    assert (x[16000:] == 0).all()

    x = set_duration(a, 16000, 2, pad_value=1)
    assert len(x) == 32000
    assert (x[16000:] == 1).all()

    x = set_duration(a, 16000, 2, mode='L')
    assert len(x) == 32000
    assert (x[:16000] == 0).all()

    x = set_duration(a, 16000, 2, mode='B')
    assert len(x) == 32000
    assert (x[:8000] == 0).all()
    assert (x[-8000:] == 0).all()

    x = set_duration(a, 32000, 1)
    assert len(x) == 32000
    assert (x[16000:] == 0).all()

def test_for_each_frame():
    a = np.random.random(16000) * 2 - 1
    x, sr = for_each_frame(a, 16000, 100 / 16000, np.sum)
    assert np.prod(x.shape) == len(a) / 100
    assert abs(np.sum(a) - np.sum(x)) < .00001

def test_compute_spectrogram():
    pass

def test_convert_spectrogram_to_audio():
    pass

def test_compute_fbank():
    pass

def test_compute_mfcc():
    pass

def test_calc_rms():
    pass

def test_shift_pitch():
    pass

def test_set_power():
    pass

def test_adjust_speed():
    pass

def test_set_speed():
    pass

def test_adjust_volume():
    pass

def test_blend():
    pass

def test_plot():
    pass

def test_convert_audio_to_db():
    pass

def test_convert_power_to_db():
    pass

def test_convert_amplitude_to_db():
    pass

def test_trim_all():
    pass

def test_trim_sides():
    pass

def test_split():
    pass

def test_find_gaps():
    pass

def test_vad_trim_all():
    pass

def test_vad_trim_sides():
    pass

def test_vad_split():
    pass
