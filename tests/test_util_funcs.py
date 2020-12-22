"""
Author: Travis Hammond
Version: 12_10_2020
"""

import pytest
import shutil

from paiutils.image import load, save
from paiutils.util_funcs import *


def test_load_directory_datasets():
    datasets = {
        'train_data': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
    }
    os.mkdir('datasets4')
    save_directory_datasets(
        'datasets4', datasets, lambda filename, image: save(filename+'.png', image))
    datasets2 = load_directory_datasets(
        'datasets4', lambda filename: load(filename))
    assert (datasets['train_data'] == datasets2['train_data']).all()

    datasets = {
        'dogs': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8'),
        'cats': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
    }
    os.mkdir('datasets5')
    save_directory_datasets(
        'datasets5', datasets, lambda filename, image: save(filename+'.png', image))
    datasets2 = load_directory_datasets(
        'datasets5', lambda filename: load(filename))
    assert (datasets['dogs'] == datasets2['dogs']).all()
    assert (datasets['cats'] == datasets2['cats']).all()

    datasets = {
        'train_x': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8'),
        'train_y': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
    }
    os.mkdir('datasets6')
    save_directory_datasets('datasets6', datasets, {'_x': lambda filename, image: save(filename+'.png', image),
                                                    '_y': lambda filename, image: save(filename+'.png', image)})
    datasets2 = load_directory_datasets('datasets6',
                                        {'_x': lambda filename: load(filename),
                                         '_y': lambda filename: load(filename)})
    assert (datasets['train_x'] == datasets2['train_x']).all()
    assert (datasets['train_y'] == datasets2['train_y']).all()

    shutil.rmtree('datasets4')
    shutil.rmtree('datasets5')
    shutil.rmtree('datasets6')


def test_save_directory_datasets():
    datasets = {
        'train_data': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
    }
    os.mkdir('datasets1')
    save_directory_datasets(
        'datasets1', datasets, lambda filename, image: save(filename+'.png', image))
    assert len(os.listdir('datasets1')) == 1
    assert len(os.listdir('datasets1/train_data')) == 10

    datasets = {
        'dogs': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8'),
        'cats': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
    }
    os.mkdir('datasets2')
    save_directory_datasets(
        'datasets2', datasets, lambda filename, image: save(filename+'.png', image))
    assert len(os.listdir('datasets2')) == 2
    assert len(os.listdir('datasets2/dogs')) == 10
    assert len(os.listdir('datasets2/cats')) == 10

    datasets = {
        'train_x': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8'),
        'train_y': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
    }
    os.mkdir('datasets3')
    save_directory_datasets('datasets3', datasets, {'_x': lambda filename, image: save(filename+'.png', image),
                                                    '_y': lambda filename, image: save(filename+'.png', image)})
    assert len(os.listdir('datasets3')) == 2
    assert len(os.listdir('datasets3/train_x')) == 10
    assert len(os.listdir('datasets3/train_y')) == 10

    shutil.rmtree('datasets1')
    shutil.rmtree('datasets2')
    shutil.rmtree('datasets3')


def test_load_directory_database():
    datasets = {
        'train_x': {
            'data': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
        },
        'train_y': {
            'data': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
        }
    }
    os.mkdir('datasets9')
    save_directory_database('datasets9', datasets, {'_x': lambda filename, image: save(filename+'.png', image),
                                                    '_y': lambda filename, image: save(filename+'.png', image)})
    datasets2 = load_directory_database('datasets9',
                                        {'_x': lambda filename: load(filename),
                                         '_y': lambda filename: load(filename)})
    assert (datasets['train_x']['data'] == datasets2['train_x']['data']).all()
    assert (datasets['train_y']['data'] == datasets2['train_y']['data']).all()

    datasets = {
        'train_x': {
            'dogs': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8'),
            'cats': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
        },
        'train_y': {
            'dogs': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8'),
            'cats': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
        }
    }
    os.mkdir('datasets10')
    save_directory_database('datasets10', datasets, {'_x': lambda filename, image: save(filename+'.png', image),
                                                     '_y': lambda filename, image: save(filename+'.png', image)})
    datasets2 = load_directory_database('datasets10',
                                        {'_x': lambda filename: load(filename),
                                         '_y': lambda filename: load(filename)})
    assert (datasets['train_x']['dogs'] == datasets2['train_x']['dogs']).all()
    assert (datasets['train_x']['cats'] == datasets2['train_x']['cats']).all()
    assert (datasets['train_y']['dogs'] == datasets2['train_y']['dogs']).all()
    assert (datasets['train_y']['cats'] == datasets2['train_y']['cats']).all()

    shutil.rmtree('datasets9')
    shutil.rmtree('datasets10')


def test_save_directory_database():
    datasets = {
        'train_x': {
            'data': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
        },
        'train_y': {
            'data': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
        }
    }
    os.mkdir('datasets7')
    save_directory_database('datasets7', datasets, {'_x': lambda filename, image: save(filename+'.png', image),
                                                    '_y': lambda filename, image: save(filename+'.png', image)})
    assert len(os.listdir('datasets7')) == 2
    assert len(os.listdir('datasets7/train_x')) == 1
    assert len(os.listdir('datasets7/train_y')) == 1
    assert len(os.listdir('datasets7/train_x/data')) == 10
    assert len(os.listdir('datasets7/train_y/data')) == 10

    datasets = {
        'train_x': {
            'dogs': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8'),
            'cats': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
        },
        'train_y': {
            'dogs': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8'),
            'cats': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
        }
    }
    os.mkdir('datasets8')
    save_directory_database('datasets8', datasets, {'_x': lambda filename, image: save(filename+'.png', image),
                                                    '_y': lambda filename, image: save(filename+'.png', image)})
    assert len(os.listdir('datasets8')) == 2
    assert len(os.listdir('datasets8/train_x')) == 2
    assert len(os.listdir('datasets8/train_y')) == 2
    assert len(os.listdir('datasets8/train_x/dogs')) == 10
    assert len(os.listdir('datasets8/train_y/dogs')) == 10
    assert len(os.listdir('datasets8/train_x/cats')) == 10
    assert len(os.listdir('datasets8/train_y/cats')) == 10

    shutil.rmtree('datasets7')
    shutil.rmtree('datasets8')


def test_load_file_datasets():
    datasets = {
        'train_x': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8'),
        'train_y': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
    }
    save_file_datasets('data.h5', datasets)
    datasets2 = load_file_datasets('data.h5')
    assert (datasets['train_x'] == datasets2['train_x']).all()
    assert (datasets['train_y'] == datasets2['train_y']).all()
    os.remove('data.h5')


def test_save_file_datasets():
    datasets = {
        'train_x': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8'),
        'train_y': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
    }
    save_file_datasets('data2.h5', datasets)
    assert os.path.isfile('data2.h5')
    os.remove('data2.h5')


def test_load_datasets():
    datasets = {
        'dogs': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8'),
        'cats': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
    }
    save_datasets('datasets12', datasets, lambda filename,
                  image: save(filename+'.png', image))
    datasets2 = load_datasets(
        'datasets12', lambda filename: load(filename+'.png'))
    print(datasets['dogs'].shape, datasets2['dogs'].shape)
    print(datasets['dogs'][0][0], datasets2['dogs'][0][0])
    print(datasets['cats'][0][0], datasets2['cats'][0][0])
    assert (datasets['dogs'] == datasets2['dogs']).all()
    assert (datasets['cats'] == datasets2['cats']).all()
    shutil.rmtree('datasets12')


def test_save_datasets():
    datasets = {
        'dogs': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8'),
        'cats': np.random.randint(0, 255, size=(10, 32, 32, 3), dtype='uint8')
    }
    save_datasets('datasets11', datasets, lambda filename,
                  image: save(filename+'.png', image))
    assert len(os.listdir('datasets11')) == 21
    shutil.rmtree('datasets11')


def test_write():
    os.mkdir('datasets13')
    write([('dogs', 'dog0'), ('cats', 'cat0')], 'datasets13')
    assert len(os.listdir('datasets13')) == 1
    shutil.rmtree('datasets13')


def test_read():
    os.mkdir('datasets14')
    write([('dogs', 'dog0'), ('cats', 'cat0')], 'datasets14')
    mappings = read('datasets14')
    assert mappings == [('dogs', 'dog0'), ('cats', 'cat0')]
    shutil.rmtree('datasets14')
