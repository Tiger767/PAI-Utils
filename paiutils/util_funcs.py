"""
Author: Travis Hammond
Version: 1_8_2020
"""


import os
import numpy as np
import h5py


def load_directory_dataset(path, file_loader_x, file_loader_y=None):
    """Loads datasets from a directory.
            Directory
              //  \\
            //     \\
           //       \\
        train_x   train_y
         //           \\
    data files     data files
            Directory
                |
            train_data
            // // \\ \\
           folders with
           names that will
           be converted to
           onehot encoddings
           and x data files
           in corresponding
           folder
    params:
        path: A string, which is a path to the dataset
        file_loader: A function for loading each X file
        file_loader_y: A function for loading each Y file
    return: A dictionary, which contains the dataset
    """
    dirs = os.listdir(path)

    def load_x(path):
        nonlocal file_loader_x
        data = []
        for file in np.sort(os.listdir(path)):
            data.append(file_loader_x(os.path.join(path, file)))
        return np.array(data)

    def load_y(path):
        nonlocal file_loader_y
        data = []
        for file in np.sort(os.listdir(path)):
            data.append(file_loader_y(os.path.join(path, file)))
        return np.array(data)

    def load_data_from_nested_folders(path):
        nonlocal file_loader_x
        folders = np.sort(os.listdir(path))
        data_x = []
        data_y = []
        for ndx, folder in enumerate(folders):
            folder_path = os.path.join(path, folder)
            if not os.path.isfile(folder_path):
                for file in np.sort(os.listdir(folder_path)):
                    data_x.append(file_loader_x(os.path.join(folder_path,
                                                             file)))
                    data_y.append(ndx)
        return np.array(data_x), np.identity(ndx+1)[data_y]

    dataset = {}
    if 'train_x' in dirs:
        dataset['train_x'] = load_x(os.path.join(path, 'train_x'))
        if file_loader_y is not None:
            if 'train_y' not in dirs:
                raise FileNotFoundError('There must be a train_y folder')
            dataset['train_y'] = load_y(
                os.path.join(path, 'train_y'))
            lx, ly = len(dataset['train_x']), len(dataset['train_y'])
            if lx != ly:
                raise Exception(f'Train data sizes to not match: '
                                f'x({lx}) != y({ly})')
    else:
        if 'train_data' in dirs:
            data = load_data_from_nested_folders(
                os.path.join(path, 'train_data')
            )
            dataset['train_x'], dataset['train_y'] = data
    if 'validation_x' in dirs:
        dataset['validation_x'] = load_x(os.path.join(path, 'validation_x'))
        if file_loader_y is not None:
            if 'validation_y' not in dirs:
                raise FileNotFoundError('There must be a validation_y folder')
            dataset['validation_y'] = load_y(
                os.path.join(path, 'validation_y'))
            lx, ly = len(dataset['validation_x']), len(dataset['validation_y'])
            if lx != ly:
                raise Exception(f'Validation data sizes to not match: '
                                f'x({lx}) != y({ly})')
    else:
        if 'validation_data' in dirs:
            data = load_data_from_nested_folders(
                os.path.join(path, 'validation_data')
            )
            dataset['validation_x'], dataset['validation_y'] = data
    if 'test_x' in dirs:
        dataset['test_x'] = load_x(os.path.join(path, 'test_x'))
        if file_loader_y is not None:
            if 'test_y' not in dirs:
                raise FileNotFoundError('There must be a test_y folder')
            dataset['test_y'] = load_y(os.path.join(path, 'test_y'))
            lx, ly = len(dataset['test_x']), len(dataset['test_y'])
            if lx != ly:
                raise Exception(f'Test data sizes to not match: '
                                f'x({lx}) != y({ly})')
    else:
        if 'test_data' in dirs:
            data = load_data_from_nested_folders(
                os.path.join(path, 'test_data')
            )
            dataset['test_x'], dataset['test_y'] = data
    return dataset


def save_directory_dataset(path, dataset, file_saver_x,
                           file_saver_y=None, labels=None):
    """Saves the dataset to a directory.
            Directory
              //  \\
            //     \\
           //       \\
        train_x   train_y
         //           \\
    data files     data files
            Directory
                |
            train_data
            // // \\ \\
           folders with
           names that will
           be converted to
           onehot encoddings
           and x data files
           in corresponding
           folder
    params:
        path: A string, which is a path to the dataset
        dataset: A dictionary, which contains the dataset
        file_saver: A function for saving each X file
        file_saver_y: A function for saving each Y file
        labels: A list of strings, which will be used to
                make a nested directory structure
    """
    def create_folders(path):
        if os.path.exists(path):
            for folder in os.listdir(path):
                if os.path.isdir(os.path.join(path, folder)):
                    for file in os.listdir(os.path.join(path, folder)):
                        os.remove(os.path.join(path, folder, file))
                    os.rmdir(os.path.join(path, folder))
                else:
                    os.remove(os.path.join(path, folder))
        else:
            os.mkdir(path)

    def save_x(path, dataset):
        nonlocal file_saver_x
        for ndx, data in enumerate(dataset):
            file_saver_x(os.path.join(path, f'{ndx}'), data)

    def save_y(path, dataset):
        nonlocal file_saver_y
        for ndx, data in enumerate(dataset):
            file_saver_y(os.path.join(path, f'{ndx}'), data)

    def save_data_to_nested_folders(path, dataset_x, dataset_y):
        nonlocal file_saver_x
        for label in labels:
            if not os.path.exists(os.path.join(path, label)):
                os.mkdir(os.path.join(path, label))
            else:
                for file in os.listdir(os.path.join(path, label)):
                    os.remove(os.path.join(path, label, file))
        labels_count = [0] * len(labels)
        for x, y in zip(dataset_x, dataset_y):
            ndx = np.argmax(y)
            file_saver_x(os.path.join(path, labels[ndx],
                                      f'{labels_count[ndx]}'),
                         x)
            labels_count[ndx] += 1

    if labels is None:
        if 'train_x' in dataset:
            dpath = os.path.join(path, 'train_x')
            create_folders(dpath)
            save_x(dpath, dataset['train_x'])
        if 'train_y' in dataset:
            dpath = os.path.join(path, 'train_y')
            create_folders(dpath)
            save_y(dpath, dataset['train_y'])
        if 'validation_x' in dataset:
            dpath = os.path.join(path, 'validation_x')
            create_folders(dpath)
            save_x(dpath, dataset['validation_x'])
        if 'validation_y' in dataset:
            dpath = os.path.join(path, 'validation_y')
            create_folders(dpath)
            save_y(dpath, dataset['validation_y'])
        if 'test_x' in dataset:
            dpath = os.path.join(path, 'test_x')
            create_folders(dpath)
            save_x(dpath, dataset['test_x'])
        if 'test_y' in dataset:
            dpath = os.path.join(path, 'test_y')
            create_folders(dpath)
            save_y(dpath, dataset['test_y'])
    else:
        if 'train_x' in dataset:
            dpath = os.path.join(path, 'train_data')
            create_folders(dpath)
            save_data_to_nested_folders(dpath, dataset['train_x'],
                                        dataset['train_y'])
        if 'validation_x' in dataset:
            dpath = os.path.join(path, 'validation_data')
            create_folders(dpath)
            save_data_to_nested_folders(dpath, dataset['validation_x'],
                                        dataset['validation_y'])
        if 'test_x' in dataset:
            dpath = os.path.join(path, 'test_data')
            create_folders(dpath)
            save_data_to_nested_folders(dpath, dataset['test_x'],
                                        dataset['test_y'])


def load_h5py(path):
    """Loads datasets from a file.
    params:
        path: A string, which is a path to the dataset
    return: A dictionary, which contains the dataset
    """
    dataset = {}
    with h5py.File(path, 'r') as hf:
        if 'train_x' in hf:
            dataset['train_x'] = hf['train_x'][:]
        if 'train_y' in hf:
            dataset['train_y'] = hf['train_y'][:]
        if 'validation_x' in hf:
            dataset['validation_x'] = hf['validation_x'][:]
        if 'validation_y' in hf:
            dataset['validation_y'] = hf['validation_y'][:]
        if 'test_x' in hf:
            dataset['test_x'] = hf['test_x'][:]
        if 'test_y' in hf:
            dataset['test_y'] = hf['test_y'][:]
    return dataset


def save_h5py(path, dataset):
    """Saves the dataset to a file.
    params:
        path: A string, which is a path to the dataset
        dataset: A dictionary, which contains the dataset
    """
    with h5py.File(path, 'w') as file:
        if 'train_x' in dataset:
            file.create_dataset('train_x', data=dataset['train_x'])
        if 'train_y' in dataset:
            file.create_dataset('train_y', data=dataset['train_y'])
        if 'validation_x' in dataset:
            file.create_dataset('validation_x', data=dataset['validation_x'])
        if 'validation_y' in dataset:
            file.create_dataset('validation_y', data=dataset['validation_y'])
        if 'test_x' in dataset:
            file.create_dataset('test_x', data=dataset['test_x'])
        if 'test_y' in dataset:
            file.create_dataset('test_y', data=dataset['test_y'])


if __name__ == '__main__':
    dataset = {
        'train_x': np.random.random((50, 2)),
        'train_y': np.random.random((50, 2)),
        'validation_x': np.random.random((10, 2)),
        'validation_y': np.random.random((10, 2)),
        'test_x': np.random.random((25, 2)),
        'test_y': np.random.random((25, 2)),
    }
    labels = ['0', '1']

    def save(path, data):
        with open(path, 'wb') as file:
            file.write(data.tobytes())

    def load(path):
        with open(path, 'rb') as file:
            return np.frombuffer(file.read())

    if not os.path.exists('data1'):
        os.mkdir('data1')
    if not os.path.exists('data2'):
        os.mkdir('data2')

    print(dataset['validation_x'])
    print(dataset['validation_y'])

    save_directory_dataset('data1', dataset, save, labels=labels)
    dataset2 = load_directory_dataset('data1', load)
    print(dataset2['validation_x'])
    print(dataset2['validation_y'])

    save_directory_dataset('data2', dataset, save, save)
    dataset2 = load_directory_dataset('data2', load, load)
    print(dataset2['validation_x'])
    print(dataset2['validation_y'])

    save_h5py('data.h5', dataset2)
    dataset3 = load_h5py('data.h5')
    print(dataset3['validation_x'])
    print(dataset3['validation_y'])
