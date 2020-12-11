"""
Author: Travis Hammond
Version: 12_10_2020
"""


import os
import csv
import numpy as np
import h5py


def load_directory_datasets(path, file_loader):
    """Loads datasets from a directory.
            Directory
             //   \\
            //     \\
           //       \\
         dogs       cats
         //           \\
    data files     data files
    params:
        path: A string, which is a path to the datasets
        file_loader: A function for loading each file, or
                     a dictionary of postfix identifiers as keys
                     and file loader functions for values
    return: A dictionary, which contains the datasets
    """

    def load(path, fl):
        data = []
        for file in np.sort(os.listdir(path)):
            data.append(fl(os.path.join(path, file)))
        return np.array(data)

    folders = [folder for folder in os.listdir(path)
               if os.path.isdir(os.path.join(path, folder))]
    datasets = {}
    if isinstance(file_loader, dict):
        file_loaders = file_loader
        for folder in folders:
            for postfix, file_loader in file_loaders.items():
                if folder.endswith(postfix):
                    datasets[folder] = load(os.path.join(path, folder),
                                            file_loader)
    else:
        for folder in folders:
            datasets[folder] = load(os.path.join(path, folder), file_loader)
    return datasets


def _create_folder(path):
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


def save_directory_datasets(path, datasets, file_saver):
    """Saves datasets to a directory.
            Directory
             //   \\
            //     \\
           //       \\
         dogs       cats
         //           \\
    data files     data files
    params:
        path: A string, which is a path to the datasets
        datasets: A dictionary, which contains the datasets
        file_saver: A function for saving each file, or a
                    dictionary of postfix identifiers as keys
                    and file loader functions for values
    """
    _create_folder(path)

    def save(path, dataset, fs):
        for ndx, data in enumerate(dataset):
            fs(os.path.join(path, f'{ndx}'), data)

    if isinstance(file_saver, dict):
        file_savers = file_saver
        for name, dataset in datasets.items():
            dpath = os.path.join(path, name)
            _create_folder(dpath)
            for postfix, file_saver in file_savers.items():
                if name.endswith(postfix):
                    save(dpath, dataset, file_saver)
                    break
    else:
        for name, dataset in datasets.items():
            dpath = os.path.join(path, name)
            _create_folder(dpath)
            save(dpath, dataset, file_saver)


def load_directory_database(path, file_loaders):
    """Loads groups of datasets from a directory.
            Directory
             //   \\
            //     \\
           //       \\
        train_x     train_y
         //           \\
       // \\         // \\
     dogs  cats    dogs  cats
      ||    ||      ||    ||
    files  files  files  files <- Groups should share file names
    params:
        path: A string, which is a path to the datasets
        file_loaders: A dictionary of postfix identifiers as keys
                      and file loader functions for values
    return: A dictionary, which contains the dataset groups
    """
    folders = [folder for folder in os.listdir(path)
               if os.path.isdir(os.path.join(path, folder))]
    database = {}
    for folder in folders:
        for postfix, file_loader in file_loaders.items():
            if folder.endswith(postfix):
                database[folder] = load_directory_datasets(
                    os.path.join(path, folder), file_loader
                )
                break
    return database


def save_directory_database(path, database, file_savers):
    """Saves groups of datasets from a directory.
            Directory
             //   \\
            //     \\
           //       \\
        train_x     train_y
         //           \\
       // \\         // \\
     dogs  cats    dogs  cats
      ||    ||      ||    ||
    files  files  files  files <- Groups should share order
    params:
        path: A string, which is a path to the datasets
        database: A dictionary, which contains the groups of datasets
        file_loaders: A dictionary of postfix identifiers as keys
                      and file loader functions for values
    return: A dictionary, which contains the dataset groups
    """
    _create_folder(path)
    for name, datasets in database.items():
        dpath = os.path.join(path, name)
        _create_folder(dpath)
        for postfix, file_saver in file_savers.items():
            if name.endswith(postfix):
                save_directory_datasets(dpath, datasets, file_saver)
                break


def load_file_datasets(path):
    """Loads datasets from a file.
    params:
        path: A string, which is a path to the datasets
    return: A dictionary, which contains the datasets
    """
    datasets = {}
    with h5py.File(path, 'r') as hf:
        for name in hf:
            datasets[name] = hf[name][:]
    return datasets


def save_file_datasets(path, datasets):
    """Saves datasets to a file.
    params:
        path: A string, which is a path to the datasets
        datasets: A dictionary, which contains the datasets
    """
    with h5py.File(path, 'w') as file:
        for name, dataset in datasets.items():
            file.create_dataset(name, data=dataset)


def load_datasets(path, file_loader):
    mappings = read(path)
    datasets = {}
    for label, filename in mappings:
        if label not in datasets:
            datasets[label] = []
        datasets[label].append(file_loader(os.path.join(path, filename)))
    return {label: np.array(dataset) for label, dataset in datasets.items()}


def save_datasets(path, datasets, file_saver):
    _create_folder(path)
    mappings = []
    for label, dataset in datasets.items():
        for ndx, data in enumerate(dataset):
            filename = f'{label}_{ndx}'
            mappings.append((label, filename))
            file_saver(os.path.join(path, filename), data)
    write(mappings, path)


def write(mappings, path, **kwargs):
    """Creates or appends a mappings csv file.
    params:
        mappings: A dictionary
        path: A string for the path to save the csv file to
    """
    newfile = not os.path.isfile(os.path.join(path, 'mappings.csv'))
    with open(
        os.path.join(path, 'mappings.csv'), 'a+', newline=''
    ) as csvfile:
        writer = csv.writer(csvfile)
        if newfile:
            writer.writerow(['label', 'filename'])
        for label, filename in mappings:
            writer.writerow([label, filename])


def read(path, **kwargs):
    """Loads a mapping file.
    params:
        path: A string from which to load the mappings.csv file
    return: A list of dictionaries
    """
    mappings = []
    with open(os.path.join(path, 'mappings.csv'), newline='') as csvfile:
        reader = csv.reader(csvfile, **kwargs)
        reader.__next__()
        for label, filename in reader:
            mappings.append((label, filename))
    return mappings
