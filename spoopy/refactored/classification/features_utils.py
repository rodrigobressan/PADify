import json

import numpy as np
import os


def load_features(path: str, subset: str) -> np.ndarray:
    """
    Used to load a given set of features from a specified subset
    :param path: the path where the features are located
    :param subset: their subset (train or test)
    :return: a np.ndarray containing the features
    """
    if subset == 'train':
        file_name = 'X_train.npy'
    elif subset == 'test':
        file_name = 'X_test.npy'

    return np.load(os.path.join(path, file_name))


def concatenate_features(path: str):
    train_features_path = os.path.join(path, 'X_train.npy')
    test_features_path = os.path.join(path, 'X_test.npy')

    train_features = np.load(train_features_path)
    test_features = np.load(test_features_path)

    features_stacked = np.vstack((train_features, test_features))
    return features_stacked


def concatenate_labels(path: str):
    train_labels_path = os.path.join(path, 'y_train.npy')
    test_labels_path = os.path.join(path, 'y_test.npy')

    train_labels = np.load(train_labels_path)
    test_labels = np.load(test_labels_path)

    labels_stacked = np.hstack((train_labels, test_labels))

    return labels_stacked


def concatenate_names(path: str):
    train_names_path = os.path.join(path, 'samples_train.txt')
    test_names_path = os.path.join(path, 'samples_test.txt')

    names_train = get_file_names(train_names_path)
    names_test = get_file_names(test_names_path)

    names_concat = names_train + names_test
    return names_concat


def concatenate_features_labels_names(path: str):
    combined_features = np.full((0, 0, 0, 0), 0)
    combined_labels = np.full(0, 0)

    train_features_path = os.path.join(path, 'train.npy')
    train_labels_path = os.path.join(path, 'y_train.npy')
    train_names_path = os.path.join(path, 'names_train.txt')

    test_features_path = os.path.join(path, 'test.npy')
    test_labels_path = os.path.join(path, 'y_test.npy')
    test_names_path = os.path.join(path, 'names_test.txt')

    train_features = np.load(train_features_path)
    train_labels = np.load(train_labels_path)

    test_features = np.load(test_features_path)
    test_labels = np.load(test_labels_path)

    features_current = np.vstack((train_features, test_features))
    labels_current = np.hstack((train_labels, test_labels))

    names_train = get_file_names(train_names_path)
    names_test = get_file_names(test_names_path)

    names_current = names_train + names_test

    # features
    if combined_features.shape == (0, 0, 0, 0):  # empty
        combined_features = features_current
    else:
        combined_features = np.vstack((combined_features, features_current))

    if combined_labels.shape == 0:
        combined_labels = labels_current
    else:
        combined_labels = np.hstack((combined_labels, labels_current))

    return combined_features, combined_labels, names_current


def get_file_names(names_path):
    with open(os.path.join(names_path), "r") as file:
        lines = file.readlines()

    names = json.loads(lines[0])
    return names
