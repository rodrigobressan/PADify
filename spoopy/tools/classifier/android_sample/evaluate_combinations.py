import glob
import itertools
import json
import os

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from classifier import evaluate_hter

BASE_PATH_DATASETS = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/evaluate/intra'

maps = [
    "depth",
    "illumination",
    "saliency"
]


def get_data_for_combinations(items_combination, map):
    print("-----")
    combined_features = np.full((0, 0, 0, 0), 0)
    combined_labels = np.full(0, 0)
    combined_names = []
    output = ""
    for item_comb in items_combination:
        item_comb_path = os.path.join(BASE_PATH_DATASETS, item_comb, 'all', map, 'features', 'resnet')
        print('Item comb path: ', item_comb_path)

        train_features_path = os.path.join(item_comb_path, 'train.npy')
        train_labels_path = os.path.join(item_comb_path, 'y_train.npy')
        train_names_path = os.path.join(item_comb_path, 'names_train.txt')

        test_features_path = os.path.join(item_comb_path, 'test.npy')
        test_labels_path = os.path.join(item_comb_path, 'y_test.npy')
        test_names_path = os.path.join(item_comb_path, 'names_test.txt')

        train_features = np.load(train_features_path)
        train_labels = np.load(train_labels_path)

        test_features = np.load(test_features_path)
        test_labels = np.load(test_labels_path)

        features_current = np.vstack((train_features, test_features))
        labels_current = np.hstack((train_labels, test_labels))

        # names
        combined_names = get_file_names(train_names_path) + get_file_names(test_names_path)

        # features
        if combined_features.shape == (0, 0, 0, 0):  # empty
            combined_features = features_current
        else:
            combined_features = np.vstack((combined_features, features_current))

        if combined_labels.shape == 0:
            combined_labels = labels_current
        else:
            combined_labels = np.hstack((combined_labels, labels_current))

        if output == "":
            output = os.path.basename(item_comb)
        else:
            output = output + "_" + os.path.basename(item_comb)

    return combined_features, combined_labels, output, combined_names

def get_file_names(names_path):
    file = open(os.path.join(names_path), "r")
    lines = file.readlines()
    names = json.loads(lines[0])
    return names


def perform_prediction(origin_features, origin_labels, target_features, target_labels, target_file_names, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    origin_features = np.reshape(origin_features, (origin_features.shape[0], -1))
    print('loaded origin features')
    target_features = np.reshape(target_features, (target_features.shape[0], -1))
    print('loaded target features')

    top_model = OneVsRestClassifier(BaggingClassifier(SVC(), n_jobs=-1))
    # top_model = svm.SVC(verbose=True)
    # top_model = RandomForestClassifier()
    top_model.fit(origin_features, origin_labels)
    joblib.dump(top_model, os.path.join(output_dir, 'fitted_model.sav'))
    print('fit done')

    y_pred = top_model.predict(target_features)
    y_pred_proba = top_model.predict_proba(target_features)
    print('predict done')

    np.save(os.path.join(output_dir, "train.npy"), origin_features)
    np.save(os.path.join(output_dir, "test.npy"), target_features)
    np.save(os.path.join(output_dir, "y_test.npy"), target_labels)
    np.save(os.path.join(output_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(output_dir, "y_pred_proba.npy"), y_pred_proba)

    file = open(os.path.join(output_dir, "names_test.txt"), "w")
    file.write(str(json.dumps(target_file_names)) + "\n")
    file.close()

    print('done!')
    hter, apcer, bpcer = evaluate_hter.evaluate_predictions(output_dir)
    print('HTER: ', hter)

BASE_PATH_OUTPUTS = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/evaluate/cross_dataset_combinations'

def evaluate_combinations():
    datasets = glob.glob(os.path.join(BASE_PATH_DATASETS, '*'))

    for L in range(0, len(datasets) + 1):
        for items_combination in itertools.combinations(datasets, L):
            print('items_combination: ', items_combination)

            if len(items_combination) < 1:
                continue

            for map in maps:

                features_origin, labels_origin, output_origin, names_origin = get_data_for_combinations(items_combination, map)
                for dataset_target in datasets:
                    if dataset_target in items_combination:
                        print('target is in combination, aborting...')
                        continue

                    print('output_origin: ', output_origin)

                    output_results = os.path.join(BASE_PATH_OUTPUTS, output_origin, os.path.basename(dataset_target), map)
                    print('output_results: ', output_results)

                    features_target, labels_target, output_target, names_target = get_data_for_combinations([dataset_target], map)
                    perform_prediction(features_origin, labels_origin, features_target, labels_target, names_target, output_results)

                print("output: ", output_origin)
                print("total features: ", features_origin.shape)
                print("total labels: ", labels_origin.shape)



if __name__ == '__main__':
    # combined_features, combined_labels, output, combined_names = get_data_for_combinations(['ra'], 'illumination')
    print('done')
    evaluate_combinations()
