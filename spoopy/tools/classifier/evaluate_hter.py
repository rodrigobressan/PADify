from __future__ import division

import sys

import os
from statistics import mode

from pandas import *

from tools.file_utils import file_helper


def evaluate_with_values(y_pred, y_test, names_test):
    dict_results = extract_results(names_test, y_pred, y_test)
    count_fake, count_real, fa, fr = analyze_results(dict_results)
    hter, apcer, bpcer = get_metrics(count_fake, count_real, fa, fr)
    return hter, apcer, bpcer


def evaluate_predictions(path):
    names, y_pred, y_test = load_file_info(path)
    dict_results = extract_results(names, y_pred, y_test)
    count_fake, count_real, fa, fr = analyze_results(dict_results)
    hter, apcer, bpcer = get_metrics(count_fake, count_real, fa, fr)
    return hter, apcer, bpcer


def get_metrics(count_fake, count_real, fa, fr):
    bpcer = fr / count_real
    apcer = fa / count_fake
    hter = (apcer + bpcer) / 2

    if hter == 0:
        print('woah')
    return hter, apcer, bpcer


def analyze_results(dict_results):
    fa = 0
    fr = 0
    count_real = 0
    count_fake = 0
    for result in dict_results:
        try:
            mode_predictions = mode(dict_results[result][0])
            truth = dict_results[result][1][0]

            if truth == 0:  # fake
                count_fake = count_fake + 1
                if mode_predictions != 0:
                    fa = fa + 1
            elif truth == 1:  # real
                count_real = count_real + 1
                if mode_predictions != 1:
                    fr = fr + 1
        except Exception as e:
            print(e)

    return count_fake, count_real, fa, fr


def extract_results(names, y_pred, y_test):
    dict_results = {}
    for i, prediction in enumerate(y_pred):
        current_id = names[i]

        if current_id not in dict_results:
            dict_results[current_id] = []
            dict_results[current_id].append([])  # prediction
            dict_results[current_id].append([])  # real

        dict_results[current_id][0].append(prediction)
        dict_results[current_id][1].append(y_test[i])
    return dict_results


def load_file_info(path):
    file = open(os.path.join(path, "names_test.txt"), "r")
    lines = file.readlines()
    names = pandas.io.json.loads(lines[0])
    y_test = np.load(os.path.join(path, 'y_test.npy'))
    y_pred = np.load(os.path.join(path, 'y_pred.npy'))
    return names, y_pred, y_test


BASE_PATH = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/evaluate/cross_dataset'
BASE_PATH_INTRA = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/evaluate/intra'
BASE_PATH_COMBINATION = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/static/evaluate/cross_dataset_combinations'


def evaluate_all_datasets_combination():
    results = [
        ['Origin', 'Target', 'Feature', 'HTER', 'APCER', 'BPCER']
    ]

    datasets_origin = file_helper.get_dirs_from_folder(BASE_PATH_COMBINATION)

    for dataset_origin in datasets_origin:
        print('Origin: ', dataset_origin)
        datasets_target = file_helper.get_dirs_from_folder(os.path.join(BASE_PATH_COMBINATION, dataset_origin))
        for dataset_target in datasets_target:
            print('  Target: ', dataset_target)
            features = file_helper.get_dirs_from_folder(
                os.path.join(BASE_PATH_COMBINATION, dataset_origin, dataset_target))
            for feature in features:
                full_path_features = os.path.join(BASE_PATH_COMBINATION, dataset_origin, dataset_target, feature)
                try:
                    hter, apcer, bpcer = evaluate_predictions(full_path_features)

                    row = [dataset_origin, dataset_target, feature, hter, apcer, bpcer]
                    results.append(row)
                except Exception as e:
                    print(e)

    df = DataFrame(results)
    print(df)
    df.to_csv('results_hter_combinations.csv', sep=' ')


def evaluate_all_datasets():
    results = [
        ['Origin', 'Target', 'Origin Type', 'Target Type', 'Feature', 'HTER', 'APCER', 'BPCER']
    ]
    datasets_origin = file_helper.get_dirs_from_folder(BASE_PATH_INTRA)

    for dataset_origin in datasets_origin:
        attacks_origin = os.listdir(os.path.join(BASE_PATH_INTRA, dataset_origin))
        for attack_origin in attacks_origin:
            datasets_target = file_helper.get_dirs_from_folder(
                (os.path.join(BASE_PATH_INTRA, dataset_origin, attack_origin)))

            for dataset_target in datasets_target:
                attacks = file_helper.get_dirs_from_folder(
                    os.path.join(BASE_PATH_INTRA, dataset_origin, attack_origin, dataset_target))

                for attack_target in attacks:
                    features = os.listdir(
                        os.path.join(BASE_PATH_INTRA, dataset_origin, attack_origin, dataset_target, attack_target))

                    for feature in features:
                        full_path_features = os.path.join(BASE_PATH_INTRA, dataset_origin, attack_origin,
                                                          dataset_target, attack_target, feature)
                        try:
                            hter, apcer, bpcer = evaluate_predictions(full_path_features)

                            row = [dataset_origin, dataset_target, attack_origin, attack_target, feature, hter, apcer,
                                   bpcer]
                            results.append(row)
                        except Exception as e:
                            print(e)

    df = DataFrame(results)
    print(df)
    df.to_csv('results_hter.csv', sep=' ')


if __name__ == '__main__':
    # hter, apcer, bpcer = evaluate_predictions('/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/evaluate/test_cbsr_ra_illum_cross_train')
    # print(hter)
    # print(evaluate_predictions('/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/evaluate/intra/cbsr/all/illumination/features/resnet'))
    evaluate_all_datasets_combination()
    # evaluate_all_datasets_combination()
