import itertools
import sys

sys.path.append('/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy')

import json
import os
import pickle

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

#from classifier import evaluate_hter
from tools.file_utils import file_helper


base_train = [
    '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/evaluate/intra/ra/all/raw',
    '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/evaluate/intra/cbsr/all/raw',
]

def evaluate(origin_base_1, target_base, output_path):

    if (os.path.exists(output_path)):
        print(output_path, ' already exists')
        return

    # todo change here to iterate over every available cnn model
    origin_features_1_path = os.path.join(origin_base_1, 'features', 'resnet')
    target_features_path = os.path.join(target_base,'features', 'resnet')

    origin_features, origin_labels, origin_names = get_features(origin_features_1_path)
    target_features, target_labels, target_names = get_features(target_features_path)

    file_helper.guarantee_path_preconditions(output_path)
    perform_prediction(origin_features, origin_labels, target_features, target_labels, target_names, origin_names, output_path)


def get_features(item_comb_path):
    combined_features = np.full((0, 0, 0, 0), 0)
    combined_labels = np.full(0, 0)

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


def get_labels(labels_path):
    return np.load(labels_path)


def get_file_names(names_path):
    file = open(os.path.join(names_path), "r")
    lines = file.readlines()
    names = json.loads(lines[0])
    return names


def perform_prediction(X_train, y_train, X_test, y_test, target_file_names, train_file_names, output_dir):
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    print('loaded origin features')
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    print('loaded target features')

    top_model = OneVsRestClassifier(BaggingClassifier(SVC(), n_jobs=-1))
    # top_model = svm.SVC(verbose=True)
    # top_model = RandomForestClassifier()
    top_model.fit(X_train, y_train)
    print('fit done')

    y_pred = top_model.predict(X_test)
    y_pred_proba = top_model.predict_proba(X_test)
    print('predict done')

    pickle.dump(top_model, open(os.path.join(output_dir, 'model.sav'), 'wb'))

    np.save(os.path.join(output_dir, "train.npy"), X_train)
    np.save(os.path.join(output_dir, "test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    np.save(os.path.join(output_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(output_dir, "y_pred_proba.npy"), y_pred_proba)

    file = open(os.path.join(output_dir, "names_test.txt"), "w")
    file.write(str(json.dumps(target_file_names)) + "\n")
    file.close()

    file = open(os.path.join(output_dir, "names_train.txt"), "w")
    file.write(str(json.dumps(train_file_names)) + "\n")
    file.close()

    print('done!')
    #hter, apcer, bpcer = evaluate_hter.evaluate_predictions(output_dir)
    #print('HTER: ', hter)



def evaluate_dataset(origin_path, target_path, current_feature, base_output):
    origin_feature_path = os.path.join(origin_path, current_feature, 'features', 'resnet')
    target_feature_path = os.path.join(target_path, current_feature, 'features', 'resnet')

    alias_origin = os.path.basename(origin_path)
    alias_target = os.path.basename(target_path)

    output_path = os.path.join(base_output, alias_origin, alias_target, current_feature)

    if os.path.exists(output_path):
        print('%s already exists' % (output_path))
        return

    print('\n\norigin %s \ntarget %s\noutput %s' % (origin_feature_path, target_feature_path, output_path))
    evaluate(origin_feature_path, target_feature_path, output_path)



"""
print - print
mobile - tablet / highdef
cut - ???
 1 - n
 
 
 treino replay attack (todo), testo em cada um do cbsr
 
 cbsr (todo) -> ra (print), ra (tablet), ra (highdef), ra (all)
 ra (todo) -> cbsr (print), cbsr (mobile) , cbsr (cut), cbsr (tablet)
 
 cbsr (print) -> ra (print)
 cbsr (mobile) -> ra (tablet) + ra (highdef)
 
"""
def evaluate_all_cross(base_features, base_output):
    datasets = file_helper.get_dirs_from_folder(base_features)
    for dataset_origin in datasets:

        origin_path = os.path.join(base_features, dataset_origin)

        # types_attack = all, cut, print, tablet, etc.
        types_attack = file_helper.get_dirs_from_folder(origin_path)

        for attack_origin in types_attack:
            full_path_origin = os.path.join(origin_path, attack_origin)
            features_origin = file_helper.get_dirs_from_folder(full_path_origin)

            for feature_origin in features_origin:
                full_path_origin = os.path.join(origin_path, attack_origin, feature_origin)
                print(full_path_origin)
                targets_datasets = file_helper.get_dirs_from_folder(base_features)
                for dataset_target in targets_datasets:
                    if dataset_target != dataset_origin and attack_origin != 'all':
                        continue

                    attacks_target = os.path.join(base_features, dataset_target)

                    for attack_target in os.listdir(attacks_target):
                        features_target = os.path.join(base_features, dataset_target, attack_target)

                        # if (dataset_target == dataset_origin and attack_target != attack_origin):
                        #     continue

                        for feature_target in os.listdir(features_target):

                            if feature_target == feature_origin:
                                full_path_target = os.path.join(features_target, feature_target)
                                output_path = os.path.join(base_output, dataset_origin, attack_origin, dataset_target,
                                                           attack_target, feature_target)

                                print('  target: %s' % (full_path_target))
                                try:
                                    evaluate(full_path_origin, full_path_target, output_path)
                                except Exception as e:
                                    print(e)
#
# def evaluate_combinations():
#     datasets = os.listdir(BASE_PATH_DATASETS)
#
#     combinations = []
#     for L in range(0, len(datasets) + 1):
#         for subset in itertools.combinations(datasets, L):
#             combinations.append(subset)
#
#     # ['cbsr', 'nuaa', ...]
#     # for origin_dataset in datasets:
#     #     dataset_dir = os.path.join(BASE_PATH_DATASETS, origin_dataset, 'all')
#     #
#     #     # ['depth', 'saliency', 'illum', ...]
#     #     for dataset_type in os.listdir(dataset_dir):
#     #         type_dir = os.path.join(dataset_dir, dataset_type)
#
#             # for combination in combinations:
#             #     print('combination: ', combination)
#             #     output_path = ""
#             #     concatenated_features = []
#             #     concatenated_labels = []
#             #     # for every possible item (cbsr, nuaa) inside the given combination
#             #     for item_combination in combination:
#             #         print(' item combination: ', item_combination)
#             #         item_combination_dir = os.path.join(BASE_PATH_DATASETS, item_co1'mbination, 'all', dataset_type)
#             #         print('   item_cbmb_dir: ', item_combination_dir)
#             #         if output_path != '':
#             #             output_path = output_path + "_" + item_combination
#             #         else:
#             #             output_path = item_combination
#             #
#
#                 ####

    # for all the possible combinations between the datasets



if __name__ == '__main__':
    #evaluate_combinations()
    evaluate_all_cross()

