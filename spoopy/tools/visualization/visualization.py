import glob

import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
import os

from scikitplot import metrics

import file_utils


def generate_roc(y_true_array, y_probas_array, labels, title, output_path):
    plt.title(title)
    for i in range(len(y_true_array)):

        y_true = y_true_array[i]
        y_probas = y_probas_array[i]
        label = labels[i]

        preds = y_probas[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(y_true, preds)
        roc_auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (label.capitalize(), roc_auc))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(output_path)
    plt.clf()


def visualize_by_fold():
    root = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/evaluate/intra'
    output_base = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/tools/visualization/roc'
    for dataset in glob.glob(os.path.join(root, '*')):
        dataset_dir = os.path.join(root, dataset)
        for fold in glob.glob(os.path.join(dataset_dir, '*')):
            current_fold = os.path.join(dataset_dir, fold)
            title = os.path.basename(dataset).capitalize() + " " + os.path.basename(fold).capitalize()

            types_array = []
            y_test_array = []
            y_proba_array = []
            for type in file_utils.file_helper.get_dirs_from_folder(current_fold):
                if type == 'raw':
                    continue

                type_dir = os.path.join(current_fold, type)

                base_path = os.path.join(type_dir, 'features', 'resnet')
                y_test = np.load(os.path.join(base_path, 'y_test.npy'))
                y_proba = np.load(os.path.join(base_path, 'y_pred_proba.npy'))

                types_array.append(os.path.basename(type))
                y_test_array.append(y_test)
                y_proba_array.append(y_proba)

            output_path = os.path.join(output_base, title.replace(' ', '_').lower() + '_roc.png')
            generate_roc(y_test_array, y_proba_array, types_array, title, output_path)

def visualize_by_map():
    maps = ['depth', 'illumination', 'saliency']
    root = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/evaluate/intra'
    output_base = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/tools/visualization/roc_map'

    for map in maps:
        for dataset in glob.glob(os.path.join(root, '*')):
            title = os.path.basename(dataset).capitalize() + " " + map.capitalize()

            types_array = []
            y_test_array = []
            y_proba_array = []

            dataset_dir = os.path.join(root, dataset)
            for fold in glob.glob(os.path.join(dataset_dir, '*')):
                current_fold = os.path.join(dataset_dir, fold)

                map_dir = os.path.join(current_fold, map)

                base_path = os.path.join(map_dir, 'features', 'resnet')
                y_test = np.load(os.path.join(base_path, 'y_test.npy'))
                y_proba = np.load(os.path.join(base_path, 'y_pred_proba.npy'))

                types_array.append(os.path.basename(fold))
                y_test_array.append(y_test)
                y_proba_array.append(y_proba)

            output_path = os.path.join(output_base, title.replace(' ', '_').lower() + '_roc.png')
            generate_roc(y_test_array, y_proba_array, types_array, title, output_path)

if __name__ == '__main__':
    # visualize_by_map()
    generate_roc(y_test_array, y_proba_array, types_array, title, output_path)
