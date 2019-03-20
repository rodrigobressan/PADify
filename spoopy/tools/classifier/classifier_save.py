from __future__ import division

import glob
import json
import os
import os.path

import sys

sys.path.append('/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy')

import numpy as np
from keras.applications import VGG16, VGG19
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from tools.classifier import evaluate_hter
from tools.file_utils import file_helper

import os
import os.path as path

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_NAME = 'mnist_convnet'
EPOCHS = 1
BATCH_SIZE = 128

np.random.seed(1)

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input


def get_X_y(dir, extension, is_loading_done):
    cur_dir = os.getcwd()
    os.chdir(dir)  # the parent folder with sub-folders

    list_fams, no_imgs, num_samples = get_number_samples(extension)

    y, indexes = compute_labels(list_fams, no_imgs, num_samples)
    X, samples_names = fetch_samples_and_names(list_fams, num_samples, extension, is_loading_done)

    os.chdir(cur_dir)

    # Encoding classes (y) into integers (y_encoded) and then generating one-hot-encoding (Y)
    encode_classes(y)
    return X, y, indexes, samples_names


def encode_classes(y_train):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_encoded = encoder.transform(y_train)

# add rule dataset
def fetch_samples_and_names(list_fams, num_samples, extension, is_loaded):
    width, height, channels = (224, 224, 3)
    X_train = np.zeros((num_samples, width, height, channels))
    cnt = 0
    samples_names = []
    print("Processing images ...")
    for i in range(len(list_fams)):
        print('current fam: ', i)
        # here change to fetch only samples that obey the type rule
        for index, img_file in enumerate(glob.glob(list_fams[i] + '/*.' + extension)):
            # print("[%d] Processing image: %s index: %d" % (cnt, img_file, index))

            if not is_loaded:
                img = image.load_img(img_file, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                X_train[cnt] = x

            cnt += 1
            delimitator = '_frame_'
            index = img_file.find(delimitator)
            samples_names.append(img_file[0:index])
    return X_train, samples_names


def compute_labels(list_fams, no_imgs, num_samples):
    y_train = np.zeros(num_samples)
    pos = 0
    label = 0
    indexes = []
    for i in no_imgs:
        indexes.append(i)
        print("Label:%2d\tFamily: %15s\tNumber of images: %d" % (label, list_fams[label], i))
        for j in range(i):
            y_train[pos] = label
            pos += 1
        label += 1
    num_classes = label
    return y_train, indexes


def get_number_samples(extension):
    list_fams = sorted(os.listdir(os.getcwd()), key=str.lower)  # vector of strings with family names
    no_imgs = []  # No. of samples per family
    for i in range(len(list_fams)):
        os.chdir(list_fams[i])
        len1 = len(glob.glob('*.' + extension))  # assuming the images are stored as 'jpg'
        no_imgs.append(len1)
        os.chdir('..')
    num_samples = np.sum(no_imgs)  # total number of all samples
    totalImg = num_samples
    return list_fams, no_imgs, num_samples

def is_dataset_classified(base_dir):
    resnet_features_path = os.path.join(base_dir, 'features', 'resnet')

    if not os.path.exists(resnet_features_path):
        return False

    files = os.listdir(resnet_features_path)

    # 7 because its the number of days god took to create earth..
    # just kidding, we use this number because we have 7 output files. is it shitty code? yeah, for now it is, but we
    # are in a hurry right now because of fucking sibgrapi :-)
    return len(files) == 7

def run(base_dir, extension):

    # temp for not running lots of times
    if is_dataset_classified(base_dir):
        print(base_dir, ' done')
        #return

    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    models = [get_resnet()]


    is_loading_img_done = True
    # first check if we need to load images again, just in case features are already extracted
    for model in models:

        done_train = is_feature_already_extracted(base_dir, 'train', model[1])
        done_test = is_feature_already_extracted(base_dir, 'test', model[1])

        print('done train: ', done_train, ' done test: ', done_test)
        if not done_train or not done_test:
            is_loading_img_done = False

    X_train, y_train, indexes_train, samples_train = get_X_y(train_dir, extension, is_loading_img_done)
    X_test, y_test, indexes_test, samples_test = get_X_y(test_dir, extension, is_loading_img_done)


    for model in models:
        print(model[0], model[1])
        classify(X_test, X_train, base_dir, samples_test, y_test, y_train, model[0], model[1])


def classify(X_test, X_train, base_dir, samples_test, y_test, y_train, model, model_alias):
    base_dir = os.path.join(base_dir, 'features')
    file_helper.guarantee_path_preconditions(base_dir)

    resnet50features_train = extract_features(X_train, base_dir, 'train', model, model_alias)
    resnet50features_test = extract_features(X_test, base_dir, 'test', model, model_alias)

    print('extract featuers done, reshaping')
    resnet50features_train = np.reshape(resnet50features_train, (resnet50features_train.shape[0], -1))
    resnet50features_test = np.reshape(resnet50features_test, (resnet50features_test.shape[0], -1))

    print('begin fit')
    top_model = OneVsRestClassifier(BaggingClassifier(SVC(verbose=False), n_jobs=-1))
    top_model.fit(resnet50features_train, y_train)
    export_model(tf.train.Saver(), model, ["conv2d_1_input"], "dense_2/Softmax")

    print('begin predict')
    y_pred = top_model.predict(resnet50features_test)
    y_pred_proba = top_model.predict_proba(resnet50features_test)

    output_dir = os.path.join(base_dir, model_alias)
    file_helper.guarantee_path_preconditions(output_dir)

    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    np.save(os.path.join(output_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(output_dir, "y_pred_proba.npy"), y_pred_proba)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)

    file = open(os.path.join(output_dir, "names_test.txt"), "w")
    file.write(str(json.dumps(samples_test)) + "\n")
    file.close()

    hter, apcer, bpcer = evaluate_hter.evaluate_predictions(output_dir)
    print('HTER: ', hter)

    # Computing the average accuracy
    acc_score = accuracy_score(y_test, y_pred)
    print('acc_score: ', acc_score)


def get_resnet():
    image_shape = (224, 224, 3)
    return [ResNet50(weights='imagenet', input_shape=image_shape, include_top=False), 'resnet']


def get_vgg_16():
    image_shape = (224, 224, 3)
    return [VGG16(weights='imagenet', input_shape=image_shape, include_top=False), 'vgg_16']


def get_vgg_19():
    image_shape = (224, 224, 3)
    return [VGG19(weights='imagenet', input_shape=image_shape, include_top=False), 'vgg_19']

def is_feature_already_extracted(dir, type, model_alias):
    output_path = os.path.join(dir, 'features', model_alias, type + '.npy')
    print('out path: ', output_path)
    return os.path.exists(output_path)

def extract_features(X, dir, type, model, model_alias):
    output_path = os.path.join(dir, model_alias)
    file_helper.guarantee_path_preconditions(output_path)

    file_path = os.path.join(output_path, type + '.npy')
    if os.path.exists(file_path):
        print('Features already present on: ', file_path)
        features = np.load(file_path)
    else:
        print('Features not present yet, predicting now..')
        features = model.predict(X)
        np.save(file_path, features)
        print('Features saved on: ', file_path)
    return features


BASE_PATH = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/evaluate/range'

def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
                         MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
                              False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
                              "save/restore_all", "save/Const:0", \
                              'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name],
        tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")

def classify_all_datasets():
    datasets = file_helper.get_dirs_from_folder(BASE_PATH)
    for dataset in datasets:
        dataset_path = os.path.join(BASE_PATH, dataset)
        types_attacks = file_helper.get_dirs_from_folder(dataset_path)

        for type_attack in types_attacks:
            attack_path = os.path.join(dataset_path, type_attack)
            features = file_helper.get_dirs_from_folder(attack_path)

            for feature in features:
                if feature != 'features':  # this folder is where we keep the results from the extraction
                    feature_path = os.path.join(dataset_path, type_attack, feature)
                    print(feature_path)
                    run(feature_path, 'jpg')


if __name__ == '__main__':
    run('/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/evaluate/range/cbsr/print/raw', 'jpg')
    #classify_all_datasets()
