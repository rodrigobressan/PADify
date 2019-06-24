import matplotlib
from keras.utils import np_utils

matplotlib.use('Agg')

import glob
from typing import Tuple, List

import keras

import numpy as np
import os
from keras_applications.imagenet_utils import preprocess_input
from keras_preprocessing import image
from os.path import join

from refactored.classification.finetuning.base_finetuner import BaseFinetuner
from refactored.feature_extraction.cnn_model import CnnModel
from refactored.preprocessing.property.property_extractor import PropertyExtractor

import math
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class InterFinetuningClassifier(BaseFinetuner):
    def classify_inter_dataset(self):
        for origin, target, prop, model in self._list_inter_combinations(self.images_root_path):
            try:
                self._classify_inter_dataset(dataset_origin=origin,
                                             dataset_target=target,
                                             model=model,
                                             prop=prop)
            except Exception as e:
                print('exception on origin %s target %s prop %s model %s' % (
                origin, target, prop.get_property_alias(), model.alias))

    def _list_inter_combinations(self, path: str):
        # datasets = os.listdir(path)
        datasets = ["cbsr", "ra"]
        for dataset_origin in datasets:
            for dataset_target in [element for element in datasets if element != dataset_origin]:

                for prop, model in self._list_variations():
                    yield [dataset_origin, dataset_target, model, prop]

    def _get_all_subsets_from_dataset(self,
                                      dataset_alias: str,
                                      model: CnnModel,
                                      prop: PropertyExtractor) -> Tuple[np.ndarray, np.ndarray, List[str]]:

        train_path = join(self.images_root_path, dataset_alias, self.target_all, "train")
        test_path = join(self.images_root_path, dataset_alias, self.target_all, "test")

        X_train, y_train, indexes_train, names_train = self._get_dataset_contents(model, train_path,
                                                                                  prop.get_property_alias())
        X_test, y_test, indexes_test, names_test = self._get_dataset_contents(model, test_path,
                                                                              prop.get_property_alias())

        features_current = np.vstack((X_train, X_test))
        labels_current = np.hstack((y_train, y_test))

        names_current = names_train + names_test

        return features_current, labels_current, names_current

    def _classify_inter_dataset(self,
                                dataset_origin: str,
                                dataset_target: str,
                                model: CnnModel,
                                prop: PropertyExtractor):

        output_dir = join(self.inter_dataset_output,
                          dataset_origin,
                          dataset_target,
                          self.target_all,
                          prop.get_property_alias(),
                          model.alias)

        if os.path.exists(output_dir):
            print('Dataset %s for property %s with model %s already trained' % (
            dataset_origin, prop.get_property_alias(), model.alias))
            return
        # train_path = join(self.images_root_path, dataset_origin, self.target_all, "train")
        # test_path = join(self.images_root_path, dataset_target, self.target_all, "test")

        X_train, y_train, names_train = self._get_all_subsets_from_dataset(dataset_origin, model, prop)
        X_test, y_test, names_test = self._get_all_subsets_from_dataset(dataset_target, model, prop)

        y_train = np_utils.to_categorical(y_train, 2)
        y_test = np_utils.to_categorical(y_test, 2)
        # fitting
        #
        num_train_samples = X_train.shape[0]
        num_valid_samples = X_test.shape[0]

        num_train_steps = math.floor(num_train_samples / self.BATCH_SIZE)
        num_valid_steps = math.floor(num_valid_samples / self.BATCH_SIZE)

        gen = keras.preprocessing.image.ImageDataGenerator()
        #
        train_data = gen.flow(X_train, y_train, shuffle=True, batch_size=self.BATCH_SIZE)
        test_data = gen.flow(X_test, y_test, shuffle=True, batch_size=self.BATCH_SIZE)
        #
        finetuned_model, history, time_callback = self.train(train_data, test_data, model.get_model(), num_train_steps,
                                                             num_valid_steps)

        y_pred = self._predict(finetuned_model, X_test)
        results = self._evaluate_results(y_pred, y_test, names_test)

        print('HTER: %f\nAPCER: %f\nBPCER: %f' % (results[0], results[1], results[2]))

        self._save_artifacts(finetuned_model, history, output_dir, y_pred, results, time_callback)

    def _get_dataset_contents(self,
                              model: CnnModel,
                              dataset_root_path: str,
                              property_alias: str) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
        """
        Used to get all the contents (features, labels, indexes, list of families) from a given dataset
        :param dataset_root_path: the dataset root
        :param property_alias: the property which we are looking
        :return: a Tuple containing the features, targets, indexes and the names of the samples
        """
        cur_dir = os.getcwd()
        os.chdir(dataset_root_path)  # the parent folder with sub-folders

        list_fams, no_imgs, num_samples = self._get_labels_info(property_alias)

        y, indexes = self._fetch_labels(list_fams, no_imgs, num_samples)
        X, samples_names = self._get_X_and_names(model, list_fams, num_samples, property_alias)

        os.chdir(cur_dir)

        return X, y, indexes, samples_names

    def _get_labels_info(self, property_alias: str) -> Tuple[List[str], List, np.ndarray]:
        """
        Used to get information from a given property
        :param property_alias: the property to be looked
        :return: a Tuple containing the List of families, the number of images and the number of samples
        """
        list_fams = sorted(os.listdir(os.getcwd()), key=str.lower)  # vector of strings with family names
        no_imgs = []  # No. of samples per family
        for i in range(len(list_fams)):
            os.chdir(join(list_fams[i], property_alias))
            len1 = len(self._fetch_all_images('./'))  # assuming the images are stored as 'jpg'
            no_imgs.append(len1)
            os.chdir('../..')
        num_samples = np.sum(no_imgs)  # total number of all samples
        return list_fams, no_imgs, num_samples

    def _fetch_labels(self, list_fams, no_imgs, num_samples) -> Tuple[np.ndarray, List]:
        """
        Used to fetch the labels alongside with the indexes
        :param list_fams: the list of families
        :param no_imgs: the number of images
        :param num_samples: the number of samples
        :return: a Tuple containing the numpy array with the y_train and the list of indexes
        """
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
        return y_train, indexes

    def _fetch_all_images(self, path) -> List[str]:
        """
        Used to fetch all the images from a given dir
        :param path: the path to be looked
        :return: a List containing all the images
        """
        files_all = []

        for ext in self.exts:
            files_all.extend(glob.glob(join(path, ext)))

        return files_all

    def _get_X_and_names(self, model: CnnModel, list_fams, num_samples, property_alias: str) -> Tuple[
        np.ndarray, List[str]]:
        """
        Used to get all the features from a given set along with their names
        :param list_fams: the list of families
        :param num_samples: the number of samples
        :param property_alias: the property to be looked
        :return: a Tuple containing a Numpy array with the features and a List containing the name of the images
        """
        channels, width, height = model.input_shape
        X_train = np.zeros((num_samples, width, height, channels))
        cnt = 0
        samples_names = []
        print("Processing images ...")
        for i in range(len(list_fams)):
            print('current fam: ', i)
            for index, img_file in enumerate(self._fetch_all_images(join(list_fams[i], property_alias))):
                img = image.load_img(img_file, target_size=(width, height))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                X_train[cnt] = x

                cnt += 1
                index = img_file.find(self.frame_delimiter)
                samples_names.append(img_file[0:index])
        return X_train, samples_names
