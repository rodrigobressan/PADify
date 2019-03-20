import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import glob
import json
from typing import List, Tuple

import numpy as np
import os
from keras_applications.imagenet_utils import preprocess_input
from keras_preprocessing import image
from os.path import join

from refactored.feature_extraction.model import BaseModel, ResNet50Model
from refactored.preprocessing.handler.datahandler import DataHandler, DiskHandler
from refactored.preprocessing.property.property_extractor import PropertyExtractor
from tools.file_utils import file_helper

NAME_FEATURES = 'X_%s.npy'
NAME_TARGETS = 'y_%s.npy'
NAME_SAMPLES = 'samples_%s.txt'


class FeatureExtractor:
    def __init__(self,
                 separated_path: str,
                 output_features: str,
                 models: List[BaseModel],
                 properties: List[PropertyExtractor],
                 data_handler: DataHandler = DiskHandler(),
                 train_alias: str = 'train',
                 test_alias: str = 'test',
                 attack_label: str = 'attack',
                 real_label: str = 'real',):
        self.separated_root = separated_path
        self.models = models
        self.properties = properties

        self.data_handler = data_handler
        self.output_features = output_features

        self.train_alias = train_alias
        self.test_alias = test_alias

        self.IMAGE_WIDTH = 224
        self.IMAGE_HEIGHT = 224
        self.IMAGE_CHANNELS = 3

        self.attack_label = attack_label
        self.real_label = real_label

        self.frame_delimiter = '_frame_'
        self.exts = ('*.jpg', '*.png')

    def perform_extraction(self) -> None:
        """
        This method is used to perform the feature extraction on all the datasets located in the separated_by_pai folder
        """

        # and then later process then
        self._process_datasets_all_frames()
        #

        # clean up
        # shutil.rmtree(self.separated_atk_folder)

    # def _process_attacks(self, datasets):
    #     for dataset in datasets:
    #         datasets_path = join(self.separated_root, dataset)
    #         subset_list = os.listdir(datasets_path)
    #
    #         for subset in subset_list:
    #             attack_path = join(self.separated_root, datasets_path, subset, self.attack_label)
    #             real_path = join(self.separated_root, datasets_path, subset, self.real_label)
    #
    #             attack_list = os.listdir(attack_path)
    #             # cut, print, tablet
    #             for attack in attack_list:
    #
    #                 self._move_attack(attack, subset, dataset)
    #                 # output_path = join(self.separated_atk_folder, attack)

    # def _move_attack(self, attack: str, subset: str, dataset: str) -> None:

    # def _move_all_attacks_to_single_folder(self, attack_path: str,
    #                                        prop: PropertyExtractor,
    #                                        output_path: str) -> None:
    #     """
    #     Used to move all the attacks (mask, tablet, etc) from a given subset (train, test) from a given property (depth,
    #     illumination, saliency) into the same folder.
    #     :param attack_path: where is the root of all the attacks
    #     :param prop: the property which we're looking for
    #     :param output_path: where the files will be stored
    #     """
    #     attacks_list = os.listdir(attack_path)
    #     frames_to_merge = []
    #     for attack in attacks_list:
    #         attack_path_prop = join(attack_path, attack, prop.get_property_alias())
    #         frames_attack = os.listdir(attack_path_prop)
    #         frames_to_merge.extend(frames_attack)
    #
    #         for frame_name in frames_attack:
    #             frame_path = join(attack_path_prop, frame_name)
    #
    #             # format: tablet_2_8_frame_0.jpg
    #             name_with_atk = '%s_%s' % (attack, frame_name)
    #             file_helper.copy_file_rename(frame_path, output_path, name_with_atk)
    #
    # def _move_real_to_temp(self, real_path: str, output_path: str) -> None:
    #     """
    #     Used to move all the real data into a new folder
    #     :param real_path: where the real data is stored
    #     :param output_path: where the data will be stored
    #     """
    #     frames_real = os.listdir(real_path)
    #
    #     for frame in frames_real:
    #         frame_path = join(real_path, frame)
    #         file_helper.copy_file(frame_path, output_path)
    #
    # def _prepare_files(self, datasets: List[str]) -> None:
    #     """
    #     Used to organise the files into the following structure:
    #
    #     Dataset [CBSR, RA]
    #         Subset [Train, Test]
    #             Labels [Real, Fake]
    #                 Frame1.jpg
    #                 Frame2.jpg
    #     :param datasets: the list of datasets
    #     """
    #     for dataset in datasets:
    #         dataset_path = join(self.separated_root, dataset)
    #         subset_list = os.listdir(dataset_path)
    #         for subset in subset_list:
    #             for prop in self.properties:
    #                 # subset root
    #                 subset_path = join(self.separated_root, dataset, subset)
    #
    #                 # where the attack and real folders are located
    #                 attack_path = join(subset_path, self.attack_label)
    #                 real_path = join(subset_path, self.real_label, prop.get_property_alias())
    #
    #                 # where it will be stored
    #                 base_tmp_dir = join(self.separated_atk_folder, dataset, subset)
    #                 attack_merged_path = join(base_tmp_dir, self.attack_label, prop.get_property_alias())
    #                 real_tmp_path = join(base_tmp_dir, self.real_label, prop.get_property_alias())
    #
    #                 # move both attack and real into new dirs
    #                 self._move_all_attacks_to_single_folder(attack_path, prop, attack_merged_path)
    #                 self._move_real_to_temp(real_path, real_tmp_path)

    def _process_datasets_all_frames(self):
        """
        Used to process the datasets once they are the following the structure defined by the method _prepare_files.
        """
        datasets = os.listdir(self.separated_root)
        for dataset in datasets:
            dataset_path = join(self.separated_root, dataset)
            attacks_list = os.listdir(dataset_path)

            for attack in attacks_list:
                attack_path = join(dataset_path, attack)

                for prop in self.properties:
                    property_alias = prop.get_property_alias()

                    path_train = join(attack_path, self.train_alias)
                    path_test = join(attack_path, self.test_alias)

                    model = ResNet50Model()

                    X_train, y_train, indexes_train, samples_train = self._get_dataset_contents(path_train,
                                                                                                property_alias)
                    X_test, y_test, indexes_test, samples_test = self._get_dataset_contents(path_test, property_alias)

                    output_features = join(self.output_features, dataset, attack, property_alias, model.get_alias())

                    features_train = self._fetch_features(X_train, model, output_features, self.train_alias)
                    features_test = self._fetch_features(X_test, model, output_features, self.test_alias)

                    # saving features
                    np.save(join(output_features, (NAME_FEATURES % self.train_alias)), features_train)
                    np.save(join(output_features, (NAME_FEATURES % self.test_alias)), features_test)

                    # saving targets
                    np.save(join(output_features, (NAME_TARGETS % self.train_alias)), y_train)
                    np.save(join(output_features, (NAME_TARGETS % self.test_alias)), y_test)
                    np.save(join(output_features, (NAME_TARGETS % self.test_alias)), y_test)

                    # saving samples names
                    self.__save_txt(join(output_features, (NAME_SAMPLES % self.train_alias)), samples_train)
                    self.__save_txt(join(output_features, (NAME_SAMPLES % self.test_alias)), samples_test)

    def __save_txt(self, output_path: str, content: List[str]) -> None:
        """
        Used to save a given content into a text file
        :param output_path: where the content will be saved
        :param content: the content to be stored
        """
        file = open(output_path, "w")
        file.write(str(json.dumps(content)) + "\n")
        file.close()

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

    def _get_X_and_names(self, list_fams, num_samples, property_alias: str) -> Tuple[np.ndarray, List[str]]:
        """
        Used to get all the features from a given set along with their names
        :param list_fams: the list of families
        :param num_samples: the number of samples
        :param property_alias: the property to be looked
        :return: a Tuple containing a Numpy array with the features and a List containing the name of the images
        """
        width, height, channels = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS)
        X_train = np.zeros((num_samples, width, height, channels))
        cnt = 0
        samples_names = []
        print("Processing images ...")
        for i in range(len(list_fams)):
            print('current fam: ', i)
            for index, img_file in enumerate(self._fetch_all_images(join(list_fams[i], property_alias))):
                img = image.load_img(img_file, target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                X_train[cnt] = x

                cnt += 1
                index = img_file.find(self.frame_delimiter)
                samples_names.append(img_file[0:index])
        return X_train, samples_names

    def _get_dataset_contents(self,
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
        X, samples_names = self._get_X_and_names(list_fams, num_samples, property_alias)

        os.chdir(cur_dir)

        return X, y, indexes, samples_names

    def _fetch_features(self, X: np.ndarray, model: BaseModel, output_path: str, subset) -> np.ndarray:
        """
        Used to fetch all the features predicted from a given model
        :param X: the np.ndarray containing the features
        :param model: the model to be used to predict
        :param output_path: where the predicted features will be saved
        :param subset: the subset we're working with
        :return: the np.ndarray containing all the features predicted
        """

        file_helper.guarantee_path_preconditions(output_path)

        file_path = join(output_path, subset + '.npy')
        if self._are_features_already_extracted(output_path, subset):
            print('Features already present on: ', file_path)
            features = np.load(file_path)
        else:
            print('Features not present yet, predicting now..')
            features = model.predict(X)
        return features

    def _are_features_already_extracted(self, output_path: str, subset: str) -> bool:
        """
        Used to check if the features were already extracted or not
        :param output_path: the path where the features could be
        :param subset: the subset we're working
        :return: a boolean indicating if the features already exist on disk or not
        """
        file_path = join(output_path, subset + '.npy')
        return os.path.exists(file_path)
