import pickle

import numpy as np
import os
from os.path import join, exists

from refactored.classification import features_utils
from refactored.classification.classifier import BaseClassifier
from refactored.classification.feature.feature_predictor import BasePredictor
from refactored.feature_extraction.cnn_model import CnnModel
from refactored.io_utils import save_txt, load_txt
from tools.file_utils import file_helper


class MetalearnerClassifier(BasePredictor):
    """
    Steps:

        TRAINING
            FIRST CLASSIFIER
                For each feature (depth, illumination, saliency, original), using the training set we gotta fit and predict on
                the same subset. The output of this operation should be the vector of probabilities for each property map.
            SECOND CLASSIFIER
                After this, we need to concatenate these generated probabilities vectors and then train our second classifier
        TESTING
            FIRST CLASSIFIER
                For each feature, using the already trained First Classifier, we predict on the test set. The output of
                this operation should be the vector of probabilities for each property map.
            SECOND CLASSIFIER
                After this, we need to concatenate the generated probabilities vectors outputted from the prediction of
                the test features on the first classifier and then concatenate then. These new vector will be later on
                be used as test set on the second classifier.
    """

    def _train_inter_probas_classifier(self) -> None:
        """
        STEP 1.2
        Used to train the second classifier (probas classifier) and output the final results
        """
        probas_path = os.path.join(self.meta_dataset_output, self.INTER_NAME, "train", "features")
        # [CBSR, RA, NUAA]
        for dataset in os.listdir(probas_path):
            # path_dataset = os.path.join(self.features_root_path, dataset, self.target_all)

            for classifier in self.classifiers:

                # [ResNet, VGG...]
                for model in self.models:
                    # TODO change to iterate properties

                    probas_original = self.__load_train_probas(dataset, "original", model, classifier)
                    probas_depth = self.__load_train_probas(dataset, "depth", model, classifier)
                    probas_illumination = self.__load_train_probas(dataset, "illumination", model, classifier)
                    probas_saliency = self.__load_train_probas(dataset, "saliency", model, classifier)

                    stacked_probas = np.stack((probas_depth, probas_illumination, probas_saliency, probas_original),
                                              axis=2)

                    labels_original = self.__load_train_labels(dataset, "original", model, classifier)
                    fitted_classifier = self._fit(classifier, stacked_probas, labels_original)

                    output_dir = join(self.meta_dataset_output,
                                      self.INTER_NAME,
                                      "train",
                                      "probas",
                                      dataset,
                                      model.alias,
                                      classifier.get_alias())

                    file_helper.guarantee_path_preconditions(output_dir)
                    model_path = os.path.join(output_dir, self.MODEL_NAME)
                    pickle.dump(fitted_classifier, open(model_path, 'wb'))
                    print('stacked done!')

    def __load_test_probas(self,
                           dataset_origin: str,
                           dataset_target: str,
                           property_alias: str,
                           model: CnnModel,
                           classifier: BaseClassifier) -> np.ndarray:
        """
        Used to load the probabilities predictions previously generated from the first classifier
        :param dataset_origin: the origin dataset we're looking to load the probabilities
        :param dataset_origin: the target dataset we're looking to load the probabilities
        :param property_alias: the property (depth, illum, saliency, etc) we are loading
        :param model: the model (ResNet50, VGG, etc)
        :param classifier: used classifier (SVC, XGB, SVM, CNN, etc)
        :return:
        """
        path_probas = join(self.meta_dataset_output, self.INTER_NAME, "test", "features", dataset_origin,
                           dataset_target,
                           property_alias,
                           model.alias,
                           classifier.get_alias(), 'y_pred_proba.npy')

        probas = np.load(path_probas)
        probas = np.reshape(probas[:, 0], (probas.shape[0], -1))
        return probas

    def __load_train_probas(self,
                            dataset_name: str,
                            property_alias: str,
                            model: CnnModel,
                            classifier: BaseClassifier) -> np.ndarray:
        """
        Used to load the probabilities predictions previously generated from the first classifier
        :param dataset_name: the dataset we're looking to load the probabilities
        :param property_alias: the property (depth, illum, saliency, etc) we are loading
        :param model: the model (ResNet50, VGG, etc)
        :param classifier: used classifier (SVC, XGB, SVM, CNN, etc)
        :return:
        """
        path_probas = join(self.meta_dataset_output, self.INTER_NAME, "train", "features", dataset_name, property_alias,
                           model.alias,
                           classifier.get_alias(), 'y_pred_proba.npy')

        probas = np.load(path_probas)
        probas = np.reshape(probas[:, 0], (probas.shape[0], -1))
        return probas

    def __load_test_labels(self, dataset_origin: str, dataset_target: str, property_alias: str, model: CnnModel,
                           classifier: BaseClassifier) -> np.ndarray:

        path_labels = join(self.meta_dataset_output, self.INTER_NAME, "test", "features", dataset_origin,
                           dataset_target,
                           property_alias,
                           model.alias,
                           classifier.get_alias(), 'labels.npy')

        return np.load(path_labels)

    def __load_train_labels(self, dataset_name: str, property_alias: str, model: CnnModel,
                            classifier: BaseClassifier) -> np.ndarray:

        path_labels = join(self.meta_dataset_output, self.INTER_NAME, "train", "features", dataset_name, property_alias,
                           model.alias,
                           classifier.get_alias(), 'labels.npy')

        return np.load(path_labels)

    def _train_inter_feature_classifier(self) -> None:
        """
        STEP 1.1
        Used to train the first classifier (feature classifier) and generate the probabilities for each property map.
        """

        # [CBSR, RA, NUAA]
        for dataset in os.listdir(self.features_root_path):
            # [ResNet, VGG, MobileNet, etc]
            for model in self.models:
                base_path = join(self.features_root_path, dataset, self.target_all)
                # [Depth, Illum, Saliency]
                for prop in self.properties:
                    property_path = join(base_path, prop.get_property_alias(), model.alias)
                    features_concatenated = features_utils.concatenate_features(property_path)
                    names_concatenated = features_utils.concatenate_names(property_path)
                    labels_concatenated = features_utils.concatenate_labels(property_path)

                    for classifier in self.classifiers:
                        output_dir = join(self.meta_dataset_output,
                                          self.INTER_NAME,
                                          "train",
                                          "features",
                                          dataset,
                                          prop.get_property_alias(),
                                          model.alias,
                                          classifier.get_alias())

                        if exists(output_dir):
                            print('Already generated, skipping.')
                            continue

                        y_pred, y_proba = self._fit_and_predict(classifier, features_concatenated, labels_concatenated,
                                                                features_concatenated)
                        results = self._evaluate_results(y_pred, labels_concatenated, names_concatenated)

                        print('HTER: %f\nAPCER: %f\nBPCER: %f' % (results[0], results[1], results[2]))

                        self._save_artifacts(classifier, output_dir, y_pred, y_proba, results)

                        save_txt(join(output_dir, 'names.txt'), names_concatenated)
                        np.save(join(output_dir, 'labels.npy'), labels_concatenated)

    def _test_inter_features_classifier(self) -> None:
        """
        STEP 2.1
        Used to predict on the test set with the already trained first classifier
        """

        for dataset_origin in os.listdir(self.features_root_path):
            for dataset_target in os.listdir(self.features_root_path):

                if dataset_origin == dataset_target:
                    print('Origin and target are the same. Skipping.')
                    continue

                # [ResNet, VGG]
                for model in self.models:
                    base_path_target = join(self.features_root_path, dataset_target, self.target_all)

                    for prop in self.properties:

                        for classifier in self.classifiers:
                            print('origin %s target %s model %s prop %s classifier %s' % (dataset_origin,
                                                                                          dataset_target,
                                                                                          model.alias,
                                                                                          prop.get_property_alias(),
                                                                                          classifier.get_alias()))
                            classifier_path = join(self.meta_dataset_output, self.INTER_NAME, "train", "features",
                                                   dataset_origin, prop.get_property_alias(),
                                                   model.alias, classifier.get_alias(), 'model.sav')

                            with open(classifier_path, 'rb') as f:
                                model_fitted = pickle.load(f)

                            path_features = join(base_path_target, prop.get_property_alias(), model.alias)

                            features_concatenated = features_utils.concatenate_features(path_features)
                            names_concatenated = features_utils.concatenate_names(path_features)
                            labels_concatenated = features_utils.concatenate_labels(path_features)

                            y_pred, y_pred_proba = self._predict(model_fitted, features_concatenated)

                            results = self._evaluate_results(y_pred, labels_concatenated, names_concatenated)
                            print('HTER: %f\nAPCER: %f\nBPCER: %f' % (results[0], results[1], results[2]))

                            output_dir = join(self.meta_dataset_output,
                                              self.INTER_NAME,
                                              "test",
                                              "features",
                                              dataset_origin,
                                              dataset_target,
                                              prop.get_property_alias(),
                                              model.alias,
                                              classifier.get_alias())

                            self._save_artifacts(classifier, output_dir, y_pred, y_pred_proba, results)
                            #
                            save_txt(join(output_dir, 'names.txt'), names_concatenated)
                            np.save(join(output_dir, 'labels.npy'), labels_concatenated)

    def _test_inter_probas_classifier(self):
        features_path = os.path.join(self.meta_dataset_output, self.INTER_NAME, "test", "features")
        # [CBSR, RA, NUAA]
        for dataset_origin in os.listdir(features_path):
            for dataset_target in os.listdir(features_path):

                if dataset_target == dataset_origin:
                    print('Origin and target are the same, skipping.')
                    continue

                # path_dataset = os.path.join(self.features_root_path, dataset, self.target_all)

                for classifier in self.classifiers:

                    # [ResNet, VGG...]
                    for model in self.models:
                        base_path = join(self.meta_dataset_output, self.INTER_NAME, "test", "features", dataset_origin,
                                         dataset_target)
                        probas_original = self.__load_test_probas(dataset_origin, dataset_target, "original", model,
                                                                  classifier)
                        probas_depth = self.__load_test_probas(dataset_origin, dataset_target, "depth", model,
                                                               classifier)
                        probas_illumination = self.__load_test_probas(dataset_origin, dataset_target, "illumination",
                                                                      model, classifier)
                        probas_saliency = self.__load_test_probas(dataset_origin, dataset_target, "saliency", model,
                                                                  classifier)

                        stacked_probas = np.stack((probas_depth, probas_illumination, probas_saliency, probas_original),
                                                  axis=2)

                        labels = self.__load_test_labels(dataset_origin, dataset_target, "original", model, classifier)
                        names = load_txt(
                            join(base_path, "original", model.alias, classifier.get_alias(), 'names.txt'))

                        classifier_path = join(self.meta_dataset_output, self.INTER_NAME, "train", "probas",
                                               dataset_origin, model.alias, classifier.get_alias(), 'model.sav')

                        with open(classifier_path, 'rb') as f:
                            model_fitted = pickle.load(f)

                        stacked_probas = np.reshape(stacked_probas, (stacked_probas.shape[0], -1))

                        y_pred, y_pred_proba = self._predict(model_fitted, stacked_probas)

                        results = self._evaluate_results(y_pred, labels, names)
                        print('HTER: %f\nAPCER: %f\nBPCER: %f' % (results[0], results[1], results[2]))
                        #
                        #     output_dir = join(self.meta_dataset_output,
                        #                       self.INTER_NAME,
                        #                       "test",
                        #                       "features",
                        #                       dataset_origin,
                        #                       dataset_target,
                        #                       prop.get_property_alias(),
                        #                       model.alias,
                        #                       classifier.get_alias())
                        #
                        #     self._save_artifacts(classifier, output_dir, y_pred, y_pred_proba, results)
                        #     #
                        #     save_txt(join(output_dir, 'names.txt'), names_concatenated)
                        #     np.save(join(output_dir, 'labels.npy'), labels_concatenated)
                        #
                        # print('stacked done!')

    def _perform_meta_classification(self):
        # self._train_inter_feature_classifier()
        # self._train_inter_probas_classifier()
        # self._test_inter_features_classifier()
        self._test_inter_probas_classifier()

    def classify_all_probas(self):
        for origin, target, model, prop, classifier in self._list_datasets(self.features_root_path):
            self._classify_probas(dataset_origin=origin,
                                  dataset_target=target,
                                  classifier=classifier,
                                  model=model,
                                  prop=prop)

    def _list_datasets(self, path: str):
        datasets = os.listdir(path)
        for dataset_origin in datasets:
            for model, prop, classifier in self._list_variations():
                yield [dataset_origin, model, prop, classifier]