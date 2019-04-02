import numpy as np
import os
from os.path import join, exists

from refactored.classification import features_utils
from refactored.classification.classifier import BaseClassifier
from refactored.classification.feature.feature_classifier import BasePredictor
from refactored.feature_extraction.model import BaseModel
from refactored.preprocessing.property.property_extractor import PropertyExtractor


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
        probas_path = os.path.join(self.meta_dataset_output, self.INTER_NAME)
        # [CBSR, RA, NUAA]
        for dataset in os.listdir(probas_path):
            # path_dataset = os.path.join(self.features_root_path, dataset, self.target_all)

            for classsifier in self.classifiers:

                # [ResNet, VGG...]
                for model in self.models:
                    # TODO change to iterate properties

                    probas_original = self.__load_probas(dataset, "original", model, classsifier)
                    probas_depth = self.__load_probas(dataset, "depth", model, classsifier)
                    probas_illumination = self.__load_probas(dataset, "illumination", model, classsifier)
                    probas_saliency = self.__load_probas(dataset, "saliency", model, classsifier)

                    stacked_probas = np.stack((probas_depth, probas_illumination, probas_saliency, probas_original),
                                              axis=0)

                    self._classify(classsifier, stacked_probas, )
                    print('stacked done!')

    def __load_probas(self, dataset_name: str, property_alias: str, model: BaseModel,
                      classifier: BaseClassifier) -> np.ndarray:
        """
        Used to load the probabilities predictions previously generated from the first classifier
        :param dataset_name: the dataset we're looking to load the probabilities
        :param property_alias: the property (depth, illum, saliency, etc) we are loading
        :param model: the model (ResNet50, VGG, etc)
        :param classifier: used classifier (SVC, XGB, SVM, CNN, etc)
        :return:
        """
        path_probas = join(self.meta_dataset_output, self.INTER_NAME, dataset_name, property_alias, model.get_alias(),
                           classifier.get_alias(), 'y_pred_proba.npy')

        return np.load(path_probas)

    def __load_labels(self, dataset_name: str, property_alias: str, model: BaseModel,
                      classifier: BaseClassifier) -> np.ndarray:

        pass

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
                    property_path = join(base_path, prop.get_property_alias(), model.get_alias())
                    features_concatenated = features_utils.concatenate_features(property_path)
                    names_concatenated = features_utils.concatenate_names(property_path)
                    labels_concatenated = features_utils.concatenate_labels(property_path)

                    for classifier in self.classifiers:
                        output_dir = join(self.meta_dataset_output,
                                          self.INTER_NAME,
                                          dataset,
                                          prop.get_property_alias(),
                                          model.get_alias(),
                                          classifier.get_alias())

                        if exists(output_dir):
                            print('Already generated, skipping.')
                            # continue

                        y_pred, y_proba = self._classify(classifier, features_concatenated, labels_concatenated,
                                                         features_concatenated)
                        results = self._evaluate_results(y_pred, labels_concatenated, names_concatenated)

                        print('HTER: %f\nAPCER: %f\nBPCER: %f' % (results[0], results[1], results[2]))

                        self._save_artifacts(classifier, output_dir, y_pred, y_proba, results)
                        np.save(join(output_dir, 'names.npy'), names_concatenated)
                        np.save(join(output_dir, 'labels.npy'), labels_concatenated)

    def _perform_meta_classification(self):
        self._train_inter_feature_classifier()
        # self._train_inter_probas_classifier()

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

    def _classify_probas(self,
                         dataset_origin: str,
                         dataset_target: str,
                         classifier: BaseClassifier,
                         model: BaseModel,
                         prop: PropertyExtractor):

        origin_path = join(self.features_root_path, dataset_origin, self.target_all, prop.get_property_alias(),
                           model.get_alias())

        target_path = join(self.features_root_path, dataset_target, self.target_all, prop.get_property_alias(),
                           model.get_alias())

        X_train = features_utils.concatenate_features(origin_path)
        X_test = features_utils.concatenate_features(target_path)

        y_train = features_utils.concatenate_labels(origin_path)
        y_test = features_utils.concatenate_labels(target_path)

        names_train = features_utils.concatenate_names(origin_path)
        names_test = features_utils.concatenate_names(target_path)

        y_pred, y_proba = self._classify(classifier, X_train, y_train, X_test)
        results = self._evaluate_results(y_pred, y_test, names_test)

        print('HTER: %f\nAPCER: %f\nBPCER: %f' % (results[0], results[1], results[2]))

        output_dir = join(self.inter_dataset_output,
                          dataset_origin,
                          dataset_target,
                          self.target_all,
                          prop.get_property_alias(),
                          model.get_alias(),
                          classifier.get_alias())

        self._save_artifacts(classifier, output_dir, y_pred, y_proba, results)
