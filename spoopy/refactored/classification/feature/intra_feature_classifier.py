from concurrent.futures import ProcessPoolExecutor

import numpy as np
import os
from os.path import join

from refactored.classification.classifier import BaseClassifier
from refactored.classification.feature.feature_predictor import BasePredictor
from refactored.feature_extraction.feature_extraction import NAME_FEATURES, NAME_TARGETS, NAME_SAMPLES
from refactored.feature_extraction.cnn_model import CnnModel
from refactored.io_utils import load_txt
from refactored.preprocessing.property.property_extractor import PropertyExtractor


class IntraBasePredictor(BasePredictor):
    def classify_intra_dataset(self):
        datasets = os.listdir(self.features_root_path)
        for dataset in datasets:
            for model, prop, classifier in self._list_variations():
                self._classify_intra_dataset(dataset=dataset,
                                             classifier=classifier,
                                             model=model,
                                             prop=prop)

    def _load_features_and_targets(self, base_path):
        X_train = np.load(os.path.join(base_path, (NAME_FEATURES % self.train_alias)))
        X_test = np.load(os.path.join(base_path, (NAME_FEATURES % self.test_alias)))

        y_train = np.load(os.path.join(base_path, (NAME_TARGETS % self.train_alias)))
        y_test = np.load(os.path.join(base_path, (NAME_TARGETS % self.test_alias)))

        names_test = load_txt(os.path.join(base_path, (NAME_SAMPLES % self.test_alias)))
        return X_train, y_train, X_test, y_test, names_test

    def _classify_intra_dataset(self, dataset: str,
                                classifier: BaseClassifier,
                                model: CnnModel,
                                prop: PropertyExtractor):

        path_features = join(self.features_root_path, dataset, self.target_all, prop.get_property_alias(),
                             model.alias)

        output_dir = os.path.join(self.intra_dataset_output, dataset,
                                  self.target_all,
                                  prop.get_property_alias(),
                                  model.alias,
                                  classifier.get_alias())

        print('features: ', path_features)
        print('output: ', output_dir)

        if os.path.exists(output_dir):
            print('Already processed, skipping.')
            return

        X_train, y_train, X_test, y_test, names_test = self._load_features_and_targets(path_features)

        y_pred, y_pred_proba = self._fit_and_predict(classifier, X_train, y_train, X_test)
        results = self._evaluate_results(y_pred, y_test, names_test)
        print('results:', results)
        self._save_artifacts(classifier, output_dir, y_test, y_pred, y_pred_proba, results)
