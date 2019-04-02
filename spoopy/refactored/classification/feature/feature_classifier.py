import pickle
from typing import List, Tuple

import numpy as np
import os

from refactored.classification.classifier import BaseClassifier
from refactored.feature_extraction.model import BaseModel
from refactored.io_utils import save_txt
from refactored.preprocessing.handler.datahandler import DataHandler, DiskHandler
from refactored.preprocessing.property.property_extractor import PropertyExtractor
from tools.classifier import evaluate_hter
from tools.file_utils import file_helper


class BasePredictor:
    PRED_NAME = "y_pred.npy"
    PROBA_NAME = "y_pred_proba.npy"
    MODEL_NAME = "model.sav"
    RESULTS_NAME = "results.txt"

    INTRA_NAME = "intra"
    INTER_NAME = "inter"
    META_NAME = "meta"

    def __init__(self,
                 features_root_path: str,
                 base_output_path: str,
                 classifiers: List[BaseClassifier],
                 properties: List[PropertyExtractor],
                 models: List[BaseModel],
                 data_handler: DataHandler = DiskHandler(),
                 train_alias: str = 'train',
                 test_alias: str = 'test',
                 target_all: str = 'all'):

        self.features_root_path = features_root_path
        self.base_output_path = base_output_path
        self.classifiers = classifiers
        self.properties = properties
        self.models = models
        self.data_handler = data_handler
        self.train_alias = train_alias
        self.test_alias = test_alias
        self.target_all = target_all

        self.intra_dataset_output = os.path.join(self.base_output_path, self.INTRA_NAME)
        self.inter_dataset_output = os.path.join(self.base_output_path, self.INTER_NAME)
        self.meta_dataset_output = os.path.join(self.base_output_path, self.META_NAME)

    def _list_variations(self):
        # classifiers = [svc, svm, etc]
        for classifier in self.classifiers:

            # prop = [depth, illum, etc]
            for prop in self.properties:

                # models = [resnet, vgg, etc]
                for model in self.models:
                    yield [model, prop, classifier]

    def _save_artifacts(self, classifier: BaseClassifier,
                        output_dir: str,
                        y_pred: np.ndarray,
                        y_pred_proba: np.ndarray,
                        results: np.ndarray):

        file_helper.guarantee_path_preconditions(output_dir)

        # save preds
        np.save(os.path.join(output_dir, self.PRED_NAME), y_pred)
        np.save(os.path.join(output_dir, self.PROBA_NAME), y_pred_proba)

        # save fitted model
        model_path = os.path.join(output_dir, self.MODEL_NAME)
        pickle.dump(classifier, open(model_path, 'wb'))

        # save HTER, APCER and BPCER
        results_path = os.path.join(output_dir, self.RESULTS_NAME)
        result = '%d\n%d\n%d' % (results[0], results[1], results[2])

        save_txt(results_path, result)

    def _classify(self, classifier: BaseClassifier,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)

        return y_pred, y_pred_proba

    def _evaluate_results(self, y_pred, y_test, names_test) -> Tuple[float, float, float]:
        hter, apcer, bpcer = evaluate_hter.evaluate_with_values(y_pred, y_test, names_test)
        return hter, apcer, bpcer
