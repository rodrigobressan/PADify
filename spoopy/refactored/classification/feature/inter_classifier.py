import os
from os.path import join

from refactored.classification import features_utils
from refactored.classification.classifier import BaseClassifier
from refactored.classification.feature.feature_classifier import BasePredictor
from refactored.feature_extraction.model import BaseModel
from refactored.preprocessing.property.property_extractor import PropertyExtractor


class InterBasePredictor(BasePredictor):
    def classify_inter_dataset(self):
        for origin, target, model, prop, classifier in self._list_inter_combinations(self.features_root_path):
            self._classify_inter_dataset(dataset_origin=origin,
                                         dataset_target=target,
                                         classifier=classifier,
                                         model=model,
                                         prop=prop)

    def _list_inter_combinations(self, path: str):
        datasets = os.listdir(path)
        for dataset_origin in datasets:
            for dataset_target in [element for element in datasets if element != dataset_origin]:

                for model, prop, classifier in self._list_variations():
                    yield [dataset_origin, dataset_target, model, prop, classifier]

    def _classify_inter_dataset(self,
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
