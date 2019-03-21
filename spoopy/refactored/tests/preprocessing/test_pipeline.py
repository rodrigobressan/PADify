import pathlib
import time
import unittest

import os
import shutil
from os.path import exists, join

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from refactored.classification.classifier import SvcClassifier
from refactored.classification.feature.feature_classifier import FeatureClassifier
from refactored.classification.feature.inter_classifier import InterFeatureClassifier
from refactored.classification.feature.intra_classifier import IntraFeatureClassifier
from refactored.feature_extraction.feature_extraction import FeatureExtractor
from refactored.feature_extraction.model import ResNet50Model
from refactored.preprocessing import preprocess


class TestIntegrationPipeline(unittest.TestCase):
    base_path_artifacts = '../artifacts_bkp'
    output_features = os.path.join(base_path_artifacts, 'features')
    output_classification = os.path.join(base_path_artifacts, 'classification')

    artifacts_being_tested = ['test/attack/%s1_3%s.%s',
                              'test/real/%s1_1%s.%s',
                              'train/attack/%s2_8%s.%s',
                              'train/real/%s2_1%s.%s']

    artifacts_separated_pai = ['all/test/attack/%s/1_3_frame_0.%s',
                               'all/test/real/%s/1_1_frame_0.%s',
                               'all/train/attack/%s/2_8_frame_0.%s',
                               'all/train/real/%s/2_1_frame_1.%s']

    def tearDown(self):
        print("Performing cleanup")
        cleanup = False

        if cleanup:
            shutil.rmtree(pathlib.Path(self.processor.separated_pai_root).parent)
            shutil.rmtree(pathlib.Path(self.processor.extracted_frames_root).parent)
            shutil.rmtree(pathlib.Path(self.processor.properties_root).parent)
            shutil.rmtree(pathlib.Path(self.processor.aligned_root).parent)
            shutil.rmtree(pathlib.Path(self.output_features))
        print("Cleanup done")

    def setUp(self):
        self.models = [ResNet50Model()]
        self.classifiers = [SvcClassifier()]
        self.processor = preprocess.make_cbsr_processor(self.base_path_artifacts)

    def organize_videos_by_subset_and_label(self):
        self.processor.organize_videos_by_subset_and_label()
        root = self.processor.separated_subset_root
        ext = 'avi'

        time.sleep(0.5)  # just in case it's still performing copy..

        for artifact in self.artifacts_being_tested:
            path = os.path.join(root, (artifact % ('', '', ext)))
            print(path)
            self.assertTrue(exists(path))

    def extract_frames_from_videos(self):
        self.processor.extract_frames_from_videos()
        root = self.processor.extracted_frames_root
        ext = 'jpg'
        for artifact in self.artifacts_being_tested:
            path = os.path.join(root, (artifact % ('', '_frame_0', ext)))
            self.assertTrue(exists(path))

    def extract_maps_from_frames(self):
        self.processor.extract_maps_from_frames()

        root = self.processor.properties_root

        for map in self.processor.properties:
            prop = map.get_property_alias() + "/"
            ext = map.get_frame_extension()

            for artifact in self.artifacts_being_tested:
                path = os.path.join(root, (artifact % (prop, '_frame_0', ext)))
                print(path)
                self.assertTrue(exists(path))

    def align_maps(self):
        self.processor.align_maps()

        root = self.processor.aligned_root
        ext = 'jpg'

        # for map in self.processor.properties:
        #     prop = map.get_property_alias() + "/"
        #
        #     for artifact in self.artifacts_being_tested:
        #         path = os.path.join(root, (artifact % (prop, '_frame_2', ext)))
        #         print(path)
        #         self.assertTrue(exists(path))

    def separate_maps_by_pai(self):
        self.processor.organize_properties_by_pai()

        path_pai = self.processor.separated_pai_root
        for artifact in self.artifacts_separated_pai:
            for prop in self.processor.properties:
                path = join(path_pai, (artifact % (prop.get_property_alias(), prop.get_frame_extension())))
                self.assertTrue(exists(path))

    def extract_features(self):
        separated_path = os.path.join(self.base_path_artifacts, 'separated_by_pai')

        extractor = FeatureExtractor(separated_path=separated_path,
                                     output_features=self.output_features,
                                     models=self.models,
                                     properties=self.processor.properties)

        extractor.perform_extraction()

        target_list = os.listdir(os.path.join(separated_path, 'cbsr'))

        expected_artifacts = ['X_train.npy',
                              'y_train.npy',
                              'y_test.npy',
                              'samples_train.txt',
                              'samples_test.txt']

        # TODO check if the number of features is equal to the number of frames

        for prop in self.processor.properties:
            for target in target_list:
                for model in self.models:
                    path = os.path.join(self.output_features,
                                        'cbsr',
                                        target,
                                        prop.get_property_alias(),
                                        model.get_alias())

                    for artifact in expected_artifacts:
                        path_artifact = os.path.join(path, artifact)
                        print(path_artifact)
                        self.assertTrue(exists(path_artifact))

    def perform_intra_feature_classification(self):
        feature_classifier = IntraFeatureClassifier(features_root_path=self.output_features,
                                                    base_output_path=self.output_classification,
                                                    classifiers=self.classifiers,
                                                    properties=self.processor.properties,
                                                    models=self.models)

        feature_classifier.classify_intra_dataset()
        self.evaluate_intra_classification(feature_classifier)

    def perform_inter_feature_classification(self):
        feature_classifier = InterFeatureClassifier(features_root_path=self.output_features,
                                                    base_output_path=self.output_classification,
                                                    classifiers=self.classifiers,
                                                    properties=self.processor.properties,
                                                    models=self.models)

        feature_classifier.classify_inter_dataset()
        # self.evaluate_intra_classification(feature_classifier)

    def evaluate_intra_classification(self, feature_classifier: FeatureClassifier):
        datasets = os.listdir(join(self.output_classification, feature_classifier.INTRA_NAME))

        expected_artifacts = ['model.sav',
                              'results.txt',
                              'y_pred.npy',
                              'y_pred_proba.npy']

        for dataset in datasets:
            for prop in self.processor.properties:
                for model in self.models:
                    for classifier in self.classifiers:
                        base_path = os.path.join(feature_classifier.intra_dataset_output,
                                                 dataset,
                                                 feature_classifier.target_all,
                                                 prop.get_property_alias(),
                                                 model.get_alias(),
                                                 classifier.get_alias())

                        for artifact in expected_artifacts:
                            path_artifact = os.path.join(base_path, artifact)
                            self.assertTrue(exists(path_artifact))

    def evaluate_inter_classification(self, feature_classifier: FeatureClassifier):
        datasets = os.listdir(join(self.output_classification, feature_classifier.INTER_NAME))

        expected_artifacts = ['model.sav',
                              'results.txt',
                              'y_pred.npy',
                              'y_pred_proba.npy']

        for dataset_origin in datasets:
            for dataset_target in datasets:
                for prop in self.processor.properties:
                    for model in self.models:
                        for classifier in self.classifiers:
                            base_path = os.path.join(feature_classifier.intra_dataset_output,
                                                     dataset_origin,
                                                     dataset_target,
                                                     feature_classifier.target_all,
                                                     prop.get_property_alias(),
                                                     model.get_alias(),
                                                     classifier.get_alias())

                            for artifact in expected_artifacts:
                                path_artifact = os.path.join(base_path, artifact)
                                self.assertTrue(exists(path_artifact))

    # def test_preprocessor(self):
    #     tasks = [
    #         # self.organize_videos_by_subset_and_label,
    #         # self.extract_frames_from_videos,
    #         # self.extract_maps_from_frames,
    #         # self.align_maps,
    #         self.separate_maps_by_pai,
    #         # self.extract_features,
    #         # self.perform_intra_feature_classification,
    #         # self.perform_inter_feature_classification
    #     ]
    #
    #     for task in tasks:
    #         try:
    #             task()
    #         except Exception as e:
    #             self.fail(e)
    #
    #     print('All done!')
    def test_sample(self):
        self.assertEqual(2, 1 + 1)
