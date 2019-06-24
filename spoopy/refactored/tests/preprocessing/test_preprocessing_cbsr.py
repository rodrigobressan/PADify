
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib

from refactored.classification.finetuning.intra_finetuning_classifier import IntraFinetuningClassifier

matplotlib.use('Agg')

import pathlib
import time
import unittest

import shutil
from os.path import exists, join

from refactored.classification.feature.inter_feature_classifier import InterBasePredictor
from refactored.classification.finetuning.inter_finetuning_classifier import InterFinetuningClassifier
from refactored.classification.metalearner.metalearner_classifier import MetalearnerClassifier


from refactored.classification.classifier import XGBoostClassifier
from refactored.classification.feature.intra_feature_classifier import IntraBasePredictor
from refactored.feature_extraction.feature_extraction import FeatureExtractor
from refactored.preprocessing import common_preprocessing


class TestPreprocessingCbsr(unittest.TestCase):
    base_path_artifacts = '/codes/bresan/remote/spoopy/spoopy/refactored/tests/artifacts_bkp'
    output_features = os.path.join(base_path_artifacts, 'features')
    output_classification = os.path.join(base_path_artifacts, 'classification')
    output_finetuning = os.path.join(base_path_artifacts, 'finetuning')
    output_separated = os.path.join(base_path_artifacts, 'separated_by_pai')

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
        self.models = common_preprocessing.get_models()
        self.classifiers = [XGBoostClassifier()]
        self.processor = common_preprocessing.make_cbsr_processor(self.base_path_artifacts)

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
        # self.processor.organize_properties_by_pai()
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
                                        model.alias)

                    for artifact in expected_artifacts:
                        path_artifact = os.path.join(path, artifact)
                        print(path_artifact)
                        self.assertTrue(exists(path_artifact))

    def print_stats(self, frames_extracted, frames_maps):
        for prop in self.processor.properties:
            prop = prop.get_property_alias()
            size = len(frames_maps[prop])
            less_perc = (100 * size) / len(frames_extracted)
            print('%s: %d perc: %d' % (prop, size, less_perc))

    def analyze_extracted_data(self):

        print('Analyzing extracted frames')
        path_extracted = self.processor.extracted_frames_root
        extracted_frames = self.processor.handler.get_frames_structured(path_extracted)

        frames_extracted = []
        for frame, label, subset in extracted_frames:
            name = '%s_%s_%s' % (frame, label, subset)
            frames_extracted.append(name)

        print('Size extracted frames: %d' % len(frames_extracted))
        ############################### 20194 + 6427 + 13824 + 4406
        print('Analyzing extracted maps')
        path_maps = self.processor.properties_root
        frames_maps = self.extract_dicts_props(path_maps)
        self.print_stats(frames_extracted, frames_maps)
        ##################################
        print('Analyzing aligned maps')
        path_aligned = self.processor.aligned_root
        frames_aligned = self.extract_dicts_props(path_aligned)
        self.print_stats(frames_extracted, frames_aligned)

        print('Printing missing')

        for prop in self.processor.properties:
            prop = prop.get_property_alias()
            diff = list(set(frames_extracted).symmetric_difference(frames_aligned[prop]))
        ##################################

        path_separated = self.processor.separated_pai_root
        pais = os.listdir(path_separated)

        for pai in pais:
            pai_path = join(path_separated, pai)
            for subset in os.listdir(pai_path):
                subset_path = join(pai_path, subset)
                for label in os.listdir(subset_path):
                    labels_path = join(subset_path, label)
                    for prop in os.listdir(labels_path):
                        prop_path = join(labels_path, prop)
                        frames = len(os.listdir(prop_path))
                        print('%s %s %s %s %d' % (pai, subset, label, prop, frames))



    def extract_dicts_props(self, path_maps):
        extracted_maps = self.processor.handler.get_frames_properties(path_maps)
        frames_maps = {}
        for frame, prop, label, subset in extracted_maps:
            name = '%s_%s_%s' % (frame, label, subset)

            if prop not in frames_maps:
                frames_maps[prop] = []
                frames_maps[prop].append(name)
            else:
                frames_maps[prop].append(name)
        return frames_maps

    def perform_intra_feature_classification(self):
        feature_classifier = IntraBasePredictor(features_root_path=self.output_features,
                                                base_output_path=self.output_classification,
                                                classifiers=self.classifiers,
                                                properties=self.processor.properties,
                                                models=self.models)

        feature_classifier.classify_intra_dataset()

    def perform_inter_feature_classification(self):
        feature_classifier = InterBasePredictor(features_root_path=self.output_features,
                                                base_output_path=self.output_classification,
                                                classifiers=self.classifiers,
                                                properties=self.processor.properties,
                                                models=self.models)

        feature_classifier.classify_inter_dataset()

    def perform_metalearning_classification(self):
        metalearner_classifier = MetalearnerClassifier(features_root_path=self.output_features,
                                                       base_output_path=self.output_classification,
                                                       classifiers=self.classifiers,
                                                       properties=self.processor.properties,
                                                       models=self.models)

        metalearner_classifier._perform_meta_classification()

    def perform_inter_finetuning(self):
        finetuner = InterFinetuningClassifier(images_root_path=self.output_separated,
                                              base_output_path=self.output_finetuning,
                                              models=self.models,
                                              properties=self.processor.properties)

        finetuner.classify_inter_dataset()

    def perform_intra_finetuning(self):
        finetuner = IntraFinetuningClassifier(images_root_path=self.output_separated,
                                              base_output_path=self.output_finetuning,
                                              models=self.models,
                                              properties=self.processor.properties)

        finetuner.classify_intra_dataset()

    def separate_frames_finetuning(self):
        self.processor.separate_for_intra_finetuning()
        self.processor.separate_for_inter_finetuning()

    def test_preprocessor(self):
        tasks = [
            # self.organize_videos_by_subset_and_label,
            # self.extract_frames_from_videos,
            # self.extract_maps_from_frames,
            # self.align_maps,
            # self.separate_maps_by_pai,
            # self.analyze_extracted_data,
            # self.extract_features,
            # self.separate_frames_finetuning,
            # self.perform_intra_feature_classification,
            # self.perform_inter_feature_classification,
            # self.perform_metalearning_classification,
            # self.perform_inter_finetuning
            self.perform_intra_finetuning
        ]

        for task in tasks:
            try:
                task()
            except Exception as e:
                self.fail(e)

        print('All done!')
