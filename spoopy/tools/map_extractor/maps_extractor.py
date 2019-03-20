#!/usr/bin/env python
import concurrent
import gc
import sys
from concurrent.futures import ProcessPoolExecutor

import os

from tools.file_utils.file_helper import count_files_is_same
from tools.map_extractor.extractors.DepthExtractor import DepthExtractor
from tools.map_extractor.extractors.IlluminantExtractor import IlluminantExtractor

PATH_PROJECT = '/codes/bresan/remote/spoopy/'
PATH_MODULE = os.path.join(PATH_PROJECT, 'spoopy')
PATH_DATA = os.path.join(PATH_MODULE, 'data')
sys.path.append(PATH_MODULE)

import os

import cv2
import imutils

from tools import face_aligner
from tools.face_detector import face_detector
from tools.file_utils import file_helper
from tools.logger.Logger import Logger

from tools.map_extractor.extractors.NoneExtractor import NoneExtractor

from tools.map_extractor.FeatureConfig import FeatureConfig
from tools.map_extractor.extractors.SaliencyExtractor import SaliencyExtractor
from tools.map_extractor.utils.ImageAligner import ImageAligner
from tools.map_extractor.utils.ImageCropper import ImageCropper

DEFAULT_FRAMES_NORMAL_PATH = 'original'
DEFAULT_FRAMES_DEPTH_PATH = 'depth'

DEFAULT_FINAL_DEPTH_VIDEO_NAME = 'video_depth'
DEFAULT_ORIGINAL_VIDEO_NAME = 'video_original'
DEFAULT_EXTENSION = '.mp4'

logs_path = os.path.join(PATH_MODULE, 'logs')
l = Logger(logs_path)


def run_pipeline(dataset_alias, pai, dataset_type, dataset_root, data_type, output_path):
    l.log("Run pipeline (run_pipeline()) method invocation. Parameters below:")
    l.log("    Fetching items from dataset")
    l.log("    Dataset root: " + dataset_root)
    l.log("    Dataset alias: " + dataset_alias)
    l.log("    PAI: " + pai)
    l.log("    Data type: " + data_type)

    detector, fa = face_aligner.align_faces.make_face_aligner()
    items_dataset = file_helper.get_dirs_from_folder(dataset_root)

    for index, item in enumerate(items_dataset):
        try:
            if is_item_processed(output_path, dataset_alias, pai, dataset_type, dataset_root, data_type, item):
                print("Item % already processed" % item)
            else:
                print("Processing %s" % item)

                process_item(output_path, dataset_alias, pai, dataset_type, dataset_root,
                             data_type, item, detector, fa)
        except Exception as exception:
            l.logE(exception)


def get_feature_configs(result_path):
    configs = []

    raw_frames_path = os.path.join(result_path, "raw")

    config_none = FeatureConfig(raw_frames_path, result_path, NoneExtractor(), "jpg", "original")
    config_depth = FeatureConfig(raw_frames_path, result_path, DepthExtractor(), "jpg", "depth")
    config_illuminant = FeatureConfig(raw_frames_path, result_path, IlluminantExtractor(), "png", "illuminant")
    config_saliency = FeatureConfig(raw_frames_path, result_path, SaliencyExtractor(), "jpg", "saliency")

    configs.append(config_none)
    # configs.append(config_depth)
    # configs.append(config_illuminant)
    configs.append(config_saliency)

    return configs


def is_item_processed(output_path, dataset_alias, pai, dataset_type, dataset_root, data_type, item):
    result_path = os.path.join(output_path, pai, dataset_alias, dataset_type, data_type, item)
    original_items_path = os.path.join(dataset_root, item)

    feature_configs = get_feature_configs(result_path)
    print('result path: ', result_path)
    return are_all_features_processed(feature_configs, original_items_path)


def are_all_features_processed(feature_configs, original_items_path):
    for config in feature_configs:
        if not config.is_raw_config and not os.path.exists(config.results_aligned_path):
            return False

        if not os.path.exists(config.results_unaligned_path):
            return False

        if not count_files_is_same(config.results_aligned_path, original_items_path):
            print("Count files not the same!!!")
            return False

        if not count_files_is_same(config.results_unaligned_path, original_items_path):
            print("Count files not the same!!!")
            return False

    return True


def process_item(output_path, dataset_alias, pai, dataset_type, dataset_root, data_type, item, detector, fa):
    # path like: /static/results/cbsr/train/fake/
    result_path = os.path.join(output_path, dataset_alias, pai, dataset_type, data_type)

    # path where the new data will be stored on
    raw_frames_path = os.path.join(result_path, "raw")  # backup images on results folder

    # first of all, we gotta make a backup of all our frames, so we can work on them and not directly with the dataset
    original_items_path = os.path.join(dataset_root, item)
    make_copy_frames(original_items_path, raw_frames_path)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # now we're going to iterate over all of our feature maps
        feature_configs = get_feature_configs(result_path)
        for config in feature_configs:

            if config.feature_alias:
                l.log('Extracting feature: ' + config.feature_alias)
                file_helper.guarantee_path_preconditions(config.results_unaligned_path)
                file_helper.guarantee_path_preconditions(config.results_aligned_path)
                executor.submit(perform_extraction, config.origin_path, config.results_unaligned_path, config.extractor)

        frames_original = file_helper.get_frames_from_folder(raw_frames_path)

        # and now align all the frames
        try:
            with ProcessPoolExecutor() as executor:

                for current_frame_name in frames_original:
                    if is_aligned_for_all_extractors(current_frame_name, feature_configs):
                        continue

                    executor.submit(align_single_frame, raw_frames_path, current_frame_name, detector, fa,
                                    feature_configs)
                    # align_single_frame(raw_frames_path, current_frame_name, detector, fa, feature_configs)
                gc.collect()
        except Exception as exception:
            l.logE(exception)


def is_aligned_for_all_extractors(frame_name, configs):
    for config in configs:
        final_aligned_path = os.path.join(config.results_aligned_path, frame_name)
        if not os.path.exists(final_aligned_path):
            return False

    return True


def make_copy_frames(origin_frames_path, copy_frames_path):
    all_original_frames = file_helper.get_frames_from_folder(origin_frames_path)
    file_helper.guarantee_path_preconditions(copy_frames_path)
    if not file_helper.count_files_is_same(origin_frames_path, copy_frames_path):
        for original_frame in all_original_frames:
            original_frame_path = os.path.join(origin_frames_path, original_frame)
            original_img = cv2.imread(original_frame_path)
            output_path = os.path.join(copy_frames_path, original_frame)
            cv2.imwrite(output_path, original_img)
            # file_helper.copy_file(, copy_frames_path)

        l.log("Copy images successfully")
    else:
        l.log("Images were already copied")


def align_single_frame(path_raw_frames, current_frame_name, detector, fa, feature_configs):
    original_frame = os.path.join(path_raw_frames, current_frame_name)
    original_angle = face_aligner.align_faces.get_face_angle(original_frame, detector, fa)
    original_rotated = imutils.rotate(cv2.imread(original_frame), original_angle)

    fc = face_detector.FaceCropper()
    coordinates = fc.get_faces_coordinates(original_rotated)

    print('coordinates none: ', coordinates is None)
    cropper = ImageCropper(fc, coordinates)

    for config in feature_configs:
        final_aligned_path = os.path.join(config.results_aligned_path, current_frame_name)

        if not os.path.isfile(final_aligned_path) and config.feature_alias:
            original_path_frame = os.path.join(config.results_unaligned_path, current_frame_name)
            aligner = ImageAligner(original_path_frame, original_angle, config.extension)
            aligned_img = aligner.align()
            cropper.crop(aligned_img, final_aligned_path)


def perform_extraction(frames_path, output_path, extractor):
    extractor.extract(frames_path, output_path)


def extract_maps_from_dataset(frames_root_path, base_output_path):
    with ProcessPoolExecutor() as executor:

        pai_list = os.listdir(frames_root_path)

        # Presentation Attack Instrument (Tablet, Print, Cut)
        for pai in pai_list:
            pai_path = os.path.join(frames_root_path, pai)
            subsets_list = os.listdir(pai_path)

            # Subset/split (Train, Test, Enrollment)
            for subset in subsets_list:
                subset_path = os.path.join(pai_path, subset)
                labels_list = os.listdir(subset_path)

                # Label of the category (Real, Attack)
                for label in labels_list:
                    labels_path = os.path.join(subset_path, label)
                    # videos_list = os.listdir(labels_path)
                    print(labels_path)

                    run_pipeline('cbsr', pai, subset, subset_path, label, base_output_path)


def extract_maps_from_dir(frames_base_path, output_path):
    for current_dataset in os.listdir(frames_base_path):
        print("Running on %s" % current_dataset)

        current_dataset_path = os.path.join(frames_base_path, current_dataset)
        # set_type = [train, test, ..]
        for set_type in os.listdir(current_dataset_path):
            set_path = os.path.join(current_dataset_path, set_type)
            # target_type = [fake, real]
            for target_type in os.listdir(set_path):
                target_path = os.path.join(set_path, target_type)  # root
                run_pipeline(current_dataset, set_type, target_path, target_type, output_path)


if __name__ == '__main__':
    frames = '/codes/bresan/remote/spoopy/spoopy/data/extracted_frames/csbr'
    output = '/codes/bresan/remote/spoopy/spoopy/data/extracted_maps'
    # extract_maps_from_dataset(frames, output)
    extract_maps_from_dataset(frames, output)
