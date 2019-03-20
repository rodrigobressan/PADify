import os

from tools import file_utils, depth, bob, data, vole, face_detector, face_aligner, logger, map_extractor, \
    saliency_extractor, classifier, classifier_probas, feature_extractor

dirname = os.path.dirname(__file__)


def get_root():
    return dirname
