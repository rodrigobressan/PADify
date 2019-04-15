from refactored.feature_extraction.cnn_model import *
from refactored.preprocessing.processor.cbsr.cbsr_processor import CbsrProcessor
from refactored.preprocessing.processor.deep_fakes.deep_fakes_processor import DeepFakesProcessor
from refactored.preprocessing.processor.nuaa.nuaa_processor import NuaaProcessor
from refactored.preprocessing.processor.replay_attack.ra_processor import RaProcessor
from refactored.preprocessing.property.depth_extractor import DepthExtractor
from refactored.preprocessing.property.illumination_extractor import IlluminationExtractor
from refactored.preprocessing.property.original_extractor import OriginalExtractor
from refactored.preprocessing.property.saliency_extractor import SaliencyExtractor


def get_properties():
    properties = []

    saliency_extractor = SaliencyExtractor()
    illumination_extractor = IlluminationExtractor()
    depth_extractor = DepthExtractor()
    original_extractor = OriginalExtractor()

    properties.append(depth_extractor)
    properties.append(original_extractor)
    properties.append(saliency_extractor)
    properties.append(illumination_extractor)

    return properties


def get_models():
    models = [
        ResNet50Model(),
        # DenseNetModel(),
        # InceptionV3Model(),
        # MobileNetModel(),
        # MobileNetV2Model(),
        # NasNetMobileModel(),
        # NasNetLargeModel(),
        # Vgg16Model(),
        # Vgg19Model(),
        # XceptionModel(),
        # InceptionResnetV2Model()§
    ]
    return models


def make_cbsr_processor(artifacts_root):
    processor = CbsrProcessor(artifacts_root=artifacts_root,
                              dataset_name='cbsr',
                              properties=get_properties())
    return processor


def make_ra_processor(artifacts_root):
    processor = RaProcessor(artifacts_root=artifacts_root,
                            dataset_name='ra',
                            properties=get_properties())
    return processor


def make_deep_fakes_processor(artifacts_root):
    processor = DeepFakesProcessor(artifacts_root=artifacts_root,
                                   dataset_name='deep',
                                   properties=get_properties())
    return processor


def make_nuaa_processor(artifacts_root):
    processor = NuaaProcessor(artifacts_root=artifacts_root,
                                   dataset_name='nuaa',
                                   properties=get_properties())
    return processor
