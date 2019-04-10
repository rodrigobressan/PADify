import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from refactored.classification.classifier import SvcClassifier
from refactored.feature_extraction.model import ResNet50Model
from refactored.preprocessing import common_preprocessing

base_path_artifacts = 'tests/artifacts_bkp'
output_features = os.path.join(base_path_artifacts, 'features')
output_classification = os.path.join(base_path_artifacts, 'classification')

models = [ResNet50Model()]
classifiers = [SvcClassifier()]
processor = common_preprocessing.make_cbsr_processor(base_path_artifacts)

processor.organize_properties_by_pai()
# import os
# dirs = os.listdir('/codes/bresan/remote/spoopy/spoopy/refactored/tests/artifacts_bkp/aligned/cbsr/train/real/original')
# print(len(dirs))