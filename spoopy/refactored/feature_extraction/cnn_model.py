from typing import Tuple

from abc import ABC
from keras.applications import ResNet50
import tensorflow as tf


class CnnModel(ABC):
    """
    This class should be used to wrap any existent model predictor. Please refer to
    ResNet50Model class to see an example of how it is used
    """

    DEFAULT_CHANNELS = 3
    DEFAULT_WIDTH = 224
    DEFAULT_WEIGHT = 224

    def __init__(self, model,
                 alias: str,
                 input_shape: Tuple[int, int, int] = [DEFAULT_CHANNELS, DEFAULT_WIDTH, DEFAULT_WEIGHT]):
        self.__model = model
        self.alias = alias
        self.input_shape = input_shape

    def get_model(self):
        with tf.device('/cpu:0'):
            return self.__model()

    def predict(self, X):
        """
        This should be used to perform the prediction on a given set of features
        :param X:
        :return:
        """
        print('Predicting with model: %s' % self.alias)
        return self.get_model().predict(X)


class ResNet50Model(CnnModel):
    def __init__(self):
        super().__init__(ResNet50, "resnet50", [3, 224, 224])
#
#
# class DenseNetModel(CnnModel):
#     def __init__(self):
#         super().__init__(DenseNet121, "densenet121")
#
#
# class InceptionV3Model(CnnModel):
#     def __init__(self):
#         super().__init__(InceptionV3, "inceptionv3")
#
#
# class MobileNetModel(CnnModel):
#     def __init__(self):
#         super().__init__(MobileNet, "mobilenet")
#
#
# class MobileNetV2Model(CnnModel):
#     def __init__(self):
#         super().__init__(MobileNetV2, "mobilenetv2")
#
#
# class NasNetMobileModel(CnnModel):
#     def __init__(self):
#         super().__init__(NASNetMobile, "nasnetmobile")
#
#
# class NasNetLargeModel(CnnModel):
#     def __init__(self):
#         super().__init__(NASNetLarge, "nasnetlarge")
#
#
# class Vgg16Model(CnnModel):
#     def __init__(self):
#         super().__init__(VGG16, "vgg16")
#
#
# class Vgg19Model(CnnModel):
#     def __init__(self):
#         super().__init__(VGG19, "vgg19")
#
#
# class XceptionModel(CnnModel):
#     def __init__(self):
#         super().__init__(Xception, "xception")
#
#
# class InceptionResnetV2Model(CnnModel):
#     def __init__(self):
#         super().__init__(InceptionResNetV2, "inceptionresnetv2")
