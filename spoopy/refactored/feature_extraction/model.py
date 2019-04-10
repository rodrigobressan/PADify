from abc import ABC
from keras.applications import ResNet50, InceptionV3, MobileNet, NASNetMobile, NASNetLarge, VGG16, VGG19, Xception, \
    InceptionResNetV2, MobileNetV2
from keras_applications.densenet import DenseNet, DenseNet121


class BaseModel(ABC):
    """
    This class should be used to wrap any existent model predictor. Please refer to
    ResNet50Model class to see an example of how it is done
    """

    def __init__(self, model, alias: str):
        self.model = model
        self.alias = alias

    def get_model(self):
        """
        This should return the compiled model to be used
        """
        return self.model

    def get_alias(self) -> str:
        """
        This should return an alias to identify the model
        :return:
        """
        return self.alias

    def predict(self, X):
        """
        This should be used to perform the prediction on a given set of features
        :param X:
        :return:
        """
        print('Predicting with model: %s' % self.alias)
        return self.model.predict(X)


class ResNet50Model(BaseModel):
    def __init__(self):
        super().__init__(ResNet50(), "resnet50")


class DenseNetModel(BaseModel):
    def __init__(self):
        super().__init__(DenseNet121(), "densenet121")


class InceptionV3Model(BaseModel):
    def __init__(self):
        super().__init__(InceptionV3(), "inceptionv3")


class MobileNetModel(BaseModel):
    def __init__(self):
        super().__init__(MobileNet(), "mobilenet")


class MobileNetV2Model(BaseModel):
    def __init__(self):
        super().__init__(MobileNetV2(), "mobilenetv2")


class NasNetMobileModel(BaseModel):
    def __init__(self):
        super().__init__(NASNetMobile(), "nasnetmobile")


class NasNetLargeModel(BaseModel):
    def __init__(self):
        super().__init__(NASNetLarge(), "nasnetlarge")


class Vgg16Model(BaseModel):
    def __init__(self):
        super().__init__(VGG16(), "vgg16")


class Vgg19Model(BaseModel):
    def __init__(self):
        super().__init__(VGG19(), "vgg19")


class XceptionModel(BaseModel):
    def __init__(self):
        super().__init__(Xception(), "xception")


class InceptionResnetV2Model(BaseModel):
    def __init__(self):
        super().__init__(InceptionResNetV2(), "inceptionresnetv2")
