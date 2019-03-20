from abc import ABC, abstractmethod
from keras.applications import ResNet50


class BaseModel(ABC):
    """
    This class should be used to wrap any existent model predictor. Please refer to
    ResNet50Model class to see an example of how it is done
    """

    @abstractmethod
    def get_model(self):
        """
        This should return the compiled model to be used
        """
        raise NotImplementedError()

    @abstractmethod
    def get_alias(self) -> str:
        """
        This should return an alias to identify the model
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        """
        This should be used to perform the prediction on a given set of features
        :param X:
        :return:
        """
        raise NotImplementedError()


class ResNet50Model(BaseModel):
    """
    This is an example model using the ResNet50 Model
    """
    def __init__(self):
        self.model = ResNet50()

    def get_alias(self) -> str:
        return "resnet"

    def get_model(self):
        return self.model

    def predict(self, X):
        return self.model.predict(X)
