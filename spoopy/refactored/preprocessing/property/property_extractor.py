from abc import ABC, abstractmethod


class PropertyExtractor(ABC):

    THRESHOLD_MISSING_FRAMES = 50
    """
    This class should be used as a base class for any property extractor. Please rely to other
    extractors (such as DepthExtractor, IlluminationExtractor, SaliencyExtractor) as a working
    example.
    """

    @abstractmethod
    def extract_from_folder(self, frames_path: str, output_path: str):
        """
        This method should be responsible to generate the properties map for a given folder which
        contains a list of frames
        :param frames_path: the folder containing the frames
        :param output_path: where they will be stored
        """
        raise NotImplementedError("You should override the method extract_from_folder")

    @abstractmethod
    def get_property_alias(self) -> str:
        """
        This method should return the alias of the property, for identification purposes.
        :return: the alias of the property (as a string)
        """
        raise NotImplementedError("You should override the method get_property_alias")

    @abstractmethod
    def get_frame_extension(self) -> str:
        """
        This method should return the extension from the generated property extension
        :return: the extension (as a string)
        """
        raise NotImplementedError("You should override the method get_frame_extension")
