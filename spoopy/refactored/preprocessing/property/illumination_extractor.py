from refactored.preprocessing.property.property_extractor import PropertyExtractor
from tools.depth import monodepth_simple
from tools.vole import predict_illuminant


class IlluminationExtractor(PropertyExtractor):
    def extract_from_folder(self, frames_path, output_path):
        predict_illuminant.predict_illuminant(frames_path, output_path)

    def get_property_alias(self) -> str:
        return "illumination"

    def get_frame_extension(self) -> str:
        return "png"
