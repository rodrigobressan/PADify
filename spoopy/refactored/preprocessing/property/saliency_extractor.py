import os

from refactored.preprocessing.property.property_extractor import PropertyExtractor
from refactored.preprocessing.property.saliency.saliency_extractor import extract_rbd_saliency_folder
from refactored.preprocessing.util.preprocessing_utils import get_not_processed_frames


class SaliencyExtractor(PropertyExtractor):
    def extract_from_folder(self, frames_path, output_path):
        missing_frames = len(get_not_processed_frames(frames_path, output_path))

        if missing_frames > self.THRESHOLD_MISSING_FRAMES:
            extract_rbd_saliency_folder(frames_path, output_path)

    def get_property_alias(self) -> str:
        return "saliency"

    def get_frame_extension(self) -> str:
        return "jpg"