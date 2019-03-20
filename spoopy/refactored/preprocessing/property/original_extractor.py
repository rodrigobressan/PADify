from concurrent.futures import ProcessPoolExecutor

import os

from refactored.preprocessing.property.property_extractor import PropertyExtractor
from refactored.preprocessing.util.preprocessing_utils import get_not_processed_frames
from tools import file_utils


class OriginalExtractor(PropertyExtractor):
    def extract_from_folder(self, frames_path, output_path):
        frames = os.listdir(frames_path)

        missing_frames = len(get_not_processed_frames(frames_path, output_path))

        if missing_frames > self.THRESHOLD_MISSING_FRAMES:
            with ProcessPoolExecutor() as exec:
                for frame_name in frames:
                    path_frame = os.path.join(frames_path, frame_name)

                    if not os.path.exists(os.path.join(output_path, frame_name)):
                        exec.submit(file_utils.file_helper.copy_file, path_frame, output_path)

    def get_property_alias(self) -> str:
        return "original"

    def get_frame_extension(self) -> str:
        return "jpg"
