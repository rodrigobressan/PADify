from refactored.preprocessing.property.property_extractor import PropertyExtractor
from refactored.preprocessing.util.preprocessing_utils import get_not_processed_frames
from tools.depth import monodepth_simple


class DepthExtractor(PropertyExtractor):
    def get_frame_extension(self) -> str:
        return "jpg"

    def extract_from_folder(self, frames_path, output_path):
        missing_frames = get_not_processed_frames(frames_path, output_path)

        if len(missing_frames) > self.THRESHOLD_MISSING_FRAMES:
            monodepth_simple.apply_depth_inference_on_folder(frames_path, output_path)

    def get_property_alias(self) -> str:
        return "depth"
