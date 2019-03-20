import os


class FeatureConfig:
    def __init__(self, origin_path, results_base_path, extractor, extension, feature_alias=None):
        self.origin_path = origin_path

        if feature_alias:
            self.is_raw_config = False
            self.results_unaligned_path = os.path.join(results_base_path, feature_alias + "_raw")
            self.results_aligned_path = os.path.join(results_base_path, feature_alias + "_aligned_cropped")
        else:
            self.is_raw_config = True
            self.results_unaligned_path = os.path.join(results_base_path, "raw")
            self.results_aligned_path = os.path.join(results_base_path, "raw_aligned_cropped")


        self.extractor = extractor
        self.extension = extension
        self.feature_alias = feature_alias