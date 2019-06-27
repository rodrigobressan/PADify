from concurrent.futures import ProcessPoolExecutor

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from refactored.preprocessing.property.property_extractor import PropertyExtractor
from refactored.preprocessing.util.preprocessing_utils import get_not_processed_frames
import subprocess
from os.path import join

class NoiseExtractor(PropertyExtractor):
    def get_frame_extension(self) -> str:
        return "bmp"

    def extract_from_folder(self, frames_path, output_path):
        missing_frames = get_not_processed_frames(frames_path, output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        noise_path = '/codes/bresan/remote/spoopy/spoopy/refactored/preprocessing/property/noise_extraction_code/bin/noise_extractor'
        for frame in missing_frames:
            output_img = join(output_path, frame + "_RF2.bmp")

            if not os.path.isfile(output_img):
                with ProcessPoolExecutor(max_workers=20) as exec:
                    output_img = join(output_path, frame)

                    exec.submit(self.extract_noise, frame, frames_path, noise_path, output_path)
        print('Extracting frames with noise')

    def extract_noise(self, frame, frames_path, noise_path, output_path):
        current_img = join(frames_path, frame)
        output_img = join(output_path, frame)
        command = noise_path + " " + current_img + " " + output_img + " RF2"
        print('command: ', command)
        os.chdir('/codes/bresan/remote/spoopy/spoopy/refactored/preprocessing/property/noise_extraction_code/bin/')
        os.system(command)
        print('done ', frame)
        return output_img

    def get_property_alias(self) -> str:
        return "noise"
