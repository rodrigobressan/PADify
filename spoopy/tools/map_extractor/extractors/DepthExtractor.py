from tools.depth import monodepth_simple


class DepthExtractor:
    def __init__(self):
        self.name = "depth"

    def extract(self, frames_path, output_path):
        print('depth...')
        monodepth_simple.apply_depth_inference_on_folder(frames_path, output_path)
