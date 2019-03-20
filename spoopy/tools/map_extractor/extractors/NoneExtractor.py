

class NoneExtractor:
    def __init__(self):
        self.name = "raw"

    def extract(self, frames_path, output_path):
        from tools.map_extractor import make_copy_frames
        make_copy_frames(frames_path, output_path)

