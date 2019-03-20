from tools.saliency_extractor import saliency

class SaliencyExtractor:
    def __init__(self):
        self.name = "saliency"

    def extract(self, frames_path, output_path):
        saliency.extract_rbd_saliency_folder(frames_path, output_path)
        print("Saliency done")