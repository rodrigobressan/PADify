from tools.vole import predict_illuminant


class IlluminantExtractor:
    def __init__(self):
        self.name = "illuminant"

    def extract(self, frames_path, output_path):
        predict_illuminant.predict_illuminant(frames_path, output_path)
        print("Illuminant done")
