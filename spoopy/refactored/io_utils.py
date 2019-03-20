import os
import pandas


def save_txt(output_path: str, content: str):
    file = open(output_path, "w")
    file.write(str(content))
    file.close()


def load_txt(path):
    file = open(os.path.join(path), "r")
    lines = file.readlines()
    contents = pandas.io.json.loads(lines[0])
    return contents
