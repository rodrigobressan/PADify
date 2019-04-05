import os
import json
import pandas
def save_txt(output_path: str, content: str):
    file = open(output_path, "w")
    file.write(str(content))
    file.close()


def load_txt(path):
    file = open(path, "r")
    lines = file.readlines()
    contents = json.loads(lines[0].replace('\'', '"'))
    return contents
