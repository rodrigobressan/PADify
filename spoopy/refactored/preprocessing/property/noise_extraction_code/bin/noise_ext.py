import os
from os.path import join


def extract_noise(frames, frames_path, output_path):
    noise_path = './noise_extractor'
    for frame in frames:
        current_img = join(frames_path, frame)
        output_img = join(output_path, frame)
        command = noise_path + " " + current_img + " " + output_img + " RF2"
        print('command: ', command)
        os.system(command)
