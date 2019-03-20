from concurrent.futures import ProcessPoolExecutor

import os

from tools.file_utils import file_helper

DEFAULT_MAX_INTENSITY = 0.98823529411764705882
DEFAULT_MIN_INTENSITY = .05882352941176470588
DEFAULT_SIGMA = 0.2
DEFAULT_K = 300
DEFAULT_MIN_SIZE = 15


def segment_all_images(vole_path, images_path, converted_path, output_path_segmented, sigma, k, min_size, max_intensity,
                       min_intensity):
    # command = "rm ../data-base/segmented/*.png"
    # os.system(command)

    already_existent_files = file_helper.get_frames_from_folder(output_path_segmented)
    im = os.listdir(images_path)

    with ProcessPoolExecutor() as exec:
        for current_image in im:
            if current_image.replace('jpg', 'png') not in already_existent_files:
                exec.submit(generate_segments,
                            converted_path, current_image, images_path, k, max_intensity, min_intensity, min_size,
                            output_path_segmented, sigma, vole_path)

            else:
                print('Segment already existent!')
                # for thread in threads:
                #     thread.join()


def generate_segments(converted_path, current_image, images_path, k, max_intensity, min_intensity, min_size,
                      output_path_segmented, sigma, vole_path):
    try:
        print(current_image)
        current_name = current_image.split(".")
        new_name = current_image

        # check file extension
        if current_name[1] != "png":
            cmd = "convert " + str(images_path) + "/" + current_image + " " + str(converted_path) + "/" + \
                  current_name[0] + ".png"

            os.system(cmd)
            new_name = current_name[0] + ".png"

        command = vole_path + " felzenszwalb " \
                              " -I " + str(converted_path) + "/" + new_name + \
                  " --deterministic_coloring " \
                  " -O " + str(output_path_segmented) + "/" + new_name + \
                  " --sigma " + str(sigma) + \
                  " --k " + str(k) + \
                  " --min_size " + str(min_size) + \
                  " --max_intensity " + str(max_intensity) + \
                  " --min_intensity " + str(min_intensity)
        print('vole: ', command)

        os.system(command)
    except:
        print("Erro ao processar imagem \n")
