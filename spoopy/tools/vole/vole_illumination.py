from concurrent.futures import ProcessPoolExecutor

import os
# Extract IlluminantMaps from all images
# IN: 		scale -- image scale folder
#		configFile -- illuminants config parameters file
# OUT: illuminant maps
from threading import Thread

from tools.file_utils import file_helper


def extract_illumination_maps(vole_path, config_path, converted_images_path, segmented_images_path,
                              output_illuminated_path):
    frames_done = file_helper.get_frames_from_folder(output_illuminated_path)
    files = []
    for i in frames_done:
        filename = os.path.basename(i)
        files.append(filename)

    threads = []

    with ProcessPoolExecutor() as exec:
        im = os.listdir(str(segmented_images_path) + "/")
        for i in im:
            exec.submit(generate_illuminant,
                            config_path, converted_images_path, files, i, output_illuminated_path,
                                  segmented_images_path, vole_path)


def generate_illuminant(config_path, converted_images_path, files, i, output_illuminated_path, segmented_images_path,
                        vole_path):
    if i not in files:
        try:
            print("Processing file %s" % (i))
            command = vole_path + " liebv " \
                                  " --img.image " + str(converted_images_path) + "/" + i + \
                      " -S " + str(segmented_images_path) + "/" + i + \
                      " -O " + str(output_illuminated_path) + "/" + i[:-4] + ".png " + \
                      "--iebv_config " + config_path

            os.system(command)
        except:
            print("Erro ao processar imagem ", i, "\n")
    else:
        print("File %s already processed!" % (i))
