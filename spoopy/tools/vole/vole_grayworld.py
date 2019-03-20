from concurrent.futures import ProcessPoolExecutor

import os
# Extract GrayWorldMaps from all images
# IN: 		scale -- image scale folder
#		sigma -- gray-world parameters
#	            n -- gray-world parameters
#	     	    p -- gray-world parameters
# OUT: gray-world maps
from threading import Thread

from tools.file_utils import file_helper


def extract_new_gray_world_maps(vole_path, converted_images_path, segmented_images_path, output_gray_world_path, sigma,
                                n, p):
    frames_done = file_helper.get_frames_from_folder(output_gray_world_path)
    files = []
    for i in frames_done:
        filename = os.path.basename(i)
        files.append(filename)

    with ProcessPoolExecutor() as exec:
        im = os.listdir(str(segmented_images_path) + "/")
        for i in im:
            exec.submit(generate_grayworld,
            converted_images_path, files, i, n, output_gray_world_path, p, segmented_images_path, sigma,
            vole_path)


def generate_grayworld(converted_images_path, files, i, n, output_gray_world_path, p, segmented_images_path, sigma,
                       vole_path):
    if i not in files:
        try:
            print("Processing file %s" % (i))
            command = vole_path + " lgrayworld " \
                                  "--img.image " + str(converted_images_path) + "/" + i + \
                      " -S " + str(segmented_images_path) + "/" + i + \
                      " -O " + str(output_gray_world_path) + "/" + i[:-4] + ".png " + \
                      "--n " + str(n) + \
                      " --p " + str(p) + \
                      " --sigma " + str(sigma)
            os.system(command)
        except:
            print("Erro ao processar imagem ", i, "\n")
    else:
        print("File %s already processed!" % (i))
