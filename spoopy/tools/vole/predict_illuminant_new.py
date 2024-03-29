import multiprocessing as mp
import os
import time
from multiprocessing import Process
from threading import Thread

from tools import file_utils
from tools.vole import vole_segment, vole_illumination, vole_grayworld

DEFAULT_MAX_INTENSITY = 0.98823529411764705882
DEFAULT_MIN_INTENSITY = .05882352941176470588
DEFAULT_SIGMA = 0.2
DEFAULT_K = 300
DEFAULT_MIN_SIZE = 15

dirname = os.path.dirname(__file__)

VOLE_PATH = os.path.join(dirname, 'source/build/vole')
CONFIG_PATH = os.path.join(dirname, 'source/config.txt')


def predict_illuminant(item_path, output_illuminant_path):
    original_images_path = os.path.join(item_path, "raw")
    converted_images_path = os.path.join(item_path, "converted")
    segments_path = os.path.join(item_path, "segments")
    grayworld_path = os.path.join(item_path, "grayworld")

    # make sure our paths will exist
    file_utils.file_helper.guarantee_path_preconditions(converted_images_path)
    file_utils.file_helper.guarantee_path_preconditions(segments_path)
    file_utils.file_helper.guarantee_path_preconditions(grayworld_path)
    file_utils.file_helper.guarantee_path_preconditions(output_illuminant_path)

    # start vole
    vole_segment.segment_all_images(vole_path=VOLE_PATH,
                                    images_path=original_images_path,
                                    converted_path=converted_images_path,
                                    output_path_segmented=segments_path,
                                    sigma=DEFAULT_SIGMA,
                                    k=DEFAULT_K,
                                    min_size=DEFAULT_MIN_SIZE,
                                    max_intensity=DEFAULT_MAX_INTENSITY,
                                    min_intensity=DEFAULT_MIN_INTENSITY)

    vole_grayworld.extract_new_gray_world_maps(vole_path=VOLE_PATH,
                                               converted_images_path=converted_images_path,
                                               segmented_images_path=segments_path,
                                               output_gray_world_path=grayworld_path,
                                               sigma=1,
                                               n=1,
                                               p=3)

    vole_illumination.extract_illumination_maps(vole_path=VOLE_PATH,
                                                config_path=CONFIG_PATH,
                                                converted_images_path=converted_images_path,
                                                segmented_images_path=segments_path,
                                                output_illuminated_path=output_illuminant_path)


def test_thread():
    original_images_path = os.path.join(dirname, 'images')
    items = file_utils.file_helper.get_dirs_from_folder(original_images_path)

    start_time = time.time()
    threads = []
    for item in items:
        current_item_path = os.path.join(original_images_path, item)
        output_path = os.path.join(current_item_path, 'illuminated')

        thread = Thread(target=predict_illuminant, args=(current_item_path, output_path))
        threads.append(thread)
        thread.start()

    for thread in threads:  # iterates over the threads
        thread.join()  # waits until the thread has finished work

    print("--- %s seconds total with threads---" % (time.time() - start_time))


def test_process():
    print("total cpus: ", mp.cpu_count())
    original_images_path = os.path.join(dirname, 'images')
    items = file_utils.file_helper.get_dirs_from_folder(original_images_path)

    start_time = time.time()
    processes = []
    for item in items:
        current_item_path = os.path.join(original_images_path, item)
        output_path = os.path.join(current_item_path, 'illuminated')

        process = Process(target=predict_illuminant, args=(current_item_path, output_path))
        processes.append(process)
        process.start()

    for process in processes:  # iterates over the processes
        process.join()  # waits until the thread has finished work

    print("--- %s seconds total with process---" % (time.time() - start_time))


def test_without_thread():
    original_images_path = os.path.join(dirname, 'images')
    items = file_utils.file_helper.get_dirs_from_folder(original_images_path)

    start_time = time.time()
    for item in items:
        current_item_path = os.path.join(original_images_path, item)
        output_path = os.path.join(current_item_path, 'illuminated')

        predict_illuminant(current_item_path, output_path)

    print("--- %s seconds total without threads---" % (time.time() - start_time))


if __name__ == '__main__':
    test_without_thread()
