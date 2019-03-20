
import sys
from threading import Thread


sys.path.append('/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy')

import os

import cv2

from tools.file_utils import file_helper

def extract_rbd_saliency_folder(folder_path, output_root):
    frames = file_helper.get_frames_from_folder(folder_path)

    threads = []
    for frame in frames:
        path_frame = os.path.join(folder_path, frame)
        output_path = os.path.join(output_root, frame)
        thread_item = Thread(target=extract_rbd_saliency, args=(path_frame, output_path))
        threads.append(thread_item)
        thread_item.start()

    for thread in threads:
        thread.join()


def extract_rbd_saliency(file_path, output_path):
    print('Before extracting for frame: ', file_path)
    #rbd = saliency.get_saliency_rbd(file_path).astype('uint8')
    #cv2.imwrite(output_path, rbd)
    print('Done extracting saliency')

if __name__ == '__main__':
    extract_rbd_saliency_folder('/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/results/cbsr_2/cbsr_test/fake/24_HR_3/raw',
                                '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/results/cbsr_2/cbsr_test/fake/24_HR_3/saliency_aligned')
