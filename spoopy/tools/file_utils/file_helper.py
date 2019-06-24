#!/usr/bin/env python

"""
This file is used to perform operations with our existent files that are related to video processing, such as splitting
a given video into a set of frames, as well as merging a set of frames into a whole video
"""
import cv2
import hashlib
# here are some basic default values for our functions, but we can definitely make it more customizable (namedtuple)
import numpy
import os
from PIL import Image
from os import path, listdir

DEFAULT_FRAME_NAME = "frame"
DEFAULT_FRAME_SEPARATOR = "_"
DEFAULT_FRAME_EXTENSION = ".jpg"


def count_files_is_same(path_1, path_2, variance=0):
    """
    This method is used to verify if the count of frames from a given directory is equals to antoher
    :param path_1: the first dir
    :param path_2: the second dir
    :return: the boolean value whether the count is the same
    """
    frames_path_1 = get_frames_from_folder(path_1)
    frames_path_2 = get_frames_from_folder(path_2)
    return abs(len(frames_path_1) - len(frames_path_2)) < variance


def get_dirs_from_folder(folder_path):
    """
    This method is used to fetch all the dirs from a folder, without fetching files
    :param folder_path: where we are looking for the dirs
    :return: a list containing all the directories
    """
    return [filename for filename in os.listdir(folder_path) if
            os.path.isdir(os.path.join(folder_path, filename))]


def get_frames_from_folder(folder):
    """
    This method is used to return all the existent frames within a given folder
    :param folder: the folder we are looking for frames
    :return: the array containing all the frames in the folder
    """
    if os.path.exists(folder):
        return [img for img in listdir(folder) if (img.endswith(DEFAULT_FRAME_EXTENSION) or img.endswith('.png') or img.endswith('.bmp'))]

    return []


def guarantee_path_preconditions(path_to_verify):
    """
    This method is used to guarantee that a given path will be existent in the disk, as well as it is going to be
    ended with a slash "/"
    :param path_to_verify: the path of the folder we want to perform the operation
    :return: the same path with any corrections, if existent
    """

    if not path_to_verify.endswith('/'):
        path_to_verify += '/'

    # guarantee that our dir will be existent
    if not path.exists(path_to_verify):
        os.makedirs(path_to_verify, exist_ok=True)

    return path_to_verify


def copy_file(original_path, final_path):
    """
    This method is used just to copy a file from a place to another
    :param original_path: where is the original file
    :param final_path: where will the file be placed
    :return:
    """

    try:
        guarantee_path_preconditions(final_path)
        os.popen('cp ' + original_path + ' ' + final_path)
    except Exception as e:
        print(e)


def copy_file_rename(original_path, final_path, final_name):
    guarantee_path_preconditions(final_path)
    final_path = os.path.join(final_path, final_name)
    os.popen('cp ' + original_path + ' ' + final_path)


def rename_file(original, final):
    """
    This method is used to rename a file
    :param original:
    :param final:
    :return:
    """
    try:
        os.rename(original, final)
    except Exception as e:
        print(e)


def split_video_into_frames(video_path, output_path, interval=1000, prefix: str = "", step_frame=1):
    """
    This method is used to split a given video into a set of frames
    :param video_path: where the video is located
    :param output_path: where the frames will be saved
    :return: a list containing the name of each of the saved frames
    """

    output_path = guarantee_path_preconditions(output_path)

    vid_capture = cv2.VideoCapture(video_path)

    success, image = vid_capture.read()
    count = 0
    success = True

    # repeat while there is something from vid_capture
    while success:

        current_frame_output = output_path + prefix + \
                               DEFAULT_FRAME_SEPARATOR + DEFAULT_FRAME_NAME + \
                               DEFAULT_FRAME_SEPARATOR + str(count) + DEFAULT_FRAME_EXTENSION

        if os.path.exists(current_frame_output):
            print(current_frame_output, " already exists!")
        else:
            # set the current position to load the image, applying the frame capture interval
            vid_capture.set(cv2.CAP_PROP_POS_MSEC, (count * interval))

            # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            cv2.imwrite(current_frame_output, image)

        success, image = vid_capture.read()
        count += step_frame

    saved_images = get_frames_from_folder(output_path)
    return saved_images


def frames_to_video(frames_path,
                    video_output_path,
                    video_name='video',
                    fps=10,
                    codec=-1,
                    extension='.avi'):
    """
    This method is used to join a set of frames into a video
    :param extension: the final extension of the video
    :param codec: the codec that will be used (remember kids, .avi is not good with browsers)
    :param fps: how many frames per second our video will have
    :param frames_path: where the frames are located (directory)
    :param video_output_path: where the video will be saved (only the directory)
    :param video_name: the name of the video artifact (just the name - default extension will be .avi)
    :return: the path where the video was saved
    """

    video_output_path = guarantee_path_preconditions(video_output_path)
    final_video_path = os.path.join(video_output_path, video_name + ".mp4")
    if os.path.isfile(final_video_path):
        return

    images = get_frames_from_folder(frames_path)
    frame = cv2.imread(path.join(frames_path, images[0]))
    height, width, layers = frame.shape

    video_output_complete = video_output_path + video_name + extension

    video = cv2.VideoWriter(video_output_complete, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))

    # simple sorting with our files, since otherwise we would have a video with messed up order of the frames
    print('images: ', images)
    images = sorted(images, key=lambda x: int(os.path.splitext(x.split('_')[1])[0]))
    for image in images:
        video.write(cv2.imread(path.join(frames_path, image)))

    cv2.destroyAllWindows()
    video.release()
    print('done: ', video_output_complete)

    command_convert = "ffmpeg -y -i " + video_output_complete + " -c:v libx264 -crf 19 -preset slow -c:a libfdk_aac -b:a 192k -ac 2 " + final_video_path
    os.system(command_convert)

    return final_video_path


def md5(filename):
    """
    This file is used to generate an md5 checksum hash for a given file name
    :param filename: the file name
    :return: the md5 digest checksum
    """
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def difference_images(path_images, channels, output_path, output_name):
    path = os.path.join(output_path, output_name + '.png')

    if os.path.isfile(path) and not FIXING_CROP:
        return

    images = get_frames_from_folder(path_images)

    # assuming all images are the same size, get dimensions of first image
    width, height = Image.open(os.path.join(path_images, images[0])).size
    number_images = len(images)

    mode = 'RGB'
    if channels == 4:
        mode = 'RGBX'

    # create a numpy array of floats to store the average (assume RGB images)
    average_array = numpy.zeros((height, width, channels), numpy.float)

    # build up average pixel intensities, casting each image as an array of floats
    for image in images:
        image_array = numpy.array(Image.open(os.path.join(path_images, image)), dtype=numpy.float)
        average_array = average_array - image_array / number_images

    # round values in array and cast as 8-bit integer
    average_array = numpy.array(numpy.round(average_array), dtype=numpy.uint8)

    # generate, save and preview final image
    out = Image.fromarray(average_array)
    print('average done success: ', path)
    out.save(path)


FIXING_CROP = True


def average_images(path_images, channels, output_path, output_name):
    """
    This method is used to generate the average image from a given set of images
    :param output_name: the name to save the file (without extension)
    :param output_path: where the file will be saved
    :param path_images: where the images are located
    :param channels: how many channels are we using in our frames (3 for rgb, 4 for rgbx)
    :return:
    """
    path = os.path.join(output_path, output_name + '.png')

    if os.path.isfile(path) and not FIXING_CROP:
        return

    images = get_frames_from_folder(path_images)

    # assuming all images are the same size, get dimensions of first image
    width, height = Image.open(os.path.join(path_images, images[0])).size
    number_images = len(images)

    mode = 'RGB'
    if channels == 4:
        mode = 'RGBX'

    # create a numpy array of floats to store the average (assume RGB images)
    average_array = numpy.zeros((height, width, channels), numpy.float)

    # build up average pixel intensities, casting each image as an array of floats
    for image in images:
        image_array = numpy.array(Image.open(os.path.join(path_images, image)), dtype=numpy.float)
        average_array = average_array + image_array / number_images

    # round values in array and cast as 8-bit integer
    average_array = numpy.array(numpy.round(average_array), dtype=numpy.uint8)

    # generate, save and preview final image
    out = Image.fromarray(average_array)
    print('average done success: ', path)
    out.save(path)


def main():
    # frames_to_video('/home/cp1500252/development/github/spoopy/spoopy/static/results/test/fake/1_3/frames/normal',
    #                 '/home/cp1500252/development/github/spoopy/spoopy/static/results/test/fake/1_3',
    #                 'output_test', 10)
    split_video_into_frames(
        '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/static/client/videos/c1000b1ecf1c736b08ca72ee8ee2d454_avi.avi',
        '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/static/client/frames/c1000b1ecf1c736b08ca72ee8ee2d454_avi/raw')

    print('File_helper main')


if __name__ == '__main__':
    main()
