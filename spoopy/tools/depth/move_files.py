import argparse
import os
import shutil

import tensorflow as tf

parser = argparse.ArgumentParser(description='Move new files.')
parser.add_argument('--folder', type=str, help='folder path', required=True)

args = parser.parse_args()


def move_files(folder):
    print("Picked folder: " + folder)

    all_dirs = os.listdir(folder)

    for current_dir in all_dirs:
        if (os.path.isdir(folder + current_dir)):
            print(" dir: " + current_dir)
            all_files = os.listdir(folder + current_dir)
            for current_file in all_files:
                current_file_path = folder + current_dir + "/" + current_file
                new_file_path = folder + "/" + current_file
                print("current_file_path " + current_file_path)
                print("new_file_path " + new_file_path)

                shutil.move(current_file_path, new_file_path)
            # print("  file: " + current_file)
        else:
            print("not dir: " + current_dir)


def remove_files_without_disp(folder):
    all_files = os.listdir(folder)

    for current_file in all_files:
        normal_file_path = folder + current_file
        disp_file_path = folder + os.path.splitext(current_file)[0] + "_disp" + os.path.splitext(current_file)[1]
        print("disp_file_path: " + disp_file_path)
        if (not os.path.exists(disp_file_path)):
            os.remove(normal_file_path)


def main(_):
    remove_files_without_disp(args.folder)


if __name__ == '__main__':
    tf.app.run()
