import os

from PIL import Image, ImageChops

dirname = os.path.dirname(__file__)


def generate_difference(img1_path, img2_path, result_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    diff = ImageChops.difference(img1, img2)

    diff.save(result_path)


if __name__ == '__main__':
    images_path = os.path.join(dirname, 'images')
    frame_1 = os.path.join(images_path, '1_frame_1.jpg')
    frame_2 = os.path.join(images_path, 'frame_1.jpg')
    output_difference = os.path.join(images_path, 'difference.jpg')

    generate_difference(frame_1, frame_2, output_difference)
