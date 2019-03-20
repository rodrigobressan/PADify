import os

import cv2
import imutils


class ImageAligner:
    def __init__(self, frame_path, angle, extension="jpg"):
        self.frame_path = os.path.splitext(frame_path)[0] + "." + extension

        self.angle = angle
        self.extension = extension

    def align(self):
        img = cv2.imread(self.frame_path)
        img_rotated = imutils.rotate(img, self.angle)
        return img_rotated
