import os
import sys

import cv2

from tools import file_utils


class FaceCropper(object):
    dirname = os.path.dirname(__file__)
    CASCADE_PATH = os.path.join(dirname, 'haarcascade_frontalface_alt.xml')

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def convertToRGB(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_faces_coordinates(self, img):
        # img = cv2.imread(image_path)

        if img is None:
            print("Error opening image")
            return 0

        faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        if faces is None:
            print('No faces')
            return 0

        return faces

    def crop_image_into_face(self, img, faces_coordinates, output_path, is_issue=False):

        if is_issue:
            print('stop')

        if len(faces_coordinates) > 0:
            (x, y, h, w) = faces_coordinates[len(faces_coordinates) - 1]

            padding = 0
            x = x - padding
            y = y - padding
            w = w + 2 * padding
            h = h + 2 * padding

            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            face_img = img[ny:ny + nr, nx:nx + nr]
        else:
            face_img = img

        try:
            lastimg = cv2.resize(face_img, (224, 224))
            cv2.imwrite(output_path, lastimg)
        except Exception as e:
            cv2.imwrite(output_path, img)
            print('saved original because exception: ', e)


if __name__ == '__main__':
    args = sys.argv
    argc = len(args)

    detecter = FaceCropper()

    base_path = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/static/results/test/cbsr/fake/1_3/'
    original_path = os.path.join(base_path, 'original')

    illuminated_path = os.path.join(base_path, 'illuminated')
    path_cropped = os.path.join(base_path, 'illuminated_cropped')

    file_utils.file_helper.guarantee_path_preconditions(path_cropped)

    frames = [item for item in os.listdir(original_path) if (item.endswith('.jpg') or item.endswith('.png'))]
    detected_faces = detecter.get_faces_coordinates(os.path.join(original_path, frames[0]))

    print('faces: ', detected_faces)


    detecter.crop_image_into_face(illuminated_path, detected_faces, path_cropped)
