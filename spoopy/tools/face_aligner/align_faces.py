import cv2
import dlib
import os
import wget
from imutils.face_utils import rect_to_bb

from tools.face_aligner import face_aligner_new

dirname = os.path.dirname(__file__)

SHAPE_PREDICTOR_FILE_PATH = os.path.join(dirname, 'shape_predictor_68_face_landmarks.dat')
SHAPE_PREDICTOR_FILE_URL = 'https://github.com/ageitgey/face_recognition_models/raw/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat'


def make_face_aligner(shape_predictor=SHAPE_PREDICTOR_FILE_PATH):
    if not os.path.exists(SHAPE_PREDICTOR_FILE_PATH):
        print(SHAPE_PREDICTOR_FILE_PATH)
        print('Downloading predictor file..')
        file_name = wget.download(SHAPE_PREDICTOR_FILE_URL, SHAPE_PREDICTOR_FILE_PATH)
        print(file_name)

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    fa = face_aligner_new.FaceAlignerNew(predictor, desiredFaceWidth=256)

    return detector, fa


def get_face_position(image_rotated, detector, fa):
    # image_rotated = cv2.imread(rotated_image_path)
    gray_rotated = cv2.cvtColor(image_rotated, cv2.COLOR_BGR2GRAY)

    rects_rotated = detector(gray_rotated, 2)
    return rect_to_bb(rects_rotated[0])


def get_face_angle(image_path, detector, fa):
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_path)
    print(image_path, ' readed, cvtcolor called')
    # gray_original = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    print(image_path, ' cvtcolor done')
    rects_original = detector(image, 2)
    print('detector done')
    if len(rects_original) == 0:
        print('len == 0')
        return 0

    print('before align')
    angle = fa.align(image, image, rects_original[0])
    print('after align')
    return -angle

#
# if __name__ == '__main__':
#     images_path = os.path.join(dirname, 'images')
#     frame_original = os.path.join(images_path, 'original.jpg')
#     frame_depth = os.path.join(images_path, 'depth.jpg')
#     output_frame_rotated = os.path.join(images_path, 'rotated.jpg')
#     output_frame_rotated_and_cropped = os.path.join(images_path, 'rotated_and_cropped.jpg')
#
#     detector, fa = make_face_aligner()
#
#     # get angle from original and rotate
#     angle_original = get_face_angle(frame_original, detector, fa)
#     img_rotated = imutils.rotate(cv2.imread(frame_original), angle_original)
#
#     (x, y, w, h) = get_face_position(img_rotated, detector, fa)
#
#     padding = 50
#     height = h
#     width = w
#
#     start_y = y - padding
#     end_y = start_y + height + 2 * padding
#
#     start_x = x - padding
#     end_x = start_x + width + 2 * padding
#
#     cv2.imwrite(output_frame_rotated, img_rotated)
#
#     img_depth = cv2.imread(frame_depth)
#     rotated_depth = imutils.rotate(img_depth, angle_original)
#     final_depth = rotated_depth[start_y:end_y, start_x:end_x]
#
#     cv2.imwrite(output_frame_rotated_and_cropped, rotated_depth[start_y:end_y, start_x:end_x])
