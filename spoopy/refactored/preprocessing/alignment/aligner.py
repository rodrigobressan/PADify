import concurrent
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List

import cv2
import imutils
import os

from refactored.preprocessing.property.property_extractor import PropertyExtractor
from tools import face_aligner
from tools.face_detector import face_detector
from tools.file_utils.file_helper import guarantee_path_preconditions
from tools.map_extractor.utils.ImageAligner import ImageAligner
from tools.map_extractor.utils.ImageCropper import ImageCropper


class Aligner():
    def __init__(self):
        self.detector, self.face_aligner = face_aligner.align_faces.make_face_aligner()

    def align_frames(self,
                     root_extracted_maps: str,
                     properties: List[PropertyExtractor],
                     output_base):

        frames_original = os.path.join(root_extracted_maps, 'original')
        frame_list = os.listdir(frames_original)

        with ThreadPoolExecutor() as exec:
            for frame_name in frame_list:
                exec.submit(self.align_single_frame, root_extracted_maps, frame_name, output_base, properties)
        # with ThreadPoolExecutor(max_workers=100) as executor:
        #     fs = [executor.submit(self.align_single_frame, root_extracted_maps, frame_name, output_base, properties) for
        #           frame_name in frame_list]
        #     concurrent.futures.wait(fs)
        #     #
            # with ProcessPoolExecutor(max_workers=100) as executor:
            #     frame_list = os.listdir(frames_original)
            #
            #     for frame_name in frame_list:
            #         executor.submit(self.align_single_frame, root_extracted_maps, frame_name, output_base, properties)

    def align_single_frame(self, base_maps_frames, frame_name, output_base, properties):

        # verify if it's not already processed
        properties_missing = []
        for property in properties:
            dir_frame_aligned = os.path.join(output_base, property.get_property_alias())
            path_frame_aligned = os.path.join(dir_frame_aligned, frame_name)

            if not os.path.exists(path_frame_aligned):
                properties_missing.append(property)
                # print('missing align for %s: ' % path_frame_aligned)

        if len(properties_missing) > 0:
            # print('Aligning %s' % frame_name)
            full_path_frame = os.path.join(base_maps_frames, 'original', frame_name)
            face_angle = face_aligner.align_faces.get_face_angle(full_path_frame, self.detector, self.face_aligner)
            image_fixed_angle = imutils.rotate(cv2.imread(full_path_frame), face_angle)

            face_cropper = face_detector.FaceCropper()
            face_coordinates = face_cropper.get_faces_coordinates(image_fixed_angle)
            cropper = ImageCropper(face_cropper, face_coordinates)
            for property in properties_missing:

                    try:
                        dir_frame_unaligned = os.path.join(base_maps_frames, property.get_property_alias())
                        dir_frame_aligned = os.path.join(output_base, property.get_property_alias())
                        path_frame_aligned = os.path.join(dir_frame_aligned, frame_name)

                        if not os.path.exists(path_frame_aligned):

                            guarantee_path_preconditions(dir_frame_aligned)
                            guarantee_path_preconditions(dir_frame_unaligned)

                            path_frame_unaligned = os.path.join(dir_frame_unaligned, frame_name)

                            aligner = ImageAligner(path_frame_unaligned, face_angle, property.get_frame_extension())
                            self.align_frame(aligner, cropper, path_frame_aligned)
                        else:
                            print('already aligned, skipping')
                    except Exception as e:
                        print('Error when trying to align frame ', e)

        # else:
        #     print('Frame %s already processed all' % frame_name)

    def align_frame(self, aligner, cropper, path_frame_aligned):
        aligned_img = aligner.align()
        cropper.crop(aligned_img, path_frame_aligned)
        # print('align done with sucess for %s' % path_frame_aligned)

