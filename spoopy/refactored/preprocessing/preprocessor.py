from concurrent.futures import ProcessPoolExecutor
from typing import List

import os
from abc import ABC, abstractmethod
from os.path import join

from refactored.preprocessing.Video import Video
from refactored.preprocessing.alignment.aligner import Aligner
from refactored.preprocessing.handler.datahandler import DataHandler, DiskHandler
from refactored.preprocessing.property.property_extractor import PropertyExtractor
from tools.file_utils.file_helper import split_video_into_frames


class Preprocessor(ABC):
    DEFAULT_ALIAS_ALL_DATA = 'all'

    PATH_ORIGINAL_VIDEOS = 'original_videos'
    PATH_SEPARATED_BY_PAI = 'separated_by_pai'
    PATH_SEPARATED_BY_SUBSET = 'separated_by_subset'
    PATH_EXTRACTED_FRAMES = 'extracted_frames'
    PATH_EXTRACTED_MAPS = 'extracted_maps'
    PATH_ALIGNED = 'aligned'

    THRESHOLD_MISSING_FRAMES = 15

    def __init__(self,
                 artifacts_root: str,
                 dataset_name: str,
                 properties: List[PropertyExtractor],
                 handler: DataHandler = DiskHandler(),
                 attack_label: str = 'attack',
                 real_label: str = 'real'):

        self.videos_root = join(artifacts_root, self.PATH_ORIGINAL_VIDEOS, dataset_name)
        self.separated_pai_root = join(artifacts_root, self.PATH_SEPARATED_BY_PAI, dataset_name)
        self.separated_subset_root = join(artifacts_root, self.PATH_SEPARATED_BY_SUBSET, dataset_name)
        self.extracted_frames_root = join(artifacts_root, self.PATH_EXTRACTED_FRAMES, dataset_name)
        self.properties_root = join(artifacts_root, self.PATH_EXTRACTED_MAPS, dataset_name)
        self.aligned_root = join(artifacts_root, self.PATH_ALIGNED, dataset_name)

        self.properties = properties
        self.handler = handler

        self.videos = []
        self.aligner = Aligner()

        self.default_attack_label = attack_label
        self.default_real_label = real_label

    @abstractmethod
    def organize_videos_by_subset_and_label(self):
        """
        This method should be overridden in order to organise the dataset files in the following structure:

            Subset [Test, Train]
                Label [Real, Attack]
                    Video1.mp4
                    Video2.mp4
                    Video3.mp4

        """
        raise NotImplementedError("You should override the method organize_videos_by_subset_and_label")

    @abstractmethod
    def organize_properties_by_pai(self) -> None:
        """
        This method should be overridden in order to organise the extracted properties maps in the following structure:

            Subset [Test, Train]
                Attack Alias [Print, Tablet, Mask]
                    Attack
                        Frame1.jpg
                    Real
                        Frame1.jpg

            E.g.:

            Train
                Tablet
                    Attack
                        Frame1.jpg
                        Frame2.jpg
                    Real
                        Frame1.jpg
                        Frame2.jpg
                Mask
                    Attack
                        Frame1.jpg
                        Frame2.jpg
                    Real
                        Frame1.jpg
                        Frame2.jpg
            Test
                ... Similar to the above ...
        :return: None
        """
        raise NotImplementedError("You should override the method organize_properties_by_pai")

    def move_video_to_proper_dir(self, video: Video) -> None:
        """
        You can use this method as auxiliary inside the organize_videos_by_subset_label method.

        This method pretty much makes a copy of the video into the proper directory
        :param video: the video to be copied
        :return:
        """
        video_path_all = join(self.separated_subset_root,
                              video.subset,
                              video.get_label())

        self.handler.move_video(video, video_path_all)

        self.videos.append(video)

    def extract_frames_from_videos(self) -> None:
        """
        This method is used to extract the frames from the videos
        """

        with ProcessPoolExecutor() as executor:
            for video_name, label, subset in self.handler.list_videos(path=self.separated_subset_root):

                if label == self.default_attack_label:
                    is_attack = True
                else:
                    is_attack = False

                video_path = os.path.join(self.separated_subset_root, subset, label, video_name)
                video = Video(path=video_path,
                              name=video_name,
                              subset=subset,
                              is_attack=is_attack,
                              person=self.get_person_from_video_name(video_name))

                # output format: <frames_root>/<subset>/<label> e.g: /data/cbsr/frames/train/real
                output_path = join(self.extracted_frames_root,
                                   video.subset,
                                   video.get_label())

                executor.submit(split_video_into_frames,
                                video_path=video.path,
                                output_path=output_path,
                                interval=100,
                                prefix=video.get_frame_prefix())

    def align_maps(self) -> None:
        """
        Used to align the maps accordingly to the face of the subjects
        """
        for label, subset in self.handler.get_subsets_and_labels(self.properties_root):
            output_alignments = join(self.aligned_root,
                                     subset,
                                     label)
            property_original_root = os.path.join(self.properties_root,
                                                  subset,
                                                  label)

            self.aligner.align_frames(property_original_root, self.properties, output_alignments)

    def extract_maps_from_frames(self) -> None:
        """
        Used to extract the property maps from the frames.
        """
        for label, subset in self.handler.get_subsets_and_labels(self.extracted_frames_root):

            # format: /<root>/<subset>/<label> e.g.: /data/train/attack
            labels_path = os.path.join(self.extracted_frames_root, subset, label)

            with ProcessPoolExecutor() as executor:
                for prop in self.properties:

                    # format: /<root>/<subset>/<label>/<prop> e.g.: /data/train/attack/depth
                    output_property = join(self.properties_root,
                                           subset,
                                           label,
                                           prop.get_property_alias())

                    if not self._is_already_processed(output_property, labels_path):
                        print(
                            'Property %s subset %s label %s not yet processed' % (prop.get_property_alias(),
                                                                                  subset,
                                                                                  label))
                        prop.extract_from_folder(labels_path, output_property)
                    else:
                        print(
                            'Property %s subset %s label %s already processed' % (prop.get_property_alias(),
                                                                                  subset,
                                                                                  label))
        # clean up afterwards any frame with issue
        frames_issues = self.get_frames_with_issues()
        self.remove_properties_frames_with_issues(frames_issues)

    def _is_already_processed(self, output_path: str, base_path: str) -> bool:
        """
        Used to check if a given dataset was already processed or not
        :param output_path: where the processed data will be stored
        :param base_path: where the data is located
        :return: a boolean indicating if the dataset was processed or not
        """
        if not os.path.exists(output_path):
            return False

        missing_frames = len(os.listdir(base_path)) - len(os.listdir(output_path))
        return missing_frames < self.THRESHOLD_MISSING_FRAMES

    def get_frames_with_issues(self) -> List:
        """
         Used to locate the frames that had some issues during the property extraction step. An example of issue can be
         a frame which at least one of the properties was not properly extracted, thus leading to a missing frame for
         that given property.
        :return: the list containing all the frames with issues
        """
        frames_with_issues = []
        for frame_name, label, subset in self.handler.get_frames_structured(self.extracted_frames_root):
            for prop in self.properties:
                frame_name_no_ext = frame_name.split('.')[0]
                frame_name_property_ext = frame_name_no_ext + "." + prop.get_frame_extension()
                frame_property = os.path.join(self.properties_root,
                                              subset,
                                              label,
                                              prop.get_property_alias(),
                                              frame_name_property_ext)

                if not os.path.exists(frame_property):
                    frames_with_issues.append([frame_name, label, subset])
        return frames_with_issues

    @abstractmethod
    def get_person_from_video_name(self, name: str) -> str:
        """
        Abstract method used to obtain the person/subject name from a given video. This method is declared as abstract
        so each custom processor needs to be implemented accordingly to the dataset naming structure.
        :param name: the name from the video without any parsing
        :return: a string containing the subject/person name
        """
        raise NotImplementedError("You should override the method get_person_from_video_name")

    def remove_properties_frames_with_issues(self, frames_issues: List) -> None:
        """
        Used to remove frames that had some issues during the preprocessing. An example of issue can be a frame which
        at least one of the properties was not properly extracted, thus leading to a missing frame for that given property.
        In this case we proceed to remove all the generated properties for that given frame.

        :param frames_issues: the list of frames containing the issues
        """
        for frame_name, label, subset in frames_issues:
            for prop in self.properties:
                frame_name_no_ext = frame_name.split('.')[0]
                frame_name_property_ext = frame_name_no_ext + "." + prop.get_frame_extension()
                frame_path = os.path.join(self.properties_root,
                                          subset,
                                          label,
                                          prop.get_property_alias(),
                                          frame_name_property_ext)

                try:
                    os.remove(frame_path)
                except FileNotFoundError as fnf:
                    # When we try to remove the frame that doesn't exist, we will receive a FileNotFound Error
                    print('Tried to remove %s but it doesnt exist' % frame_path)
