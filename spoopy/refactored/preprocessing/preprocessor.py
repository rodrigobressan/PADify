from concurrent.futures import ProcessPoolExecutor
from typing import List

import os
from abc import ABC, abstractmethod
from os.path import join, exists

from refactored.preprocessing.Video import Video
from refactored.preprocessing.alignment.aligner import Aligner
from refactored.preprocessing.handler.datahandler import DataHandler, DiskHandler
from refactored.preprocessing.property.property_extractor import PropertyExtractor
from tools import file_utils
from tools.file_utils.file_helper import split_video_into_frames


class Preprocessor(ABC):
    DEFAULT_ALIAS_ALL_DATA = 'all'

    PATH_ORIGINAL_VIDEOS = 'original_videos'
    PATH_SEPARATED_BY_PAI = 'separated_by_pai'
    PATH_SEPARATED_BY_FINETUNING = 'separated_by_finetuning'
    PATH_SEPARATED_BY_SUBSET = 'separated_by_subset'
    PATH_EXTRACTED_FRAMES = 'extracted_frames'
    PATH_EXTRACTED_MAPS = 'extracted_maps'
    PATH_ALIGNED = 'aligned'

    THRESHOLD_MISSING_FRAMES = 0

    def __init__(self,
                 artifacts_root: str,
                 dataset_name: str,
                 properties: List[PropertyExtractor],
                 handler: DataHandler = DiskHandler(),
                 attack_label: str = 'attack',
                 real_label: str = 'real',
                 all_attacks_alias: str = 'all'):

        self.videos_root = join(artifacts_root, self.PATH_ORIGINAL_VIDEOS, dataset_name)
        self.separated_pai_root = join(artifacts_root, self.PATH_SEPARATED_BY_PAI, dataset_name)
        self.finetuning_base_path = join(artifacts_root, self.PATH_SEPARATED_BY_FINETUNING)

        self.separated_finetuning_intra_root = join(artifacts_root, self.PATH_SEPARATED_BY_FINETUNING, 'intra',
                                                    dataset_name)
        self.separated_finetuning_inter_root = join(artifacts_root, self.PATH_SEPARATED_BY_FINETUNING, 'inter',
                                                    dataset_name)
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
        self.all_attacks_alias = all_attacks_alias

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
    def get_person_from_video_name(self, name: str) -> str:
        """
        Abstract method used to obtain the person/subject name from a given video. This method is declared as abstract
        so each custom processor needs to be implemented accordingly to the dataset naming structure.
        :param name: the name from the video without any parsing
        :return: a string containing the subject/person name
        """
        raise NotImplementedError("You should override the method get_person_from_video_name")

    @abstractmethod
    def get_attack_alias_from_frame_name(self, frame_name) -> str:
        """
        This method should be overridden in order to return the name of the attack (tablet, print, mask) from a given
        frame name
        :param frame_name: the name of the frame
        :return: a str object with the attack alias
        """
        raise NotImplementedError("You should override the method get_attack_alias_from_frame_name")

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

    def separate_for_intra_finetuning(self) -> None:
        """
        Used to separate the files for finetuning in the following structure:
            Dataset [CBSR, RA, Rose]
                Property [Depth, Original, Illumination]
                    Subset [Train, Test, Valid]
                        Label [Real, Fake]
                            Frame1.jpg
                            Frame2.jpg
                            ...

        :return:
        """
        path_input = self.aligned_root

        for frame_name, prop, label, subset in self.handler.get_frames_properties(path_input):
            original_path = join(path_input, subset, label, prop, frame_name)
            all_output_path = join(self.separated_finetuning_root, 'intra', prop, subset, label)

            self.copy_if_not_exists(original_path, all_output_path, frame_name)

    def separate_for_intra_finetuning(self) -> None:
        """
        Used to separate the files for finetuning in the following structure:
            Dataset [CBSR, RA, Rose]
                Property [Depth, Original, Illumination]
                    Subset [Train, Test, Valid]
                        Label [Real, Fake]
                            Frame1.jpg
                            Frame2.jpg
                            ...

        :return:
        """
        path_input = self.aligned_root

        for frame_name, prop, label, subset in self.handler.get_frames_properties(path_input):
            original_path = join(path_input, subset, label, prop, frame_name)
            all_output_path = join(self.separated_finetuning_intra_root, prop, subset, label)

            self.copy_if_not_exists(original_path, all_output_path, frame_name)

    def separate_for_inter_finetuning(self) -> None:

        def __move_dataset_into_path(dataset_name: str, output_path: str) -> None:
            dataset_train = os.path.join(self.finetuning_base_path, 'intra', dataset_name,
                                         prop.get_property_alias(), 'train')

            dataset_test = os.path.join(self.finetuning_base_path, 'intra', dataset_name,
                                        prop.get_property_alias(), 'test')

            __move_all_frames_from_set(dataset_train, output_path, dataset_name, 'train')
            __move_all_frames_from_set(dataset_test, output_path, dataset_name, 'test')

        def __move_all_frames_from_set(path_set: str, output_path: str, ds_name: str, ds_set: str):
            """
            Used to move all frames from a given set (train, test, valid) into a new dir
            :param path_set: the path where the labels are located
            :return:
            """
            labels = os.listdir(path_set)

            for label in labels:
                path_label = join(path_set, label)
                for frame in os.listdir(path_label):
                    path_frame = join(path_label, frame)
                    output = join(output_path, label)
                    self.copy_if_not_exists(path_frame, output, frame)

        # gotta have extracted first the intra..
        datasets = os.listdir(join(self.finetuning_base_path, 'intra'))

        for prop in self.properties:
            for origin_dataset in datasets:
                for target_dataset in [d for d in datasets if d != origin_dataset]:
                    output_path = os.path.join(self.finetuning_base_path,
                                               'inter',
                                               origin_dataset, target_dataset,
                                               prop.get_property_alias())

                    if not os.path.exists(output_path):
                        os.makedirs(output_path, exist_ok=True)

                    __move_dataset_into_path(origin_dataset, join(output_path, 'train'))
                    __move_dataset_into_path(target_dataset, join(output_path, 'test'))

    def organize_properties_by_pai(self) -> None:
        """
        Used to organise the properties by their PAI. The output of this process will
        be the frames in the following order:
            Root
                All (Containing all attacks and all authentics)
                    Attack
                        Prop [Depth, Illum]
                            Frame1.jpg
                            Frame2.jpg
                    Real
                        Prop [Depth, Illum]
                            Frame1.jpg
                            Frame2.jpg
                Attack Alias [Print, Cut, Mask, Tablet]
                    Attack
                        Prop [Depth, Illum]
                            Frame1.jpg
                            Frame2.jpg
                    Real
                        Prop [Depth, Illum]
                            Frame1.jpg
                            Frame2.jpg

        """

        path_input = self.aligned_root

        for frame_name, prop, label, subset in self.handler.get_frames_properties(path_input):

            original_path = join(path_input, subset, label, prop, frame_name)
            all_output_path = join(self.separated_pai_root, self.all_attacks_alias, subset, label, prop)

            self.copy_if_not_exists(original_path, all_output_path, frame_name)

            if label == self.default_attack_label:
                attack_alias = self.get_attack_alias_from_frame_name(frame_name)

                # format: /root/attack_alias/subset/label/property
                # print('attack alias name:', frame_name)
                output_path = join(self.separated_pai_root, attack_alias, subset, label, prop)

                self.copy_if_not_exists(original_path, output_path, frame_name)
            else:
                # make a copy into each of the attack folders
                for attack_type in self.pai_config.pai_dict:
                    output_path = join(self.separated_pai_root, attack_type, subset, label, prop)
                    self.copy_if_not_exists(original_path, output_path, frame_name)

    def copy_if_not_exists(self, original_path: str, output_path: str, file_name: str):
        if not exists(join(output_path, file_name)):
            file_utils.file_helper.copy_file(original_path, output_path)
