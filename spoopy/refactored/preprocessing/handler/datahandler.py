from typing import Tuple

import os
from abc import ABC, abstractmethod

from refactored.preprocessing import Video
from tools.file_utils.file_helper import copy_file_rename


class DataHandler(ABC):
    """
    This is the base class for fetching data to load into our Preprocessor. You can choose
    to load from disk or even from database. The objective here is to provide a valid path to where
    the videos and frames are located.
    """

    @abstractmethod
    def list_files(self, path):
        pass

    @abstractmethod
    def move_video(self, video: Video, path: str) -> None:
        pass

    @abstractmethod
    def get_subsets_and_labels(self, path: str) -> [str, str]:
        pass

    @abstractmethod
    def get_frames_structured(self, path: str) -> [str, str, str]:
        pass

    @abstractmethod
    def get_frames_properties(self, path: str) -> [str, str, str, str]:
        pass


    @abstractmethod
    def list_videos(self, path: str) -> Tuple[str, str, str]:
        pass


class DiskHandler(DataHandler):
    """
    This class is a disk-based implementation for the DataHandler base class
    """
    DEFAULT_LABEL_ATTACK = 'attack'

    def list_videos(self, path: str) -> Tuple[str, str, str]:

        """
        This method is used to list all the videos from a given directory once they are organised by
        their subsets and labels.

        The structure should be in the following mode:
            Subset [Train, Test]
                Label [Real, Attack]
                    Video1.mp4
                    Video2.mp4
                    VideoN.mp4
        :param path: the path we're looking to list the videos
        :return: yields each video name along with it's label and subset
        """
        subset_list = self.list_files(path)

        # Subset/split (Train, Test, Enrollment)
        for subset in subset_list:
            subset_path = os.path.join(path, subset)
            labels_list = self.list_files(subset_path)

            # Label of the category (Real, Attack)
            for label in labels_list:
                labels_path = os.path.join(subset_path, label)
                videos_list = self.list_files(labels_path)

                # Name of the current video (1.mp4, 2.mp4)
                for video_name in videos_list:
                    yield [video_name, label, subset]

    def get_subsets_and_labels(self, path: str) -> Tuple[str, str]:
        """
        Yields the subsets and labels from a given path
        :param path: the path to be searched
        :return: a Tuple containing the label and subsets
        """
        subsets_list = os.listdir(path)

        # Subset/split (Train, Test, Enrollment)
        for subset in subsets_list:
            subset_path = os.path.join(path, subset)
            labels_list = os.listdir(subset_path)

            # Label of the category (Real, Attack)
            for label in labels_list:
                yield [label, subset]


    def get_frames_structured(self, base_path: str) -> [str, str, str]:
        """
        Yields the frames alongside with the labels and subsets from a given path
        :param base_path: the path to be searched
        :return: a Tuple containing the frame name, label and subset
        """
        subsets_list = os.listdir(base_path)
        # Subset/split (Train, Test, Enrollment)
        for subset in subsets_list:
            subset_path = os.path.join(base_path, subset)
            labels_list = os.listdir(subset_path)

            # Label of the category (Real, Attack)
            for label in labels_list:
                labels_path = os.path.join(subset_path, label)
                frames_list = os.listdir(labels_path)

                for frame in frames_list:
                    yield [frame, label, subset]

    def get_frames_properties(self, path: str) -> [str, str, str, str]:
        """
        Yields the frame names, along with the properties, labels and subsets
        :param path: the path where we should search
        :return: a Tuple containing the frame name, property alias, label and subset
        """
        subsets_list = os.listdir(path)

        # Subset/split (Train, Test, Enrollment)
        for subset in subsets_list:
            subset_path = os.path.join(path, subset)
            labels_list = os.listdir(subset_path)

            # Label of the category (Real, Attack)
            for label in labels_list:
                labels_path = os.path.join(subset_path, label)
                properties_list = os.listdir(labels_path)

                for prop in properties_list:
                    property_path = os.path.join(labels_path, prop)
                    frames_list = os.listdir(property_path)

                    for frame in frames_list:
                        yield [frame, prop, label, subset]

    def move_video(self, video: Video, output_path: str) -> None:
        """
        This method should used to move a video to a given path
        :param video: the current Video
        :param output_path: where the video should be stored
        """
        file_name_final = video.person + '_' + video.name
        copy_file_rename(video.path, output_path, file_name_final)

    def list_files(self, path):
        """
        Used to list files from disk
        :param path: path to look
        :return: a List containing the list of dirs
        """
        return os.listdir(path)


class TestHandler(DataHandler):
    """
    This class is a child from DataHandler. It should be used mostly for testing.
    """

    def list_files(self, path):
        return ["a", "b", "c"]
