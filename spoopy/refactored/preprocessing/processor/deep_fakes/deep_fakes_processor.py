from typing import List

from os.path import join

from refactored.preprocessing.Video import Video
from refactored.preprocessing.pai.pai import Pai
from refactored.preprocessing.preprocessor import Preprocessor
from refactored.preprocessing.processor.replay_attack.ra_pai import DefaultPaiConfig
from refactored.preprocessing.property.property_extractor import PropertyExtractor


class DeepFakesProcessor(Preprocessor):
    """
    Used to preprocess the Replay Attack dataset
    """

    def __init__(self, artifacts_root: str,
                 dataset_name: str,
                 properties: List[PropertyExtractor]):
        self.pai_config = DefaultPaiConfig()

        super(DeepFakesProcessor, self).__init__(artifacts_root=artifacts_root,
                                          dataset_name=dataset_name,
                                          properties=properties)

    """
    Overridden methods
    """

    def organize_videos_by_subset_and_label(self):
        """
        Overridden method from Preprocessor class
        Organize files in the following order:

        Dataset name (Cbsr, RA)
            PAI Type (Print, tablet, All)
                Set (Train, Test)
                    Label (Real, Fake)
                    ====
                        Frames (.png, .jpg)
                        Features (.npy)
                        Models (.sav)
       """

        subset_list = self.handler.list_files(self.videos_root)  # List all subsets (train, test)

        for subset in subset_list:
            print(subset)
            subset_path = join(self.videos_root, subset)
            labels_list = self.handler.list_files(subset_path)  # List all labels (attack, real)

            for label in labels_list:
                label_path = join(subset_path, label)
                video_list = self.handler.list_files(label_path)  # List all videos from a given person

                for video_name in video_list:
                    person = video_name.split('_')[2]
                    self.process_video(label_path, person, subset, video_name)

    def get_attack_alias_from_frame_name(self, frame_name) -> str:
        """"
        Overridden method from Preprocessor class.
        Used to get the attack alias from a given frame name.
        :param frame_name: the name of the frame currently being looked
        :return: the attack alias (print, tablet, mask) from that given frame name
        """
        # index = self._get_index_from_name(frame_name)
        attack_alias = self._get_attack_type_from_frame(frame_name)
        return attack_alias

    def get_person_from_video_name(self, name: str) -> str:
        """
        Overriden method from PreProcessor class
        Used to get the person ID from a given video name
        :param name: the name of the file
        :return: the subject/person ID
        """
        # Format: PERSON_VIDEONAME.ext
        return name.split('_')[0]

    """
    Helper methods
    """

    def _get_index_from_name(self, frame_name: str) -> str:
        """
        Used to parse the video index from the frame name.
        :param frame_name: The complete frame name
        :return: the index from the frame
        """

        raise NotImplementedError()

    def _get_attack_type_from_frame(self, name: str) -> str:
        """
        Used to get the PAI alias from the file name
        :param name: the name of the file
        :return: the PAI alias
        """
        return name.split('_')[6]

    def remove_video_extension(self, video_name: str) -> str:
        """
        Used to remove the extension from a given video
        :param video_name: the video name with extension
        :return: the video name without the ext
        """
        return video_name.split('.')[0]

    def process_video(self, persons_path, person, subset, video_name):
        """
        Used to move a given video into the proper folder
        :param persons_path:
        :param person:
        :param subset:
        :param video_name:
        :return:
        """
        video_without_ext = self.remove_video_extension(video_name)
        pai_alias = self.pai_config.get_pai_alias(video_without_ext)

        # when there is no pai alias, it means that the video is from an authentic user
        if not pai_alias:
            is_attack = False
        else:
            is_attack = True

        video_path = join(persons_path, video_name)
        video = Video(path=video_path,
                      name=video_name,
                      person=person,
                      subset=subset,
                      is_attack=is_attack,
                      pai=pai_alias)

        self.move_video_to_proper_dir(video)
        print(video_name, " - ", is_attack)
