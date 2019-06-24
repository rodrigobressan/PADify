from os.path import join
from typing import List

from refactored.preprocessing.Video import Video
from refactored.preprocessing.preprocessor import Preprocessor
from refactored.preprocessing.processor.replay_attack.ra_pai import DefaultPaiConfig
from refactored.preprocessing.property.property_extractor import PropertyExtractor
from tools.file_utils import file_helper


class NuaaProcessor(Preprocessor):
    """
    Used to preprocess the Replay Attack dataset
    """

    def __init__(self, artifacts_root: str,
                 dataset_name: str,
                 properties: List[PropertyExtractor]):
        self.pai_config = DefaultPaiConfig()

        super(NuaaProcessor, self).__init__(artifacts_root=artifacts_root,
                                            dataset_name=dataset_name,
                                            properties=properties, attack_label='fake')

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
        root_path = self.videos_root
        train_fake_indexes = join(root_path, 'imposter_train_raw.txt')
        train_real_indexes = join(root_path, 'client_train_raw.txt')

        test_fake_indexes = join(root_path, 'imposter_test_raw.txt')
        test_real_indexes = join(root_path, 'client_test_raw.txt')

        output_path = self.extracted_frames_root
        real_path = join(root_path, 'ClientRaw')
        fake_path = join(root_path, 'ImposterRaw')


        self.move_files_into_set(output_path, real_path, train_real_indexes, 'train', self.default_real_label)
        self.move_files_into_set(output_path, real_path, test_real_indexes, 'test', self.default_real_label)

        self.move_files_into_set(output_path, fake_path, train_fake_indexes, 'train', self.default_attack_label)
        self.move_files_into_set(output_path, fake_path, test_fake_indexes, 'test', self.default_attack_label)

    def move_files_into_set(self, output_path: str, origin_path: str, indexes_path: str, set, label: str):
        print(set + ' ' + label)
        with (open(indexes_path, 'r')) as f:
            files = f.readlines()
            for file in files:
                file_fixed = file.replace('\\', '/').replace('\n', '')
                file_name = file_fixed.split('/')[1]
                current_file = join(origin_path, file_fixed)

                output = join(output_path, set, label)
                print('    current file:', current_file)
                print('    output:', output)
                print('    final name:', file_name)
                file_helper.copy_file_rename(current_file, output, file_name)

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

        0 - id
        1 - glasses
            0 = with, 1 = no
        2 - location and light conditions
            01: up-down-rotate
            02: up-down-twist
            03: left-right-rotate
            04: left-right-twist
            05: close--window-open-lights
            07: open-window-open-lights
            08: open-widow-shut-lights
            08: still
        3 - session
        4 - pic number
        """

        variation = name.split('_')[2]


        if variation == '01' or variation == '00':
            return 'up-down-rotate'
        if variation == '02':
            return 'up-down-twist'
        if variation == '03':
            return 'left-right-rotate'
        if variation == '04':
            return 'left-right-twist'
        if variation == '05':
            return 'close--window-open-lights'
        if variation == '06':
            return 'open-window-open-lights'
        if variation == '07':
            return 'open-widow-shut-lights'
        if variation == '08':
            return 'still'

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
