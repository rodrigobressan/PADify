from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

from os.path import join, exists

from refactored.preprocessing.Video import Video
from refactored.preprocessing.cbsr.cbsr_pai import CbsrPaiConfig
from refactored.preprocessing.pai.pai import Pai
from refactored.preprocessing.preprocessor import Preprocessor
from refactored.preprocessing.property.property_extractor import PropertyExtractor
from tools import file_utils


class CbsrProcessor(Preprocessor):
    """
    Used to preprocess the CBSR dataset
    """

    def __init__(self, artifacts_root: str,
                 dataset_name: str,
                 properties: List[PropertyExtractor]):
        self.pai_config = CbsrPaiConfig()
        self.pai_config.add_pai(Pai('cut', ['5', '6', 'HR_3']))
        self.pai_config.add_pai(Pai('print', ['3', '4', 'HR_2']))
        self.pai_config.add_pai(Pai('tablet', ['7', '8', 'HR_4']))

        super(CbsrProcessor, self).__init__(artifacts_root=artifacts_root,
                                            dataset_name=dataset_name,
                                            properties=properties)

    def organize_properties_by_pai(self) -> None:
        """
        Used to organise the properties by their respectives PAI. The output of this process will
        be the frames in the following order:

        """
        for frame_name, prop, label, subset in self.handler.get_frames_aligned(self.aligned_root):

            with ProcessPoolExecutor(max_workers=500) as exec:
                original_path = join(self.aligned_root, subset, label, prop, frame_name)

                # # move all frames into 'all' folder
                all_output_path = join(self.separated_pai_root, 'all', subset, label, prop)
                file_utils.file_helper.copy_file(original_path, all_output_path)

                if label == self.default_attack_label:
                    index_frame = self.get_index_from_name(frame_name)
                    attack_alias = self.get_attack_type_from_name(index_frame)

                    # format: /root/attack_alias/subset/label/property
                    output_path = join(self.separated_pai_root, attack_alias, subset, label, prop)
                    exec.submit(self.copy_if_not_exists, original_path, output_path, frame_name)
                else:
                    # make a copy into each of the attack folders
                    for attack_type in self.pai_config.pai_dict:
                        output_path = join(self.separated_pai_root, attack_type, subset, label, prop)
                        exec.submit(self.copy_if_not_exists, original_path, output_path, frame_name)

    def copy_if_not_exists(self, original_path: str, output_path: str, file_name: str):
        if not exists(join(output_path, file_name)):
            file_utils.file_helper.copy_file(original_path, output_path)
        else:
            print('%s already moved pai' % file_name)

    def get_index_from_name(self, frame_name: str) -> str:
        frame_name = frame_name.split('_frame_')[0]

        if 'HR' in frame_name:
            frame_index = '_'.join(frame_name.split('_')[1:3])
        else:
            frame_index = frame_name.split('_')[1]

        return frame_index

    def get_output_for_attack(self, frame_name: str, output_path: str, property: str) -> Tuple[str, str]:
        """
        Used to return the output path for a given PAI
        :param frame_name: the name of the frame
        :param output_path: where it will be saved
        :param property: the property we're working with
        :return: a Tuple containing both the name of the frame and where it should be stored
        """
        frame_index = self.get_index_from_name(frame_name)

        # retrieve the attack type
        attack_type = self.get_attack_type_from_name(frame_index)

        # <root>/<subset>/<label_atk>/<type_atk>/<prop>/<frame> e.g.: /data/train/attack/mask/depth/frameN.jpg
        output_path = join(output_path, self.default_attack_label, attack_type, property)
        return frame_name, output_path

    def get_attack_type_from_name(self, name: str) -> str:
        """
        Used to get the PAI alias from the file name
        :param name: the name of the file
        :return: the PAI alias
        """
        video_without_ext = self.remove_video_extension(name)
        pai_alias = self.pai_config.get_pai_alias(video_without_ext)
        return pai_alias

    def organize_videos_by_subset_and_label(self):
        """
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
            persons_list = self.handler.list_files(subset_path)  # List all persons/subjects from the given subset

            for person in persons_list:
                persons_path = join(subset_path, person)
                video_list = self.handler.list_files(persons_path)  # List all videos from a given person

                for video_name in video_list:
                    self.process_video(persons_path, person, subset, video_name)

    def get_person_from_video_name(self, name: str) -> str:
        """
        Used to get the person ID from a given video name
        :param name: the name of the file
        :return: the subject/person ID
        """
        # Format: PERSON_VIDEONAME.ext
        return name.split('_')[0]

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
