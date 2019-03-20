from typing import List

from abc import abstractmethod


class Pai():
    def __init__(self, alias: str, name_files: List):
        """
        :param alias: the name where the files will be finally stored
        :param names_files: the names of the files that are associated with the given PAI
        """
        self.alias = alias
        self.name_files = name_files


class BasePaiConfig:
    """
    This class should be used as a base class to create PAI configurations. Please refer to
    CbsrPaiConfig class in order to see a working example.
    """

    @abstractmethod
    def get_pai_alias(self, file_name: str) -> str:
        """
        This method should return the PAI alias (as a string) from a given file name
        :param file_name: the file name to be checked
        :return: the PAI alias
        """
        raise NotImplementedError("You should override the method get_pai_alias")
