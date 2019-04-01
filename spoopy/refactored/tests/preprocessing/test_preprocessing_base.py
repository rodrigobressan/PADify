import unittest

from abc import abstractmethod


class TestPreprocessingBase(unittest.TestCase):

    @abstractmethod
    def get_processor(self):
        raise NotImplementedError("You should override the method get_processor")