import unittest

import numpy as np

from tools.classifier_probas.classifier_probas import extract_results


class TestClassifierProbas(unittest.TestCase):
    def test_extract_results(self):
        names = ['file1', 'file1', 'file2', 'file2', 'file3', 'file3', 'file4', 'file4']
        y_pred_proba = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]])
        y_test = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        dict_results = extract_results(names, y_pred_proba, y_pred, y_test)

        self.assertEquals(dict_results, )
        print(dict_results)


if __name__ == '__main__':
    unittest.main()
