from abc import ABC, abstractmethod
import numpy as np


class ProbasClassifier:

    def __init__(self,
                 train_features: np.ndarray,
                 train_labels: np.ndarray,
                 test_features: np.ndarray,
                 test_labels: np.ndarray):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels


