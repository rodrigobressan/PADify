from abc import ABC, abstractmethod
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


class BaseClassifier(ABC):
    def __init__(self, classifier):
        self.classifier = classifier

    @abstractmethod
    def get_alias(self):
        raise NotImplementedError()

    def fit(self, X_list, y_list):
        self.classifier.fit(X_list, y_list)

    def predict(self, X_list):
        return self.classifier.predict(X_list)

    def predict_proba(self, X):
        """"
        Some classifiers might not have the method to predict with probabilities. If that's
        the case, you don't need to do anything, otherwise you just need to override method
        in the child class"""
        pass


class SvcClassifier(BaseClassifier):
    def __init__(self):
        classifier = OneVsRestClassifier(BaggingClassifier(SVC(verbose=True), n_jobs=-1))
        super().__init__(classifier)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_alias(self):
        return "svc"


class XGBoostClassifier(BaseClassifier):
    def __init__(self):
        classifier = XGBClassifier()
        super().__init__(classifier)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_alias(self):
        return "xgb"
