from abc import ABC, abstractmethod

class ScaleHandlerStrategy(ABC):
    def __init__(self, numerical_features):
        self._numerical_features = numerical_features

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X, y=None):
        pass