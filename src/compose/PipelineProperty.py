from typing import Dict, List


class PipelineProperty:
    def __init__(self, features: Dict[str, List[int]]):
        self._features = features
        self._X_transformed = None
        self._y_transformed = None

    @property
    def features(self):
        return self._features

    @property
    def X_transformed(self):
        return self._X_transformed

    @X_transformed.setter
    def X_transformed(self, value):
        self._X_transformed = value

    @property
    def y_transformed(self):
        return self._y_transformed

    @y_transformed.setter
    def y_transformed(self, value):
        self._y_transformed = value
