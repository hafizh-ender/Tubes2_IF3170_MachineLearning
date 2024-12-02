import pandas as pd

from src.preprocessing.scale.strategy import ScaleHandlerStrategy
from src.preprocessing.exception import NotFittedException


class ZScoreScaleHandlerStrategy(ScaleHandlerStrategy):
    def __init__(self, numerical_features):
        super().__init__(numerical_features)
        self._mean = None
        self._std = None

    def fit(self, X):
        self._mean = pd.Series()
        self._std = pd.Series()

        for feature in self._numerical_features:
            self._mean[feature] = X[feature].mean()
            self._std[feature] = X[feature].std()

    def transform(self, X, y=None):
        if self._mean is None or self._std is None:
            raise NotFittedException()

        X_scaled = X.copy()
        for feature in self._numerical_features:
            if self._std[feature] > 0:
                X_scaled[feature] = (X_scaled[feature] - self._mean[feature]) / self._std[feature]

        return X_scaled