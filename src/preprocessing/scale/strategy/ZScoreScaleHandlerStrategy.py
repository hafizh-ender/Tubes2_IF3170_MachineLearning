import pandas as pd

from src.preprocessing.common import TransformerStrategy
from src.preprocessing.exception import NotFittedException


class ZScoreScaleHandlerStrategy(TransformerStrategy):
    def __init__(self):
        self._mean = None
        self._std = None

    def fit(self, X, y=None):
        self._mean = pd.Series()
        self._std = pd.Series()

        for feature in X.columns:
            self._mean[feature] = X[feature].mean()
            self._std[feature] = X[feature].std()

    def transform(self, X):
        if self._mean is None or self._std is None:
            raise NotFittedException()

        X_scaled = X.copy()
        for feature in X_scaled.columns:
            if self._std[feature] > 0:
                X_scaled[feature] = (X_scaled[feature] - self._mean[feature]) / self._std[feature]

        return X_scaled
