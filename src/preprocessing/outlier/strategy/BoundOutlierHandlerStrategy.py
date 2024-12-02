import pandas as pd

from src.preprocessing.outlier.strategy import OutlierHandlerStrategy
from src.preprocessing.exception import NotFittedException


class BoundOutlierHandlerStrategy(OutlierHandlerStrategy):
    def __init__(self, numerical_features):
        super().__init__(numerical_features)
        self._values = None

    def fit(self, X):
        self._values = pd.Series()

        for feature in self._numerical_features:
            lower_bound = X[feature].quantile(0.25) - 1.5 * (
                    X[feature].quantile(0.75) - X[feature].quantile(0.25))
            upper_bound = X[feature].quantile(0.75) + 1.5 * (
                    X[feature].quantile(0.75) - X[feature].quantile(0.25))
            self._values[feature] = pd.Series({'upper_bound': upper_bound, 'lower_bound': lower_bound})

    def transform(self, X, y=None):
        if self._values is None:
            raise NotFittedException()

        X_completed = X.copy()
        for feature in self._numerical_features:
            X_completed[feature] = X_completed[feature].apply(lambda x: self._values[feature]['lower_bound'] if x < self._values[feature]['lower_bound']
            else self._values[feature]['upper_bound'] if x > self._values[feature]['upper_bound'] else x)

        return X_completed