import pandas as pd

from src.preprocessing.exception import NotFittedException
from src.preprocessing.common import TransformerStrategy


class IQRDropOutlierHandlerStrategy(TransformerStrategy):
    def __init__(self):
        self._values = None

    def fit(self, X, y=None):
        self._values = pd.Series()

        for feature in X.columns:
            lower_bound = X[feature].quantile(0.25) - 1.5 * (
                    X[feature].quantile(0.75) - X[feature].quantile(0.25))
            upper_bound = X[feature].quantile(0.75) + 1.5 * (
                    X[feature].quantile(0.75) - X[feature].quantile(0.25))
            self._values[feature] = pd.Series({'upper_bound': upper_bound, 'lower_bound': lower_bound})

    def transform(self, X):
        if self._values is None:
            raise NotFittedException()

        X_transformed = X.copy()
        for feature in X_transformed.columns:
            lower_bound = self._values[feature]['lower_bound']
            upper_bound = self._values[feature]['upper_bound']

            X_transformed = X_transformed[
                (X_transformed[feature] >= lower_bound) &
                (X_transformed[feature] <= upper_bound)
            ]

        return X_transformed
