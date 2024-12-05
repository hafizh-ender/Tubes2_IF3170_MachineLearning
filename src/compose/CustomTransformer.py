from typing import List
import pandas as pd

from src.preprocessing.common import TransformerStrategy


class CustomTransformer(TransformerStrategy):
    def __init__(self, transformers):
        self._transformers = transformers
        self._N = len(transformers)

    def fit(self, X, y=None):
        for transformer in self._transformers:
            transformer[0].fit(X[transformer[1]], y)

    def transform(self, X):
        X_transformed = X.copy()
        for transformer in self._transformers:
            # X_transformed[transformer[1]] = pd.DataFrame(transformer[0].transform(X[transformer[1]]),
            #                                              columns=transformer[1], index=X_transformed.index)
            transformed = transformer[0].transform(X_transformed[transformer[1]])
            if isinstance(transformed, pd.DataFrame):
                X_transformed = pd.merge(X_transformed.drop(columns=transformer[1]), transformed,
                                         on=X_transformed.index.name)
            else:
                X_transformed[transformer[1]] = pd.DataFrame(transformed, columns=transformer[1],
                                                             index=X_transformed.index)

        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)

        return self.transform(X)
