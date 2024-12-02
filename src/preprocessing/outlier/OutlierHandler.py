from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing.outlier.strategy import OutlierHandlerStrategy, BoundOutlierHandlerStrategy

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features, strategy=None):
        self._numerical_features = numerical_features
        self._strategy = BoundOutlierHandlerStrategy(numerical_features) if strategy is None else strategy

    def set_strategy(self, strategy: OutlierHandlerStrategy):
        self._strategy = strategy

        return self

    def fit(self, X):
        self._strategy.fit(X)

    def transform(self, X, y=None):
        return self._strategy.transform(X, y)
