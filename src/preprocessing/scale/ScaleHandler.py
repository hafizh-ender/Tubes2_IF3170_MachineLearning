from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing.scale.strategy import ScaleHandlerStrategy, ZScoreScaleHandlerStrategy

class ScaleHandler(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features, strategy=None):
        self._numerical_features = numerical_features
        self._strategy = ZScoreScaleHandlerStrategy(numerical_features) if strategy is None else strategy

    def set_strategy(self, strategy: ScaleHandlerStrategy):
        self._strategy = strategy

        return self

    def fit(self, X):
        self._strategy.fit(X)

    def transform(self, X, y=None):
        return self._strategy.transform(X, y)
