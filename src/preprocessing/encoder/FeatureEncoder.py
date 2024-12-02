from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from src.preprocessing.exception import NotFittedException

class FeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features, categorical_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self._features = None

    def fit(self, X):
        X_encoded = pd.get_dummies(X, drop_first=False)
        self._features = X_encoded.columns

    def transform(self, X):
        if self._features is None:
            raise NotFittedException()

        X_encoded = pd.get_dummies(X, drop_first=False)

        missing_features = [feature for feature in self._features if feature not in X_encoded.columns]
        for feature in missing_features:
            X_encoded[feature] = 0

        return X_encoded[self._features]