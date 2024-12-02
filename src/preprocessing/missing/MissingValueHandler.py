from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features, categorical_features):
        self._numerical_features = numerical_features
        self._categorical_features = categorical_features
        self._impute_values = pd.Series()

    def fit(self, X):
        self._impute_values = pd.Series()

        for feature in self._categorical_features:
            self._impute_values[feature] = X[feature].mode()[0]

        for feature in self._numerical_features:
            self._impute_values[feature] = X[feature].mean()

    def transform(self, X, y=None):
        X_imputed = X.copy()

        for feature in self._categorical_features:
            X_imputed[feature] = X_imputed[feature].fillna(self._impute_values[feature])

        for feature in self._numerical_features:
            X_imputed[feature] = X_imputed[feature].fillna(self._impute_values[feature])

        return X_imputed