from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

from src.preprocessing.exception import NotFittedException

class FeatureSelectionHandler(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features):
        self._rf = RandomForestClassifier(random_state=42)
        self._numerical_features = numerical_features
        self._selected_features = None
        self._selected_numerical_features = []
        self._selected_categorical_features = []

    @property
    def selected_features(self):
        return self._selected_features

    @property
    def selected_numerical_features(self):
        return self._selected_numerical_features

    @property
    def selected_categorical_features(self):
        return self._selected_categorical_features

    def fit(self, X, y):
        X_selected = X.copy()
        X_selected = X_selected.drop(columns=[feature for feature in self._numerical_features if not X_selected[feature].std() > 0])

        self._rf.fit(X_selected, y)
        feature_importance_df = (pd.DataFrame({'Feature': X_selected.columns, 'Importance': self._rf.feature_importances_})
                                 .sort_values(by='Importance', ascending=False))
        N = int(len(self._rf.feature_importances_) * 0.8) # top 80%
        self._selected_features = feature_importance_df['Feature'][:N].values

        self._selected_numerical_features = [feature for feature in self._numerical_features if feature in self._selected_features]
        self._selected_categorical_features = [feature for feature in self._selected_features if feature not in self._selected_numerical_features]

    def transform(self, X):
        if self._selected_features is None:
            raise NotFittedException()

        X_selected = X.copy()

        return X_selected[self._selected_features]


