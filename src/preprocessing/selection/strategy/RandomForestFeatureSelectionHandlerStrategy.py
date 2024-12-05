import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing.common import TransformerStrategy
from src.preprocessing.exception import NotFittedException


class RandomForestFeatureSelectionHandlerStrategy(TransformerStrategy):
    def __init__(self):
        self._rf = RandomForestClassifier(random_state=42)
        self._selected_features = None

    def fit(self, X, y):
        X_transformed = X.copy()

        self._rf.fit(X_transformed, y)
        feature_importance_df = (
            pd.DataFrame({'Feature': X_transformed.columns, 'Importance': self._rf.feature_importances_})
            .sort_values(by='Importance', ascending=False))
        N = int(len(self._rf.feature_importances_) * 0.8)  # top 80%
        self._selected_features = feature_importance_df['Feature'][:N].values

    def transform(self, X):
        if self._selected_features is None:
            raise NotFittedException()

        X_transformed = X.copy()

        return X_transformed[self._selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)

        return self.transform(X)
