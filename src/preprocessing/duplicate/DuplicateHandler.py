from sklearn.base import BaseEstimator, TransformerMixin

class DuplicateHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X, y=None):
        X_duplicates_removed = X.copy()
        X_duplicates_removed.drop_duplicates(inplace=True)

        return X_duplicates_removed
