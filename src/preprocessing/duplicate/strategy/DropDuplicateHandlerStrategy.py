from src.preprocessing.common import TransformerStrategy


class DropDuplicateHandlerStrategy(TransformerStrategy):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        X_duplicates_removed = X.copy()
        X_duplicates_removed.drop_duplicates(inplace=True)

        return X_duplicates_removed
