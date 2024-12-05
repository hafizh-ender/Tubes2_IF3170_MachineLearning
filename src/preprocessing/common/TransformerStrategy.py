from abc import abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin


class TransformerStrategy(BaseEstimator, TransformerMixin):
    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y)

        return self.transform(X)
