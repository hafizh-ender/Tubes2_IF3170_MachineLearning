import numpy as np
from . import DistanceStrategy


class EuclideanDistanceStrategy(DistanceStrategy):
    def calculate(self, X_train, X_test):
        return np.sqrt(-2 * np.dot(X_test, X_train.T) + np.sum(X_train ** 2, axis=1) + (np.sum(X_test ** 2, axis=1))[:, np.newaxis])

