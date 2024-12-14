import numpy as np
from . import DistanceStrategy


class ManhattanDistanceStrategy(DistanceStrategy):
    def calculate(self, X_train, X_test) -> float:
        return np.abs(X_test[:,None] - X_train).sum(-1)

