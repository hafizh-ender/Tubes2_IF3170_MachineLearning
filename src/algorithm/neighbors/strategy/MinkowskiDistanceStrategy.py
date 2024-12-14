import numpy as np
from . import DistanceStrategy


class MinkowskiDistanceStrategy(DistanceStrategy):
    def __init__(self, p: int = 1):
        if p < 1:
            raise ValueError("p must be equal or greater than 1")

        self.p = p

    def calculate(self, X_train, X_test) -> float:
        return np.sum(np.abs(X_test[:, None] - X_train) ** self.p, axis=-1) ** (1 / self.p)
