import numpy as np
from . import DistanceStrategy


class MinkowskiDistance(DistanceStrategy):
    def __init__(self, p: int = 1):
        if p < 1:
            raise ValueError("p must be equal or greater than 1")

        self.p = p

    def calculate(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.sum(np.abs(point1 - point2) ** self.p) ** (1 / self.p)
