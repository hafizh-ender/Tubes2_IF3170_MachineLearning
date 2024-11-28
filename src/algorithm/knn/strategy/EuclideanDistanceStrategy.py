import numpy as np
from . import DistanceStrategy


class EuclideanDistanceStrategy(DistanceStrategy):
    def calculate(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.sqrt(np.sum((point1 - point2) ** 2))
