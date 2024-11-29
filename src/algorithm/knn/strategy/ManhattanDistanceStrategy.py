import numpy as np
from . import DistanceStrategy


class ManhattanDistanceStrategy(DistanceStrategy):
    def calculate(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.sum(np.abs(point1 - point2))
