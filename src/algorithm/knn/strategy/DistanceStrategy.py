from abc import ABC, abstractmethod


class DistanceStrategy(ABC):
    @abstractmethod
    def calculate(self, x1, x2) -> float:
        pass
