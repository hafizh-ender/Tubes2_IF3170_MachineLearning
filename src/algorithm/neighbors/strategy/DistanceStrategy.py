from abc import ABC, abstractmethod


class DistanceStrategy(ABC):
    @abstractmethod
    def calculate(self, X_train, X_test) -> float:
        pass
