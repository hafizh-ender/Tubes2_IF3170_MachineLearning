from abc import ABC, abstractmethod

from src.preprocessing.common import TransformerStrategy


class IOTransformerStrategy(ABC):
    def __init__(self, strategy: TransformerStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: TransformerStrategy):
        self._strategy = strategy

        return self

    @property
    def strategy(self):
        return self._strategy

    @abstractmethod
    def fit(self, pl_property):
        pass

    @abstractmethod
    def transform(self, pl_property):
        pass

    def fit_transform(self, pl_property):
        self.fit(pl_property)

        return self.transform(pl_property)
