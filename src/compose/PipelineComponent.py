from typing import Callable, Any

from src.preprocessing.common import TransformerStrategy
from src.compose.io.strategy import CommonIOTransformerStrategy
from src.compose.io import IOTransformerStrategy
from src.compose import PipelineProperty


class PipelineComponent():
    def __init__(self, strategy: TransformerStrategy):
        self._strategy = strategy
        self._io_strategy = CommonIOTransformerStrategy(self._strategy)
        self._post_handler: Callable[[PipelineProperty], Any] = lambda pl_property: None

    def __call__(self, *args, **kwargs):
        return self._post_handler(*args, **kwargs)

    @property
    def io_strategy(self) -> CommonIOTransformerStrategy:
        return self._io_strategy

    def set_io_strategy(self, io_strategy: IOTransformerStrategy):
        self._io_strategy = io_strategy

        return self

    def set_strategy(self, strategy: TransformerStrategy):
        self._io_strategy.set_strategy(strategy)

        return self

    def set_post_handler(self, post_handler: Callable[[PipelineProperty], Any]):
        if callable(post_handler):
            self._post_handler = post_handler
        else:
            raise ValueError("post_handler must be callable")

        return self

    @property
    def strategy(self):
        return self._strategy

    @property
    def post_handler(self):
        return self._post_handler

    """
    fit(), transform(), and fit_transform() modify the input
    """

    def fit(self, pl_property: PipelineProperty):
        self._io_strategy.fit(pl_property)

    def transform(self, pl_property: PipelineProperty):
        return self._io_strategy.transform(pl_property)

    def fit_transform(self, pl_property: PipelineProperty):
        self._io_strategy.fit(pl_property)

        return self._io_strategy.transform(pl_property)
