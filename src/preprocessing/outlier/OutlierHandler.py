from src.compose import PipelineComponent, PipelineProperty
from src.preprocessing.common import TransformerStrategy


class OutlierHandler(PipelineComponent):
    def __init__(self, strategy: TransformerStrategy):
        super().__init__(strategy)
