from src.compose import PipelineComponent
from src.preprocessing.common import TransformerStrategy


class FeatureEncoder(PipelineComponent):
    def __init__(self, strategy: TransformerStrategy):
        super().__init__(strategy)
