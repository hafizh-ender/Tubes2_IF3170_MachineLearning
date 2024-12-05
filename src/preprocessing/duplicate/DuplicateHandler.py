from src.compose import PipelineComponent
from src.preprocessing.common import TransformerStrategy
from src.compose import PipelineProperty


def duplicate_post_handler(pl_property: PipelineProperty):
    pl_property.y_transformed = pl_property.y_transformed.loc[
        pl_property.y_transformed.index.isin(pl_property.X_transformed.index)]


class DuplicateHandler(PipelineComponent):
    def __init__(self, strategy: TransformerStrategy):
        super().__init__(strategy)
        self.set_post_handler(duplicate_post_handler)
