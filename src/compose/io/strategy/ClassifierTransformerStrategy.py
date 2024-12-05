from src.compose.io.IOTransformerStrategy import IOTransformerStrategy
from src.preprocessing.common import TransformerStrategy
from src.compose import PipelineProperty


class ClassifierTransformerStrategy(IOTransformerStrategy):
    def __init__(self, strategy: TransformerStrategy):
        super().__init__(strategy)

    def fit(self, pl_property: PipelineProperty):
        self._strategy.fit(pl_property.X_transformed, pl_property.y_transformed)

    def transform(self, pl_property: PipelineProperty):
        pl_property.y_transformed = self._strategy.transform(pl_property.X_transformed)

    def fit_transform(self, pl_property: PipelineProperty):
        self._strategy.fit(pl_property.X_transformed, pl_property.y_transformed)

        return self.transform(pl_property)