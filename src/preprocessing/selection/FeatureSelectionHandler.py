from src.compose import PipelineComponent, PipelineProperty
from src.preprocessing.common import TransformerStrategy


def feature_selection_post_handler(pl_property: PipelineProperty):
    pl_property.features['numerical_features'] = [feature for feature in pl_property.X_transformed.columns if
                                                  feature in pl_property.features['numerical_features']]


class FeatureSelectionHandler(PipelineComponent):
    def __init__(self, strategy: TransformerStrategy):
        super().__init__(strategy)
        self.set_post_handler(feature_selection_post_handler)
