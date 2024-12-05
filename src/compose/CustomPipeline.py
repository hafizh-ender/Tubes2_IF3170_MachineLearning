from src.compose import PipelineProperty
from src.compose import CustomTransformer, PipelineComponent


class CustomPipeline:
    """
    args is list of:
    ('name', PipelineComponent)
    """

    def __init__(self, args, pl_property: PipelineProperty, verbose=False):
        self._args = args
        self._pl_property = pl_property
        self._skipped_steps = []
        self._verbose = verbose

    @property
    def pl_property(self):
        return self._pl_property

    def fit(self, X, y=None):
        self._pl_property.X_transformed = X.copy()
        self._pl_property.y_transformed = y.copy() if y is not None else None

        for idx, arg in enumerate(self._args):
            if self._verbose:
                print(f"component: {arg[0]}")

            component = arg[1]

            if len(arg) > 2:
                key = arg[2]
                self._args[idx][1].set_strategy(
                    CustomTransformer(transformers=[(component.strategy, self._pl_property.features[key])]))

                component = self._args[idx][1]

            component.fit_transform(self._pl_property)

            component.post_handler(self._pl_property)

    def transform(self, X, y=None):
        if self._pl_property.X_transformed is None and self._pl_property.y_transformed is None:
            raise Exception()

        self.pl_property.X_transformed = X.copy()
        self.pl_property.y_transformed = y.copy() if y is not None else None

        for arg in self._args:
            if self._verbose:
                print(f"component: {arg[0]}")

            component = arg[1]
            component.transform(self._pl_property)

        return self._pl_property.X_transformed, self._pl_property.y_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)

        X_transformed = X.copy()
        X_transformed = self.transform(X_transformed)

        return X_transformed
