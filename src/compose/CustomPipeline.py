from src.compose import PipelineProperty
from src.compose import CustomTransformer, PipelineComponent
import pandas as pd


class CustomPipeline:
    """
    args is list of:
    ('name', PipelineComponent)
    """

    def __init__(self, args, pl_property: PipelineProperty, verbose=False, **kwargs):
        self._args = args
        self._pl_property = pl_property
        self._skipped_steps = []
        self._verbose = verbose
        self._config = kwargs

        self._config['counter'] = pd.Series()

    @property
    def pl_property(self):
        return self._pl_property

    def fit(self, X, y=None):
        self._pl_property.X_transformed = X.copy()
        self._pl_property.y_transformed = y.copy() if y is not None else None

        for idx, arg in enumerate(self._args.copy()):
            if 'only_once' in self._config and arg[0] in self._config['only_once'] and arg[0] in self._config[
                'counter'] and self._config['counter'][arg[0]] > 0:
                continue

            if self._verbose:
                print(f"fit: {arg[0]}")

            component = arg[1]

            if len(arg) > 2:
                key = arg[2]
                self._args[idx][1].set_strategy(
                    CustomTransformer(transformers=[(component.strategy, self._pl_property.features[key])]))

                component = self._args[idx][1]

            component.fit_transform(self._pl_property)

            component.post_handler(self._pl_property)

            if arg[0] in self._config['counter']:
                self._config['counter'][arg[0]] += 1
            else:
                self._config['counter'][arg[0]] = 1

    def transform(self, X, y=None):
        if self._pl_property.X_transformed is None and self._pl_property.y_transformed is None:
            raise Exception()

        self.pl_property.X_transformed = X.copy()
        self.pl_property.y_transformed = y.copy() if y is not None else None

        for arg in self._args:
            if 'only_once' in self._config and arg[0] in self._config['only_once'] and arg[0] in self._config[
                'counter'] and self._config['counter'][arg[0]] > 0:
                continue

            if self._verbose:
                print(f"transform: {arg[0]}")

            component = arg[1]
            component.transform(self._pl_property)

        return self._pl_property.X_transformed, self._pl_property.y_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)

        X_transformed = X.copy()
        X_transformed = self.transform(X_transformed)

        return X_transformed
