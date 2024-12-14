from typing import List, Union, Any, Tuple

import numpy as np
from numpy import ndarray, dtype
from scipy.stats import mode
import joblib
import os

from src.algorithm.neighbors.strategy import DistanceStrategy, EuclideanDistanceStrategy
from src.exception import InconsistentTrainingAndTestingNumberOfFeaturesException, \
    InconsistentTrainingNumberOfInstancesException, \
    NumberOfNeighborsException, TrainingDataIsNotDefinedException, TestingDataIsNotDefinedException, \
    InvalidPathToModelException
from src import Utils


class KNearestNeighborsClassifier:
    _X_train: ndarray[Any, dtype[Any]]
    _Y_train: ndarray[Any, dtype[Any]]
    _X_test: ndarray[Any, dtype[Any]]

    _n_instances_train: int
    _n_instances_test: int
    _n_features: int

    _k_neighbors_distances: ndarray[Any, dtype[Any]]

    def __init__(self, n_neighbors: int = 2, distance_strategy: DistanceStrategy = EuclideanDistanceStrategy(), batch_size=5000):
        self._k_neighbors, self._distance_strategy = n_neighbors, distance_strategy
        self._batch_size = batch_size

    @property
    def n_neighbors(self) -> int:
        return self._k_neighbors

    @n_neighbors.setter
    def n_neighbors(self, val: int) -> None:
        if val < 1:
            raise NumberOfNeighborsException()

        self._k_neighbors = val

    @property
    def distance_strategy(self) -> DistanceStrategy:
        return self._distance_strategy

    @property
    def k_neighbors_distances(self) -> List[List[Union[int, float]]]:
        return self._k_neighbors_distances

    def set_distance_strategy(self, strategy: DistanceStrategy) -> None:
        self._distance_strategy = strategy

    def fit(self, X_train: List[List[Union[int, float]]], Y_train: List[Union[int, float]]) -> None:
        X_train, Y_train = np.array(X_train), np.array(Y_train)
        if X_train.shape[0] != Y_train.shape[0]:
            raise InconsistentTrainingNumberOfInstancesException()

        self._X_train, self._Y_train = X_train, Y_train
        self._n_instances_train, self._n_features = self._X_train.shape

    def predict(self, X_test: List[List[Union[int, float]]]) -> List[Union[int, float]]:
        if self._X_train is None or self._Y_train is None:
            raise TrainingDataIsNotDefinedException()

        X_test = np.array(X_test)
        if X_test.shape[1] != self._n_features:
            raise InconsistentTrainingAndTestingNumberOfFeaturesException

        self._X_test, self._n_instances_test = X_test, X_test.shape[0]

        Y_predicted = []
        batch_size = min(self._batch_size, self._n_instances_test)
        num_of_batches = round(self._n_instances_test / batch_size)

        for i in range(num_of_batches):
            print(f"batch: {i+1} of {num_of_batches}")

            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, self._n_instances_test)

            X_test_batch = X_test[start_idx:end_idx]

            for distances in self._distance_strategy.calculate(self._X_train, X_test_batch):
                indices = distances.argsort()[:self.n_neighbors]
                value, count = np.unique(self._Y_train[indices], return_counts=True)
                Y_predicted.append(value[np.argmax(count)])

        return Y_predicted

    def save(self, filepath: str) -> None:
        if Utils.get_file_extension(filepath) != ".pkl":
            raise InvalidPathToModelException()

        filepath, directory = os.path.join(os.path.abspath(os.getcwd()), filepath), os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str):
        if Utils.get_file_extension(filepath) != ".pkl":
            raise InvalidPathToModelException()

        filepath = os.path.join(os.path.abspath(os.getcwd()), filepath)

        return joblib.load(filepath)
