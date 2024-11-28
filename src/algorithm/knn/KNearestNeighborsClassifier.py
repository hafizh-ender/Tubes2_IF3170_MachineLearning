from typing import List, Union, Any

import numpy as np
from numpy import ndarray, dtype
from scipy.stats import mode

from .strategy import DistanceStrategy, EuclideanDistanceStrategy
from src.exception import InconsistentTrainingAndTestingNumberOfFeaturesException, \
    InconsistentTrainingNumberOfInstancesException, \
    NumberOfNeighborsException, TrainingDataIsNotDefinedException, TestingDataIsNotDefinedException


class KNearestNeighborsClassifier:
    _X_train: ndarray[Any, dtype[Any]]
    _Y_train: ndarray[Any, dtype[Any]]
    _X_test: ndarray[Any, dtype[Any]]

    _n_instances_train: int
    _n_instances_test: int
    _n_features: int

    _k_neighbors_distances: ndarray[Any, dtype[Any]]

    def __init__(self, n_neighbors: int = 2, distance_strategy: DistanceStrategy = EuclideanDistanceStrategy):
        self._n_neighbors, self._distance_strategy = n_neighbors, distance_strategy

    @property
    def n_neighbors(self) -> int:
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, val: int) -> None:
        if val < 1:
            raise NumberOfNeighborsException()

        self._n_neighbors = val

    @property
    def distance_strategy(self) -> DistanceStrategy:
        return self._distance_strategy

    @property
    def k_neighbors_distances(self) -> List[List[Union[int, float]]]:
        return self._k_neighbors_distances

    def set_distance_strategy(self, strategy: DistanceStrategy):
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

        Y_predicted = np.zeros(self._n_instances_test)
        self._k_neighbors_distances = np.array([np.zeros(self._n_neighbors) for _ in range(self._n_instances_test)])

        for i in range(self._n_instances_test):
            k_neighbors, distances = self._find_k_neighbors(self._X_test[i])

            self._k_neighbors_distances[i], Y_predicted[i] = distances, mode(k_neighbors).mode

        return Y_predicted

    # return the k nearest neighbors and the distances
    def _find_k_neighbors(self, x: List[List[Union[int, float]]]) -> (
            List[Union[int, float]], List[List[Union[int, float]]]):
        if self._X_test is None:
            raise TestingDataIsNotDefinedException()

        distances = np.zeros(self._n_instances_train)
        for i in range(self._n_instances_train):
            distances[i] = self._distance_strategy.calculate(x, self._X_train[i])

        indices = distances.argsort()[:self.n_neighbors]

        return self._Y_train[indices], distances[indices]
