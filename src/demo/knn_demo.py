# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# import numpy as np
# import pandas as pd
#
# from src.algorithm.knn import KNearestNeighborsClassifier, EuclideanDistanceStrategy, ManhattanDistanceStrategy, \
#     MinkowskiDistance
#
#
# def compare(metric, verbose: bool = False, p: int = None):
#     if p is None:
#         sklearnKNN = KNeighborsClassifier(k, metric=metric)
#     else:
#         sklearnKNN = KNeighborsClassifier(k, metric=metric, p=p)
#
#     # sklearn
#     sklearnKNN.fit(X_train, Y_train)
#     Y_predicted_sklearn = sklearnKNN.predict(X_test)
#     accuracy_sklearn = accuracy_score(Y_test, Y_predicted_sklearn)
#     distances_sklearn, _ = sklearnKNN.kneighbors(X_test)
#     print(f"[sklearnKNN] Accuracy using {metric} distance: {accuracy_sklearn * 100:.2f}%")
#
#     # scratch
#     scratchKNN.fit(X_train, Y_train)
#     Y_predicted_scratch = scratchKNN.predict(X_test)
#     accuracy_scratch = accuracy_score(Y_test, Y_predicted_scratch)
#     distances_scratch = scratchKNN.k_neighbors_distances
#
#     print(
#         f"[KNearestNeighborsClassifier] Accuracy using {scratchKNN.distance_strategy.__class__.__name__} "
#         f"distance: {accuracy_scratch * 100:.2f}%")
#
#     if verbose is True:
#         df = pd.DataFrame({
#             "Left (sklearn)": list(np.round(distances_sklearn, 4)),
#             "Right (scratch)": list(np.round(distances_scratch, 4)),
#             "Comparison (Left == Right)": [np.array_equal(a1, a2) for a1, a2 in
#                                            zip(distances_sklearn, distances_scratch)]
#         })
#
#         # Set display options to show the entire DataFrame
#         pd.set_option('display.max_columns', None)  # Show all columns
#         pd.set_option('display.max_rows', None)  # Show all rows
#         pd.set_option('display.width', None)  # Adjust the width to None for unlimited
#         pd.set_option('display.max_colwidth', None)  # Show full column content
#
#         print(df.head(5))
#
#     # check
#     check_equal(Y_predicted_sklearn, Y_predicted_scratch, "Y_predicted")
#     check_equal(distances_sklearn, distances_scratch, "distances")
#
#
# def check_equal(Y1, Y2, name: str) -> None:
#     if np.array_equal(Y1, Y2):
#         print(f"{name} are exactly the same")
#     else:
#         print(f"{name} are different.")
#
#
# # number of k nearest neighbors
# k = 3
#
# # Dummy data with random_state
# X, Y = make_classification(
#     n_samples=100,
#     n_features=2,
#     n_classes=2,
#     n_informative=2,  # All features are informative
#     n_redundant=0,  # No redundant features
#     n_repeated=0,  # No repeated features
#     random_state=42
# )
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
#
# scratchKNN = KNearestNeighborsClassifier(n_neighbors=k)
#
# print("===============================================================")
#
# # Euclidean distance
# # scratchKNN.set_distance_strategy(EuclideanDistanceStrategy())
# compare("euclidean", True)
#
# print("===============================================================")
#
# # Manhattan distance
# scratchKNN.set_distance_strategy(ManhattanDistanceStrategy())
# compare("manhattan", True)
#
# print("===============================================================")
#
# # Minkowski distance (p=1)
# scratchKNN.set_distance_strategy(MinkowskiDistance(1))
# compare("minkowski", True, 1)
#
# print("===============================================================")
#
# scratchKNN.save("../../model/knn.pkl")
#
# loaded_model = scratchKNN.load("../../model/knn.pkl")
#
# print(loaded_model.distance_strategy.__class__.__name__)
