import numpy as np
from aeon.clustering import BaseClusterer
from aeon.distances import pairwise_distance
from sklearn.utils import check_random_state

from tsml_eval.estimators.clustering.partition.kmedoids_package._kmedoids_wrapper import (
    KMedoids,
)


class KmedoidsPackage(BaseClusterer):

    def __init__(
        self,
        n_clusters: int = 8,
        method: str = "pam",
        distance: str = "euclidean",
        distance_params: dict = None,
        init_algorithm: str = "random",
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = None,
    ):
        self.method = method
        self.distance = distance
        self.init_algorithm = init_algorithm
        self.distance_params = distance_params
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_init = n_init
        self.random_state = random_state
        self.max_iter = max_iter

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

        super().__init__(n_clusters=n_clusters)

    def _score(self, X, y=None):
        return -self.inertia_

    def _predict(self, X, y=None) -> np.ndarray:
        if isinstance(self.distance, str):
            pairwise_matrix = pairwise_distance(
                X,
                self.cluster_centers_,
                metric=self.distance,
                **(self.distance_params or {}),
            )
        else:
            pairwise_matrix = pairwise_distance(
                X,
                self.cluster_centers_,
                metric=self.distance,
                **(self.distance_params or {}),
            )
        return pairwise_matrix.argmin(axis=1)

    def _fit(self, X, y=None):
        if self.distance == "precomputed":
            precomputed_distances = X
        else:
            precomputed_distances = pairwise_distance(
                X, metric=self.distance, **(self.distance_params or {})
            )
        random_state = check_random_state(self.random_state)

        best_centers = None
        best_inertia = np.inf
        best_labels = None

        for i in range(self.n_init):
            initial_medoids = random_state.choice(
                X.shape[0], self.n_clusters, replace=False
            )
            model = KMedoids(
                n_clusters=self.n_clusters,
                metric="precomputed",
                method=self.method,
                init=self.init_algorithm,
                random_state=random_state,
                max_iter=self.max_iter,
                initial_medoids=initial_medoids,
            )
            model.fit(precomputed_distances)
            centers = X[model.medoid_indices_]
            labels = model.labels_
            inertia = model.inertia_
            if inertia < best_inertia:
                best_centers = centers
                best_labels = labels
                best_inertia = inertia
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.cluster_centers_ = best_centers
