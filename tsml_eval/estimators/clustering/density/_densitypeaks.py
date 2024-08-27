import math

import numpy as np
from aeon.clustering.base import BaseClusterer
from aeon.distances import pairwise_distance


class TimeSeriesDensityPeaks(BaseClusterer):

    def __init__(
        self,
        distance="euclidean",
        dc="auto",
        verbose=True,
        gauss_cutoff=False,
        threshold_metric="kneepoint",  # 'kneepoint' or 'median'
        density_threshold=None,
        distance_threshold=None,
        anormal=True,
        distance_params=None,
    ):
        self.dc = dc
        self.verbose = verbose
        self.gauss_cutoff = gauss_cutoff
        self.threshold_metric = threshold_metric
        self.density_threshold = density_threshold
        self.distance_threshold = distance_threshold
        self.anormal = anormal
        self.distance_params = distance_params
        self.distance = distance

        self.labels_ = None
        self.center_ = None

        super().__init__(n_clusters=1)

    def _fit(self, X, y=None):
        distances, max_dis, min_dis = self.build_distance_matrix(X)

        self.dc = self._select_dc(distances, max_dis, min_dis)
        rho = self._local_density(distances, self.dc)

        delta, nneigh = self._min_neighbor_and_distance(distances, rho, max_dis)

        self.labels_, self.center_ = self._collapse(
            distances, rho, delta, nneigh, self.dc
        )
        print(f"Density threshold: {self.density_threshold}")
        print(f"Distance threshold: {self.distance_threshold}")
        print(f"DC: {self.dc}")

        return self.labels_

    def fit_predict(self, X, y=None) -> np.ndarray:
        self.fit(X, y)
        return self.labels_

    def _predict(self, X, y=None):
        raise NotImplementedError("Predict is supported for TimeSeriesDensityPeaks")

    def _score(self, X, y=None) -> float:
        return 0.0

    def build_distance_matrix(self, X):
        distance_matrix = pairwise_distance(
            X, metric=self.distance, **(self.distance_params or {})
        )
        triangle_upper = np.triu_indices_from(distance_matrix, k=1)
        max_dis, min_dis = np.max(distance_matrix[triangle_upper]), np.min(
            distance_matrix[triangle_upper]
        )
        return distance_matrix, max_dis, min_dis

    def _auto_select_dc(self, distances, max_dis, min_dis):
        dc = (max_dis + min_dis) / 2
        while True:
            nneighs = np.sum(distances < dc) / distances.shape[0] ** 2
            if 0.01 <= nneighs <= 0.002:
                break
            if nneighs < 0.01:
                min_dis = dc
            else:
                max_dis = dc
            dc = (max_dis + min_dis) / 2
            if max_dis - min_dis < 0.0001:
                break
        return dc

    def _select_dc(self, distances, max_dis, min_dis):
        if self.dc == "auto":
            return self._auto_select_dc(distances, max_dis, min_dis)
        percent = 2.0
        position = int(
            distances.shape[0] * (distances.shape[0] + 1) / 2 * percent / 100
        )
        return np.sort(distances[distances > 0])[position]

    def _local_density(self, distances, dc):
        guass_func = lambda dij, dc: math.exp(-((dij / dc) ** 2))
        cutoff_func = lambda dij, dc: 1 if dij < dc else 0
        func = guass_func if self.gauss_cutoff else cutoff_func
        rho = np.zeros(distances.shape[0], dtype=np.float32)
        for i in range(distances.shape[0]):
            for j in range(i + 1, distances.shape[0]):
                temp = func(distances[i, j], dc)
                rho[i] += temp
                rho[j] += temp
        return rho

    def _min_neighbor_and_distance(self, distances, rho, max_dis):
        sort_rho_idx = np.argsort(-rho)
        delta = np.full(distances.shape[0], max_dis, dtype=np.float32)
        nneigh = np.zeros(distances.shape[0], dtype=np.int32)
        delta[sort_rho_idx[0]] = -1.0
        for i in range(1, distances.shape[0]):
            for j in range(i):
                old_i, old_j = sort_rho_idx[i], sort_rho_idx[j]
                if distances[old_i, old_j] < delta[old_i]:
                    delta[old_i] = distances[old_i, old_j]
                    nneigh[old_i] = old_j
        delta[sort_rho_idx[0]] = np.max(delta)
        return delta, nneigh

    def _collapse(self, distances, rho, delta, nneigh, dc):
        cluster, center = {}, {}
        if self.threshold_metric == "median":
            if self.density_threshold is None:
                self.density_threshold = np.median(rho)
            if self.distance_threshold is None:
                self.distance_threshold = np.median(delta)

        elif self.threshold_metric == "kneepoint":
            from kneed import KneeLocator

            try:
                if self.density_threshold is None:
                    self.distance_threshold = KneeLocator(
                        np.sort(rho)[np.sort(rho) > 0],
                        range(1, len(np.sort(rho)[np.sort(rho) > 0]) + 1),
                        curve="convex",
                        direction="decreasing",
                    ).knee
                if self.density_threshold is None:
                    self.density_threshold = np.median(rho)
            except:
                self.density_threshold = np.median(rho)

            try:
                if self.distance_threshold is None:
                    self.distance_threshold = KneeLocator(
                        np.sort(delta)[np.sort(delta) >= 0],
                        range(1, len(np.sort(delta)[np.sort(delta) >= 0]) + 1),
                        curve="convex",
                        direction="decreasing",
                    ).knee
                if self.distance_threshold is None:
                    self.distance_threshold = np.median(delta)
            except:
                self.distance_threshold = np.median(delta)

        for idx, (ldensity, mdistance, nneigh_item) in enumerate(
            zip(rho, delta, nneigh)
        ):
            if (
                ldensity >= self.density_threshold
                and mdistance >= self.distance_threshold
            ):
                center[idx] = idx
                cluster[idx] = idx
            else:
                cluster[idx] = -1

        ordrho = np.argsort(-rho)
        for i in range(ordrho.shape[0]):
            if cluster[ordrho[i]] == -1:
                cluster[ordrho[i]] = cluster[nneigh[ordrho[i]]]

        halo, bord_rho = {}, {}
        for i in range(ordrho.shape[0]):
            halo[i] = cluster[i]
        if len(center) > 0:
            for idx in center.keys():
                bord_rho[idx] = 0.0
            for i in range(rho.shape[0]):
                for j in range(i + 1, rho.shape[0]):
                    if cluster[i] != cluster[j] and distances[i, j] <= dc:
                        rho_aver = (rho[i] + rho[j]) / 2.0
                        if rho_aver > bord_rho[cluster[i]]:
                            bord_rho[cluster[i]] = rho_aver
                        if rho_aver > bord_rho[cluster[j]]:
                            bord_rho[cluster[j]] = rho_aver
            for i in range(rho.shape[0]):
                if rho[i] < bord_rho[cluster[i]]:
                    halo[i] = 0

        if self.anormal:
            for i in range(rho.shape[0]):
                if halo[i] == 0:
                    cluster[i] = -1

        labels_ = np.zeros(distances.shape[0]).astype(int)
        for k in cluster:
            labels_[k] = cluster[k]

        return labels_, list(center.values())


dataset_name = "GunPoint"

if __name__ == "__main__":
    from aeon.datasets import load_from_tsfile
    from aeon.transformations.collection import TimeSeriesScaler
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
        rand_score,
    )

    DATASET_PATH = "/Users/chris/Documents/Phd-data/Datasets/Univariate_ts"
    X_train, y_train = load_from_tsfile(
        f"{DATASET_PATH}/{dataset_name}/{dataset_name}_TRAIN.ts"
    )
    scaler = TimeSeriesScaler()
    X_train = scaler.fit_transform(X_train)
    sz = X_train.shape[-1]
    n_clusters = len(set(y_train))
    dp = TimeSeriesDensityPeaks(distance="euclidean", dc="auto")
    y_pred = dp.fit_predict(X_train)

    print(y_pred)
    print(f"n_clusters: {len(set(y_pred))}")

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"\nAMI {adjusted_mutual_info_score(y_pred, y_train)}")
    print(f"NMI {normalized_mutual_info_score(y_pred, y_train)}")
    print(f"ARI {adjusted_rand_score(y_pred, y_train)}")
    print(f"Rand {rand_score(y_pred, y_train)}")
