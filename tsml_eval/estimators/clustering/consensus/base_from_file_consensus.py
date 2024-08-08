from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from tsml.base import _clone_estimator
from sklearn import preprocessing


class BaseFromFileConsensus(BaseEstimator, ClusterMixin, ABC):

    def __init__(self, clusterers: list[str], n_clusters=8, random_state=None, skip_y_check=False, overwrite_y=False):
        self.clusterers = clusterers
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.overwrite_y = overwrite_y
        self.skip_y_check = skip_y_check

        self.labels_ = None

        self._clusterers = None

    def _check_x(self, X):
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        return self._validate_data(X=X, ensure_min_samples=self.n_clusters)

    def _load_results_from_file(self, X, file_name, y=None):
        cluster_assignments = np.zeros(
            (len(self.clusterers), X.shape[0]), dtype=np.int32
        )
        for i, path in enumerate(self.clusterers):
            f = open(path + file_name)
            lines = f.readlines()
            line2 = lines[2].split(",")

            # verify file matches data
            if len(lines) - 3 != X.shape[0]:
                raise ValueError(
                    f"n_instances of {path + file_name} does not match X, "
                    f"expected {X.shape[0]}, got {len(lines) - 3}"
                )
            if (
                    y is not None
                    and not self.skip_y_check
                    and len(np.unique(y)) != int(line2[5])
            ):
                raise ValueError(
                    f"n_classes of {path + file_name} does not match X, "
                    f"expected {len(np.unique(y))}, got {line2[6]}"
                )

            for j in range(X.shape[0]):
                line = lines[j + 3].split(",")

                if self.overwrite_y:
                    if i == 0:
                        y[j] = float(line[0])
                    elif not self.skip_y_check:
                        assert y[j] == float(line[0])
                elif y is not None and not self.skip_y_check:
                    if i == 0:
                        le = preprocessing.LabelEncoder()
                        y = le.fit_transform(y)
                    assert float(line[0]) == y[j]

                cluster_assignments[i][j] = int(line[1])

            uc = np.unique(cluster_assignments[i])
            clusters = np.arange(self.n_clusters)
            for c in uc:
                if c not in clusters:
                    raise ValueError(
                        "Input clusterers must have cluster labels in the range "
                        "0 to  n_clusters - 1."
                    )
        return cluster_assignments

    def fit(self, X, y=None):
        """Fit model to X using IVC."""
        X = self._check_x(X)

        if self.random_state is not None:
            file_name = f"trainResample{self.random_state}.csv"
        else:
            file_name = "trainResample.csv"

        cluster_assignments = self._load_results_from_file(X, file_name, y)
        # load train file at path (trainResample.csv if random_state is None,
        # trainResample{self.random_state}.csv otherwise)

        self.labels_ = self._build_ensemble(cluster_assignments)
        return self

    def predict(self, X):
        """Predict cluster labels for X."""
        X = self._check_x(X)

        if self.random_state is not None:
            file_name = f"testResample{self.random_state}.csv"
        else:
            file_name = "testResample.csv"

        cluster_assignments = self._load_results_from_file(X, file_name)
        return self._build_ensemble(cluster_assignments)



    def predict_proba(self, X):
        """Predict cluster probabilities for X."""
        preds = self.predict(X)
        n_cases = len(preds)
        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = int(max(preds)) + 1
        dists = np.zeros((X.shape[0], n_clusters))
        for i in range(n_cases):
            dists[i, preds[i]] = 1
        return dists

    @abstractmethod
    def _build_ensemble(self, cluster_assignments) -> np.ndarray:
        """Build ensemble from cluster assignments.

        If you had 5 clusterers then cluster_assignments would be a 5xN array where
        N is the number of instances in the dataset and the value is their cluster
        assignment.

        Parameters
        ----------
        cluster_assignments : np.ndarray
            Array of cluster assignments.

        Returns
        -------
        np.ndarray
            Ensemble cluster assignments.
        """
        ...
