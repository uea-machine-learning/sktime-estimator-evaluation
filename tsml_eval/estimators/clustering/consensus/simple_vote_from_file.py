import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from tsml.base import _clone_estimator

from tsml_eval.estimators.clustering.consensus.simple_vote import SimpleVote


class FromFileSimpleVote(SimpleVote):
    """
    SimpleVote clustering ensemble.

    Parameters
    ----------
    clusterers : ?
        ?
    n_clusters : int, default=8
        The number of clusters to form.
    skip_y_check : bool, default=False
        ?
    random_state : int, default=None
        The seed for random number generation.

    Attributes
    ----------
    labels_ : ?
        ?

    Examples
    --------
    >>> from tsml_eval.estimators.clustering.consensus.simple_vote import SimpleVote
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> sv = SimpleVote(n_clusters=3, random_state=0)
    >>> sv.fit(iris.data)
    >>> sv.labels_
    none
    """

    def __init__(self, clusterers=None, n_clusters=8, skip_y_check=False, random_state=None):
        self.skip_y_check = skip_y_check

        super(FromFileSimpleVote, self).__init__(
            clusterers=clusterers, n_clusters=n_clusters, random_state=random_state
        )

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "FromFileSimpleVote is not a time series classifier. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=self.n_clusters)

        # load train file at path (trainResample.csv if random_state is None,
        # trainResample{self.random_state}.csv otherwise)
        if self.random_state is not None:
            file_name = f"trainResample{self.random_state}.csv"
        else:
            file_name = "trainResample.csv"

        cluster_assignments = np.zeros((len(self.clusterers), X.shape[0]), dtype=np.int32)
        for i, path in enumerate(self.clusterers):
            f = open(path + file_name, "r")
            lines = f.readlines()
            line2 = lines[2].split(",")

            # verify file matches data
            if len(lines) - 3 != X.shape[0]:
                raise ValueError(
                    f"n_instances of {path + file_name} does not match X, "
                    f"expected {X.shape[0]}, got {len(lines) - 3}"
                )
            if y is not None and not self.skip_y_check and len(np.unique(y)) != int(line2[5]):
                raise ValueError(
                    f"n_classes of {path + file_name} does not match X, "
                    f"expected {len(np.unique(y))}, got {line2[6]}"
                )

            for j in range(X.shape[0]):
                line = lines[j + 3].split(",")

                if y is not None and not self.skip_y_check:
                    if i == 0:
                        le = preprocessing.LabelEncoder()
                        y = le.fit_transform(y)
                    assert float(line[0]) == y[j]

                cluster_assignments[i][j] = int(line[1])

            uc = np.unique(cluster_assignments[i])
            if uc.shape[0] != self.n_clusters:
                raise ValueError(
                    "Input clusterers must have the same number of clusters as the "
                    "FromFileSimpleVote n_clusters."
                )
            elif (np.sort(uc) != np.arange(self.n_clusters)).any():
                raise ValueError(
                    "Input clusterers must have cluster labels in the range "
                    "0 to  n_clusters - 1."
                )

        self._build_ensemble(cluster_assignments)

        return self

    def predict_proba(self, X):
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "FromFileSimpleVote is not a time series classifier. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X = self._validate_data(X=X, reset=False)

        # load train file at path (trainResample.csv if random_state is None,
        # trainResample{self.random_state}.csv otherwise)
        if self.random_state is not None:
            file_name = f"testResample{self.random_state}.csv"
        else:
            file_name = "testResample.csv"

        cluster_assignments = np.zeros((len(self.clusterers), X.shape[0]), dtype=np.int32)
        for i, path in enumerate(self.clusterers):
            f = open(path + file_name, "r")
            lines = f.readlines()

            # verify file matches data
            if len(lines) - 3 != len(X):
                if len(lines) - 3 != X.shape[0]:
                    raise ValueError(
                        f"n_instances of {path + file_name} does not match X, "
                        f"expected {X.shape[0]}, got {len(lines) - 3}"
                    )

            if i == 0:
                for j in range(len(X)):
                    line = lines[j + 3].split(",")
                    cluster_assignments[i][j] = int(line[1])
            else:
                for j in range(len(X)):
                    line = lines[j + 3].split(",")
                    cluster_assignments[i][j] = self._new_labels[i-1][int(line[1])]

        votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=self.n_clusters),
                                    axis=0, arr=cluster_assignments).transpose()

        return votes / len(self.clusterers)
