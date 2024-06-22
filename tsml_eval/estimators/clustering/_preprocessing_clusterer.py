"""A tsml wrapper for sklearn clusterers."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["PreprocessingClusterer"]

import numpy as np
from typing import Union
from sklearn.base import ClusterMixin
from sklearn.utils.validation import check_is_fitted
from aeon.clustering.base import BaseClusterer
from aeon.transformations.base import BaseTransformer

VALID_PREPROCESSING_METHODS = ["pca", "umap", "tsfresh", "catch22", "summary"]
VALID_CLUSTERERS = ["kmeans", "kmedoids", "pam", "clara", "clarans"]


class PreprocessingClusterer(BaseClusterer):
    """Clusterer where you perform some preprocessing before clustering.

    Parameters
    ----------
    transformer : object, default=None
        A transformer object that implements fit_transform.
    clusterer : object, default=None
        A clusterer object that implements fit_predict.
    """
    def __init__(
            self,
            preprocessing_method: Union[str, BaseTransformer] = "pca",
            clusterer: Union[str, BaseClusterer] = "kmeans",
            random_state=None,
            n_clusters=None,
    ):
        self.preprocessing_method = preprocessing_method
        self.clusterer = clusterer
        self.random_state = random_state

        self._preprocessing_method = None
        self._clusterer = None

        super().__init__(n_clusters=n_clusters)

    def _check_preprocessing_method(self):
        if isinstance(self.preprocessing_method, BaseTransformer):
            pass
        elif self.preprocessing_method not in VALID_PREPROCESSING_METHODS:
            pass
        else:
            raise ValueError(f"Invalid preprocessing method {self.preprocessing_method}")

    def _check_clusterer(self):
        if isinstance(self.clusterer, BaseClusterer):
            pass
        elif self.clusterer in VALID_CLUSTERERS:
            pass
        else:
            raise ValueError(f"Invalid clusterer {self.clusterer}")

    def _check_valid_params(self):
        self._check_preprocessing_method()
        self._check_clusterer()

    def _score(self, X, y=None):
        pass

    def _predict(self, X, y=None) -> np.ndarray:
        preprocess_x = self._preprocessing_method.transform(X)
        return self._clusterer.predict(preprocess_x)

    def _fit(self, X, y=None):
        self._check_valid_params()
        preprocess_x = self._preprocessing_method.fit_transform(X)
        return self._clusterer.fit(preprocess_x)
