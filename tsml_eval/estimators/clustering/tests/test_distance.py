"""Test distance based clusterers."""

import numpy as np
import pytest
from aeon.distances import pairwise_distance
from aeon.testing.data_generation import make_example_3d_numpy

from tsml_eval.estimators.clustering import (
    TimeSeriesAgglomerative,
    TimeSeriesDBScan,
    TimeSeriesHDBScan,
    TimeSeriesOPTICS,
)

n_cases = 20
n_channels = 1
n_timepoints = 10


def _run_distance_test(clusterer, distance, **kwargs):
    X = make_example_3d_numpy(n_cases, n_channels, n_timepoints, return_y=False)
    model = clusterer(**kwargs, distance=distance).fit(X)
    assert model.labels_.shape[0] == n_cases
    assert model.precomputed_distances is None

    precomputed = pairwise_distance(X, metric=distance)

    precomputed_model = clusterer(
        **kwargs, distance=distance, precomputed_distances=precomputed
    ).fit(X)
    assert precomputed_model.labels_.shape[0] == n_cases
    assert np.array_equal(model.labels_, precomputed_model.labels_)
    assert precomputed_model.precomputed_distances is None

    fit_predict_model = clusterer(**kwargs, distance=distance).fit_predict(X)

    assert np.array_equal(model.labels_, fit_predict_model)


@pytest.mark.parametrize(
    "distance",
    [
        "euclidean",
        "adtw",
        "dtw",
        "ddtw",
        "wdtw",
        "wddtw",
        "erp",
        "edr",
        "lcss",
        "twe",
        "msm",
        "shape_dtw",
    ],
)
def test_dbscan(distance):
    """Test DBScan clusterer."""
    _run_distance_test(TimeSeriesDBScan, distance)


@pytest.mark.parametrize(
    "distance",
    [
        "euclidean",
        "adtw",
        "dtw",
        "ddtw",
        "wdtw",
        "wddtw",
        "erp",
        "edr",
        "lcss",
        "twe",
        "msm",
        "shape_dtw",
    ],
)
def test_hdbscan(distance):
    """Test HDBScan clusterer."""
    _run_distance_test(TimeSeriesHDBScan, distance)


@pytest.mark.parametrize(
    "distance",
    [
        "euclidean",
        "adtw",
        "dtw",
        "ddtw",
        "wdtw",
        "wddtw",
        "erp",
        "edr",
        "lcss",
        "twe",
        "msm",
        "shape_dtw",
    ],
)
def test_agglomerative(distance):
    """Test Agglomerative clusterer."""
    _run_distance_test(TimeSeriesAgglomerative, distance)

    X = make_example_3d_numpy(n_cases, n_channels, n_timepoints, return_y=False)
    model = TimeSeriesAgglomerative(distance=distance)
    # Test as experiments set n_clusters like this
    model.set_params(n_clusters=5)
    model.fit(X)
    assert model.n_clusters_ == 5


@pytest.mark.parametrize(
    "distance",
    [
        "euclidean",
        "adtw",
        "dtw",
        "ddtw",
        "wdtw",
        "wddtw",
        "erp",
        "edr",
        "lcss",
        "twe",
        "msm",
        "shape_dtw",
    ],
)
def test_optics(distance):
    """Test OPTICS clusterer."""
    _run_distance_test(TimeSeriesOPTICS, distance)
