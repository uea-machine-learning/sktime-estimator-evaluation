"""Set classifier function."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import numpy as np
from aeon.clustering import (
    TimeSeriesCLARA,
    TimeSeriesCLARANS,
    TimeSeriesKMeans,
    TimeSeriesKMedoids,
    TimeSeriesKShapes,
)
from aeon.clustering._holdit_k_means import HoldItKmeans
from aeon.distances._distance import DISTANCES_DICT
from aeon.transformations.collection import TimeSeriesScaler
from sklearn.cluster import KMeans

from tsml_eval.estimators.clustering import (
    ElasticSOM,
    KmedoidsPackage,
    KSpectralCentroid,
    TimeSeriesAgglomerative,
    TimeSeriesDBScan,
    TimeSeriesHDBScan,
)
from tsml_eval.estimators.clustering.partition._kshapes_extended import (
    TimeSeriesKShapesExtended,
)
from tsml_eval.utils.datasets import load_experiment_data
from tsml_eval.utils.functions import str_in_nested_list

distance_based_clusterers = [
    # ================================= K-Shapes ===================================
    "kshapes",
    # ================================= K-Shapes ===================================
    # ================================ KSC ===================================
    "kspectralcentroid",
    "ksc",
    "kspectral-centroid",
    # ================================ KSC ===================================
    # ================================= K-Means ===================================
    "kmeans-euclidean",
    "kmeans-squared",
    "kmeans-dtw",
    "kmeans-ddtw",
    "kmeans-wdtw",
    "kmeans-wddtw",
    "kmeans-lcss",
    "kmeans-erp",
    "kmeans-edr",
    "kmeans-twe",
    "kmeans-msm",
    "kmeans-adtw",
    "kmeans-shape_dtw",
    "kmeans-soft_dtw",
    # ================================= K-Means ===================================
    # ================================ K-Medoids ==================================
    "kmedoids-euclidean",
    "kmedoids-squared",
    "kmedoids-dtw",
    "kmedoids-ddtw",
    "kmedoids-wdtw",
    "kmedoids-wddtw",
    "kmedoids-lcss",
    "kmedoids-erp",
    "kmedoids-edr",
    "kmedoids-twe",
    "kmedoids-msm",
    "kmedoids-adtw",
    "kmedoids-shape_dtw",
    "kmedoids-soft_dtw",
    # ================================ K-Medoids ==================================
    # ================================= CLARANS ===================================
    "clarans-euclidean",
    "clarans-squared",
    "clarans-dtw",
    "clarans-ddtw",
    "clarans-wdtw",
    "clarans-wddtw",
    "clarans-lcss",
    "clarans-erp",
    "clarans-edr",
    "clarans-twe",
    "clarans-msm",
    "clarans-adtw",
    "clarans-shape_dtw",
    "clarans-soft_dtw",
    # ================================= CLARANS ===================================
    # ================================== CLARA ====================================
    "clara-euclidean",
    "clara-squared",
    "clara-dtw",
    "clara-ddtw",
    "clara-wdtw",
    "clara-wddtw",
    "clara-lcss",
    "clara-erp",
    "clara-edr",
    "clara-twe",
    "clara-msm",
    "clara-adtw",
    "clara-shape_dtw",
    "clara-soft_dtw",
    # ================================== CLARA ====================================
    # ===================================== PAM ====================================
    "pam-euclidean",
    "pam-squared",
    "pam-dtw",
    "pam-ddtw",
    "pam-wdtw",
    "pam-wddtw",
    "pam-lcss",
    "pam-erp",
    "pam-edr",
    "pam-twe",
    "pam-msm",
    "pam-adtw",
    "pam-shape_dtw",
    "pam-soft_dtw",
    # ===================================== PAM ====================================
    # ===================================== PAMSIL =================================
    "pamsil-euclidean",
    "pamsil-squared",
    "pamsil-dtw",
    "pamsil-ddtw",
    "pamsil-wdtw",
    "pamsil-wddtw",
    "pamsil-lcss",
    "pamsil-erp",
    "pamsil-edr",
    "pamsil-twe",
    "pamsil-msm",
    "pamsil-adtw",
    "pamsil-shape_dtw",
    "pamsil-soft_dtw",
    # ===================================== PAMSIL =================================
    # ===================================== PAMMEDSIL ==============================
    "pammedsil-euclidean",
    "pammedsil-squared",
    "pammedsil-dtw",
    "pammedsil-ddtw",
    "pammedsil-wdtw",
    "pammedsil-wddtw",
    "pammedsil-lcss",
    "pammedsil-erp",
    "pammedsil-edr",
    "pammedsil-twe",
    "pammedsil-msm",
    "pammedsil-adtw",
    "pammedsil-shape_dtw",
    "pammedsil-soft_dtw",
    # ===================================== PAMMEDSIL ==============================
    # ===================================== FasterPAM ==============================
    "fasterpam-euclidean",
    "fasterpam-squared",
    "fasterpam-dtw",
    "fasterpam-ddtw",
    "fasterpam-wdtw",
    "fasterpam-wddtw",
    "fasterpam-lcss",
    "fasterpam-erp",
    "fasterpam-edr",
    "fasterpam-twe",
    "fasterpam-msm",
    "fasterpam-adtw",
    "fasterpam-shape_dtw",
    "fasterpam-soft_dtw",
    # ===================================== FasterPAM ==============================
    # ===================================== BA =====================================
    "kmeans-ba-dtw",
    "kmeans-ba-ddtw",
    "kmeans-ba-wdtw",
    "kmeans-ba-wddtw",
    "kmeans-ba-lcss",
    "kmeans-ba-erp",
    "kmeans-ba-edr",
    "kmeans-ba-twe",
    "kmeans-ba-msm",
    "kmeans-ba-adtw",
    "kmeans-ba-shape_dtw",
    "kmeans-ba-soft_dtw",
    # ===================================== BA =====================================
    # ================================== SSG-BA ====================================
    "kmeans-ssg-ba-dtw",
    "kmeans-ssg-ba-ddtw",
    "kmeans-ssg-ba-wdtw",
    "kmeans-ssg-ba-wddtw",
    "kmeans-ssg-ba-erp",
    "kmeans-ssg-ba-edr",
    "kmeans-ssg-ba-twe",
    "kmeans-ssg-ba-msm",
    "kmeans-ssg-ba-adtw",
    "kmeans-ssg-ba-shape_dtw",
    "kmeans-ssg-ba-soft_dtw",
    # ================================== SSG-BA ====================================
    # ================================== soft-DBA ====================================
    "kmeans-dba-soft_dtw",
    # ================================== soft-DBA ====================================
    # =================================== DBSCAN ===================================
    "DBSCAN-euclidean",
    "DBSCAN-squared",
    "DBSCAN-dtw",
    "DBSCAN-ddtw",
    "DBSCAN-wdtw",
    "DBSCAN-wddtw",
    "DBSCAN-lcss",
    "DBSCAN-erp",
    "DBSCAN-edr",
    "DBSCAN-twe",
    "DBSCAN-msm",
    "DBSCAN-adtw",
    "DBSCAN-shape_dtw",
    # =================================== DBSCAN ===================================
    # =================================== HDBSCAN ==================================
    "HDBSCAN-euclidean",
    "HDBSCAN-squared",
    "HDBSCAN-dtw",
    "HDBSCAN-ddtw",
    "HDBSCAN-wdtw",
    "HDBSCAN-wddtw",
    "HDBSCAN-lcss",
    "HDBSCAN-erp",
    "HDBSCAN-edr",
    "HDBSCAN-twe",
    "HDBSCAN-msm",
    "HDBSCAN-adtw",
    "HDBSCAN-shape_dtw",
    # =================================== HDBSCAN ==================================
    # ================================ Agglomerative ===============================
    "agglomerative-euclidean",
    "agglomerative-squared",
    "agglomerative-dtw",
    "agglomerative-ddtw",
    "agglomerative-wdtw",
    "agglomerative-wddtw",
    "agglomerative-lcss",
    "agglomerative-erp",
    "agglomerative-edr",
    "agglomerative-twe",
    "agglomerative-msm",
    "agglomerative-adtw",
    "agglomerative-shape_dtw",
    # ================================ Agglomerative ===============================
    # =================================== OPTICS ===================================
    "OPTICS-euclidean",
    "OPTICS-squared",
    "OPTICS-dtw",
    "OPTICS-ddtw",
    "OPTICS-wdtw",
    "OPTICS-wddtw",
    "OPTICS-lcss",
    "OPTICS-erp",
    "OPTICS-edr",
    "OPTICS-twe",
    "OPTICS-msm",
    "OPTICS-adtw",
    "OPTICS-shape_dtw",
    # =================================== OPTICS ===================================
    # =================================== SOM ===================================
    "som-euclidean",
    "som-squared",
    "som-adtw",
    "som-dtw",
    "som-ddtw",
    "som-wdtw",
    "som-wddtw",
    "som-erp",
    "som-msm",
    "som-twe",
    "som-shape_dtw",
    "som-soft_dtw",
    # =================================== SOM ===================================
    # ================================ GENERIC NAMES ================================
    "timeserieskmeans",
    "timeserieskmedoids",
    "timeseriesclarans",
    "timeseriesclara",
    "elasticsom",
    "timeserieskshapes",
    # ================================ GENERIC NAMES ================================
]

feature_based_clusterers = [
    ["catch22", "catch22clusterer"],
    ["tsfresh", "tsfreshclusterer"],
    ["summary", "summaryclusterer"],
]

other_clusterers = [
    ["dummyclusterer", "dummy", "dummyclusterer-tsml"],
    "dummyclusterer-aeon",
    "dummyclusterer-sklearn",
]
vector_clusterers = [
    ["kmeans", "kmeans-sklearn"],
    "dbscan",
]

experimental_clusterers = [
    "faster-ssg-adtw",
    "faster-ssg-dtw",
    "faster-ssg-msm",
    "faster-ssg-twe",
    "window-ssg-adtw",
    "window-ssg-dtw",
    "window-ssg-msm",
    "window-ssg-twe",
    "faster-window-ssg-adtw",
    "faster-window-ssg-dtw",
    "faster-window-ssg-msm",
    "faster-window-ssg-twe",
    "40-faster-ssg-adtw",
    "40-faster-ssg-dtw",
    "40-faster-ssg-msm",
    "40-faster-ssg-twe",
    "30-faster-ssg-adtw",
    "30-faster-ssg-dtw",
    "30-faster-ssg-msm",
    "30-faster-ssg-twe",
    "20-faster-ssg-adtw",
    "20-faster-ssg-dtw",
    "20-faster-ssg-msm",
    "20-faster-ssg-twe",
    "10-faster-ssg-adtw",
    "10-faster-ssg-dtw",
    "10-faster-ssg-msm",
    "10-faster-ssg-twe",
    # New ones
    "proper-stopping-ssg-adtw",
    "proper-stopping-ssg-msm",
    "proper-stopping-ssg-twe",
    "approx-stopping-ssg-adtw",
    "approx-stopping-ssg-msm",
    "approx-stopping-ssg-twe",
    "avg-change-stopping-ssg-adtw",
    "avg-change-stopping-ssg-msm",
    "avg-change-stopping-ssg-twe",
    # Initially attempts
    "greedy-kmeans++",
    "random-init",
    "forgy-init",
    "random-init-10-restarts",
    "forgy-init-10-restarts",
    "forgy-init-10-restarts-average-number-iterations",
    # Full window dtw runs
    "k-means-full-window-dtw",
    "k-means-full-window-ddtw",
    "k-means-ba-full-window-dtw",
    "k-means-ba-full-window-ddtw",
    # 5% window dtw runs
    "k-means-5-percent-window-dtw",
    "k-means-5-percent-window-ddtw",
    "k-means-ba-5-percent-window-dtw",
    # ssg-clusterer with proper early stopping
    "kmeans-ssg-forgy-restarts-full-adtw",
    "kmeans-ssg-forgy-restarts-full-msm",
    "kmeans-ssg-forgy-restarts-full-twe",
    "kmeans-ssg-kmeans++-full-adtw",
    "kmeans-ssg-kmeans++-full-msm",
    "kmeans-ssg-kmeans++-full-twe",
    "kmeans-ssg-kmeans++-increase-iterations-full-adtw",
    "kmeans-ssg-kmeans++-increase-iterations-full-msm",
    "kmeans-ssg-kmeans++-increase-iterations-full-twe",
    # With window
    "50-kmeans-ssg-kmeans++-increase-iterations-adtw",
    "50-kmeans-ssg-kmeans++-increase-iterations-msm",
    "50-kmeans-ssg-kmeans++-increase-iterations-twe",
    "40-kmeans-ssg-kmeans++-increase-iterations-adtw",
    "40-kmeans-ssg-kmeans++-increase-iterations-msm",
    "40-kmeans-ssg-kmeans++-increase-iterations-twe",
    "30-kmeans-ssg-kmeans++-increase-iterations-adtw",
    "30-kmeans-ssg-kmeans++-increase-iterations-msm",
    "30-kmeans-ssg-kmeans++-increase-iterations-twe",
    "20-kmeans-ssg-kmeans++-increase-iterations-adtw",
    "20-kmeans-ssg-kmeans++-increase-iterations-msm",
    "20-kmeans-ssg-kmeans++-increase-iterations-twe",
    "10-kmeans-ssg-kmeans++-increase-iterations-adtw",
    "10-kmeans-ssg-kmeans++-increase-iterations-msm",
    "10-kmeans-ssg-kmeans++-increase-iterations-twe",
    # Additional
    "k-means++-k-shapes",
    "k-means++-increase-iterations-k-shapes",
    "k-means++-k-means-soft-dba",
    "k-means++-increase-iterations-k-means-soft-dba",
    # Percentage to use
    "50-average-kmeans-ssg-kmeans++-increase-iterations-msm",
    "50-average-kmeans-ssg-kmeans++-increase-iterations-twe",
    "40-average-kmeans-ssg-kmeans++-increase-iterations-msm",
    "40-average-kmeans-ssg-kmeans++-increase-iterations-twe",
    "30-average-kmeans-ssg-kmeans++-increase-iterations-msm",
    "30-average-kmeans-ssg-kmeans++-increase-iterations-twe",
    "20-average-kmeans-ssg-kmeans++-increase-iterations-msm",
    "20-average-kmeans-ssg-kmeans++-increase-iterations-twe",
    "10-average-kmeans-ssg-kmeans++-increase-iterations-msm",
    "10-average-kmeans-ssg-kmeans++-increase-iterations-twe",
    # Both
    "50-both-kmeans-ssg-kmeans++-increase-iterations-msm",
    "50-both-kmeans-ssg-kmeans++-increase-iterations-twe",
    "40-both-kmeans-ssg-kmeans++-increase-iterations-msm",
    "40-both-kmeans-ssg-kmeans++-increase-iterations-twe",
    "30-both-kmeans-ssg-kmeans++-increase-iterations-msm",
    "30-both-kmeans-ssg-kmeans++-increase-iterations-twe",
    "20-both-kmeans-ssg-kmeans++-increase-iterations-msm",
    "20-both-kmeans-ssg-kmeans++-increase-iterations-twe",
    "10-both-kmeans-ssg-kmeans++-increase-iterations-msm",
    "10-both-kmeans-ssg-kmeans++-increase-iterations-twe",
    # Window in assignment half data in average
    "50-assignment-average-kmeans-ssg-kmeans++-increase-iterations-msm",
    "50-assignment-average-kmeans-ssg-kmeans++-increase-iterations-twe",
    "40-assignment-average-kmeans-ssg-kmeans++-increase-iterations-msm",
    "40-assignment-average-kmeans-ssg-kmeans++-increase-iterations-twe",
    "30-assignment-average-kmeans-ssg-kmeans++-increase-iterations-msm",
    "30-assignment-average-kmeans-ssg-kmeans++-increase-iterations-twe",
    "20-assignment-average-kmeans-ssg-kmeans++-increase-iterations-msm",
    "20-assignment-average-kmeans-ssg-kmeans++-increase-iterations-twe",
    "10-assignment-average-kmeans-ssg-kmeans++-increase-iterations-msm",
    "10-assignment-average-kmeans-ssg-kmeans++-increase-iterations-twe",
    # Final model KESBA
    "kesba-forgy-restarts-twe",
    "kesba-forgy-restarts-msm",
]


def get_clusterer_by_name(
    clusterer_name,
    random_state=None,
    n_jobs=1,
    fit_contract=0,
    checkpoint=None,
    data_vars=None,
    row_normalise=False,
    **kwargs,
):
    """Return a clusterer matching a given input name.

    Basic way of creating a clusterer to build using the default or alternative
    settings. This set up is to help with batch jobs for multiple problems and to
    facilitate easy reproducibility through run_clustering_experiment.

    Generally, inputting a clusterer class name will return said clusterer with
    default settings.

    Parameters
    ----------
    clusterer_name : str
        String indicating which clusterer to be returned.
    random_state : int, RandomState instance or None, default=None
        Random seed or RandomState object to be used in the clusterer if available.
    n_jobs: int, default=1
        The number of jobs to run in parallel for both clusterer ``fit`` and
        ``predict`` if available. `-1` means using all processors.
    fit_contract: int, default=0
        The number of data points to use in the clusterer ``fit`` if available.
    checkpoint: str, default=None
        Checkpoint to save model
    data_vars: list, default=None
        List of arguments to load the dataset using
        `tsml_eval.utils.experiments import load_experiment_data`.
    row_normalise: bool, default=False
        Whether to row normalise the data if it is loaded using data_vars.

    Return
    ------
    clusterer: A BaseClusterer.
        The clusterer matching the input clusterer name.
    """
    c = clusterer_name.lower()

    if str_in_nested_list(distance_based_clusterers, c):
        return _set_clusterer_distance_based(
            c,
            random_state,
            n_jobs,
            fit_contract,
            checkpoint,
            data_vars,
            row_normalise,
            kwargs,
        )
    elif str_in_nested_list(feature_based_clusterers, c):
        return _set_clusterer_feature_based(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(other_clusterers, c):
        return _set_clusterer_other(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(vector_clusterers, c):
        return _set_clusterer_vector(
            c, random_state, n_jobs, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(experimental_clusterers, c):
        return _set_experimental_clusterer(
            c,
            random_state,
            n_jobs,
            fit_contract,
            checkpoint,
            data_vars,
            row_normalise,
            kwargs,
        )
    else:
        raise ValueError(f"UNKNOWN CLUSTERER: {c} in set_clusterer")


def _set_experimental_clusterer(
    c,
    random_state,
    n_jobs,
    fit_contract,
    checkpoint,
    data_vars,
    row_normalise,
    kwargs,
):
    if "distance" in kwargs:
        distance = kwargs["distance"]
    else:
        distance = c.split("-")[-1]

    if distance not in DISTANCES_DICT:
        distance = "dtw"

    if "distance_params" in kwargs:
        distance_params = kwargs["distance_params"]
    else:
        distance_params = _get_distance_default_params(
            distance, data_vars, row_normalise
        )

    average_params = {"distance": distance, **distance_params.copy()}

    average_params = {
        **average_params,
        "method": "holdit_stopping",
        "holdit_num_ts_to_use_percentage": 1.0,
    }
    potential_size_arg = ["50", "40", "30", "20", "10"]
    if any(arg in c for arg in potential_size_arg):
        size = int(c.split("-")[0])

        if "assignment-average" in c:
            window = size / 100
            average_params = {
                **average_params,
                "holdit_num_ts_to_use_percentage": size / 100,
            }
            distance_params = {**distance_params, "window": window}
        elif "average" in c:
            average_params = {
                **average_params,
                "holdit_num_ts_to_use_percentage": size / 100,
            }
        elif "both" in c:
            window = size / 100
            average_params = {
                **average_params,
                "holdit_num_ts_to_use_percentage": size / 100,
                "window": window,
            }
            distance_params = {**distance_params, "window": window}
        else:
            window = size / 100
            distance_params = {**distance_params, "window": window}
            average_params = {
                **average_params,
                "window": window,
            }

    curr_experiments_forgy_restarts = [
        "kmeans-ssg-forgy-restarts-full-adtw",
        "kmeans-ssg-forgy-restarts-full-msm",
        "kmeans-ssg-forgy-restarts-full-twe",
    ]
    if c in curr_experiments_forgy_restarts:
        return HoldItKmeans(
            max_iter=50,
            n_init=10,
            init_algorithm="random",
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            averaging_method="ba",
            average_params=average_params,
            verbose=True,
            **kwargs,
        )

    curr_experiments_kmeans_plus_plus = [
        "kmeans-ssg-kmeans++-full-adtw",
        "kmeans-ssg-kmeans++-full-msm",
        "kmeans-ssg-kmeans++-full-twe",
    ]
    if c in curr_experiments_kmeans_plus_plus:
        return HoldItKmeans(
            max_iter=50,
            n_init=1,
            init_algorithm="kmeans++",
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            averaging_method="ba",
            average_params={
                **average_params,
            },
            verbose=True,
            **kwargs,
        )

    curr_experiments_kmeans_plus_plus_increase_iterations = [
        "kmeans-ssg-kmeans++-increase-iterations-full-adtw",
        "kmeans-ssg-kmeans++-increase-iterations-full-msm",
        "kmeans-ssg-kmeans++-increase-iterations-full-twe",
    ]
    if c in curr_experiments_kmeans_plus_plus_increase_iterations:
        return HoldItKmeans(
            max_iter=300,
            n_init=1,
            init_algorithm="kmeans++",
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            averaging_method="ba",
            average_params={
                **average_params,
                "max_iters": 300,
            },
            verbose=True,
            **kwargs,
        )

    curr_window_experiments = [
        "50-kmeans-ssg-kmeans++-increase-iterations-adtw",
        "50-kmeans-ssg-kmeans++-increase-iterations-msm",
        "50-kmeans-ssg-kmeans++-increase-iterations-twe",
        "40-kmeans-ssg-kmeans++-increase-iterations-adtw",
        "40-kmeans-ssg-kmeans++-increase-iterations-msm",
        "40-kmeans-ssg-kmeans++-increase-iterations-twe",
        "30-kmeans-ssg-kmeans++-increase-iterations-adtw",
        "30-kmeans-ssg-kmeans++-increase-iterations-msm",
        "30-kmeans-ssg-kmeans++-increase-iterations-twe",
        "20-kmeans-ssg-kmeans++-increase-iterations-adtw",
        "20-kmeans-ssg-kmeans++-increase-iterations-msm",
        "20-kmeans-ssg-kmeans++-increase-iterations-twe",
        "10-kmeans-ssg-kmeans++-increase-iterations-adtw",
        "10-kmeans-ssg-kmeans++-increase-iterations-msm",
        "10-kmeans-ssg-kmeans++-increase-iterations-twe",
    ]
    if c in curr_window_experiments:
        return HoldItKmeans(
            max_iter=300,
            n_init=1,
            init_algorithm="kmeans++",
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            averaging_method="ba",
            average_params={
                **average_params,
                "max_iters": 300,
            },
            verbose=True,
            **kwargs,
        )
    curr_percentage_experiments = [
        "50-average-kmeans-ssg-kmeans++-increase-iterations-msm",
        "50-average-kmeans-ssg-kmeans++-increase-iterations-twe",
        "40-average-kmeans-ssg-kmeans++-increase-iterations-msm",
        "40-average-kmeans-ssg-kmeans++-increase-iterations-twe",
        "30-average-kmeans-ssg-kmeans++-increase-iterations-msm",
        "30-average-kmeans-ssg-kmeans++-increase-iterations-twe",
        "20-average-kmeans-ssg-kmeans++-increase-iterations-msm",
        "20-average-kmeans-ssg-kmeans++-increase-iterations-twe",
        "10-average-kmeans-ssg-kmeans++-increase-iterations-msm",
        "10-average-kmeans-ssg-kmeans++-increase-iterations-twe",
    ]
    if c in curr_percentage_experiments:
        return HoldItKmeans(
            max_iter=300,
            n_init=1,
            init_algorithm="kmeans++",
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            averaging_method="ba",
            average_params={
                **average_params,
                "max_iters": 300,
            },
            verbose=True,
            **kwargs,
        )

    curr_both_experiments = [
        "50-both-kmeans-ssg-kmeans++-increase-iterations-msm",
        "50-both-kmeans-ssg-kmeans++-increase-iterations-twe",
        "40-both-kmeans-ssg-kmeans++-increase-iterations-msm",
        "40-both-kmeans-ssg-kmeans++-increase-iterations-twe",
        "30-both-kmeans-ssg-kmeans++-increase-iterations-msm",
        "30-both-kmeans-ssg-kmeans++-increase-iterations-twe",
        "20-both-kmeans-ssg-kmeans++-increase-iterations-msm",
        "20-both-kmeans-ssg-kmeans++-increase-iterations-twe",
        "10-both-kmeans-ssg-kmeans++-increase-iterations-msm",
        "10-both-kmeans-ssg-kmeans++-increase-iterations-twe",
    ]
    if c in curr_both_experiments:
        return HoldItKmeans(
            max_iter=300,
            n_init=1,
            init_algorithm="kmeans++",
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            averaging_method="ba",
            average_params={
                **average_params,
                "max_iters": 300,
            },
            verbose=True,
            **kwargs,
        )

    curr_assignment_average_experiments = [
        "50-assignment-average-kmeans-ssg-kmeans++-increase-iterations-msm",
        "50-assignment-average-kmeans-ssg-kmeans++-increase-iterations-twe",
        "40-assignment-average-kmeans-ssg-kmeans++-increase-iterations-msm",
        "40-assignment-average-kmeans-ssg-kmeans++-increase-iterations-twe",
        "30-assignment-average-kmeans-ssg-kmeans++-increase-iterations-msm",
        "30-assignment-average-kmeans-ssg-kmeans++-increase-iterations-twe",
        "20-assignment-average-kmeans-ssg-kmeans++-increase-iterations-msm",
        "20-assignment-average-kmeans-ssg-kmeans++-increase-iterations-twe",
        "10-assignment-average-kmeans-ssg-kmeans++-increase-iterations-msm",
        "10-assignment-average-kmeans-ssg-kmeans++-increase-iterations-twe",
    ]
    if c in curr_assignment_average_experiments:
        return HoldItKmeans(
            max_iter=300,
            n_init=1,
            init_algorithm="kmeans++",
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            averaging_method="ba",
            average_params={
                **average_params,
                "max_iters": 300,
            },
            verbose=True,
            **kwargs,
        )

    curr_kshapes_experiments = [
        "k-means++-k-shapes",
        "k-means++-increase-iterations-k-shapes",
    ]
    if c in curr_kshapes_experiments:
        if c == "k-means++-k-shapes":
            return TimeSeriesKShapesExtended(
                init_algorithm="kmeans++",
                max_iter=50,
                n_init=10,
                tol=1e-06,
                random_state=random_state,
                **kwargs,
            )
        elif c == "k-means++-increase-iterations-k-shapes":
            return TimeSeriesKShapesExtended(
                init_algorithm="kmeans++",
                max_iter=300,
                n_init=1,
                tol=1e-06,
                random_state=random_state,
                **kwargs,
            )

    curr_kmeans_soft_dba_experiments = [
        "k-means++-k-means-soft-dba",
        "k-means++-increase-iterations-k-means-soft-dba",
    ]
    if c in curr_kmeans_soft_dba_experiments:
        average_params = {
            **average_params,
            "method": "soft_dba",
        }
        if c == "k-means++-k-means-soft-dba":
            return TimeSeriesKMeans(
                max_iter=50,
                n_init=10,
                init_algorithm="kmeans++",
                distance="soft_dtw",
                distance_params=distance_params,
                random_state=random_state,
                averaging_method="ba",
                average_params=average_params,
                **kwargs,
            )
        elif c == "k-means++-increase-iterations-k-means-soft-dba":
            return TimeSeriesKMeans(
                max_iter=300,
                n_init=1,
                init_algorithm="kmeans++",
                distance="soft_dtw",
                distance_params=distance_params,
                random_state=random_state,
                averaging_method="ba",
                average_params={
                    **average_params,
                    "max_iters": 300,
                },
                **kwargs,
            )

    curr_kesba_experiments = [
        "kesba-forgy-restarts-twe",
        "kesba-forgy-restarts-msm",
    ]
    if c in curr_kesba_experiments:
        if distance == "twe":
            distance_params = {
                **distance_params,
                "window": 0.4,
            }
            average_params = {
                **average_params,
                "window": 0.4,
                "holdit_num_ts_to_use_percentage": 0.4,
            }
        if distance == "msm":
            distance_params = {
                **distance_params,
                "window": 0.5,
            }
            average_params = {
                **average_params,
                "window": 0.5,
                "holdit_num_ts_to_use_percentage": 0.5,
            }
        return HoldItKmeans(
            max_iter=300,
            n_init=10,
            init_algorithm="random",
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            averaging_method="ba",
            average_params={
                **average_params,
                "max_iters": 300,
            },
            verbose=True,
            **kwargs,
        )
    raise ValueError(f"UNKNOWN CLUSTERER: {c} in set_clusterer")


def _set_clusterer_distance_based(
    c,
    random_state,
    n_jobs,
    fit_contract,
    checkpoint,
    data_vars,
    row_normalise,
    kwargs,
):
    if "init_algorithm" in kwargs:
        init_algorithm = kwargs["init_algorithm"]
    else:
        init_algorithm = "random"

    if "distance" in kwargs:
        distance = kwargs["distance"]
    else:
        distance = c.split("-")[-1]

    if distance not in DISTANCES_DICT:
        distance = "dtw"

    if "distance_params" in kwargs:
        distance_params = kwargs["distance_params"]
    else:
        distance_params = _get_distance_default_params(
            distance, data_vars, row_normalise
        )

    if "precomputed_distance_path" in kwargs:
        precomputed_distances = np.load(kwargs["precomputed_distance_path"])
    else:
        precomputed_distances = None

    if "kmeans" in c or "timeserieskmeans" in c:
        if "average_params" in kwargs:
            average_params = kwargs["average_params"]
        else:
            average_params = {"distance": distance, **distance_params.copy()}

        if "ssg" in c:
            # Sets to use subgradient BA
            average_params = {
                **average_params,
                "method": "subgradient",
            }
            return TimeSeriesKMeans(
                max_iter=50,
                n_init=10,
                init_algorithm=init_algorithm,
                distance=distance,
                distance_params=distance_params,
                random_state=random_state,
                averaging_method="ba",
                average_params=average_params,
                **kwargs,
            )
        elif "dba-soft_dtw" in c:
            average_params = {
                **average_params,
                "method": "soft_dba",
            }
            return TimeSeriesKMeans(
                max_iter=50,
                n_init=10,
                init_algorithm=init_algorithm,
                distance="soft_dtw",
                distance_params=distance_params,
                random_state=random_state,
                averaging_method="ba",
                average_params=average_params,
                **kwargs,
            )
        elif "ba" in c:
            return TimeSeriesKMeans(
                max_iter=50,
                n_init=10,
                init_algorithm=init_algorithm,
                distance=distance,
                distance_params=distance_params,
                random_state=random_state,
                averaging_method="ba",
                average_params=average_params,
                **kwargs,
            )
        else:
            return TimeSeriesKMeans(
                max_iter=50,
                n_init=10,
                init_algorithm=init_algorithm,
                distance=distance,
                distance_params=distance_params,
                random_state=random_state,
                averaging_method="mean",
                **kwargs,
            )
    elif "kmedoids" in c or "timeserieskmedoids" in c:
        return TimeSeriesKMedoids(
            max_iter=50,
            n_init=10,
            init_algorithm=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            method="alternate",
            **kwargs,
        )
    elif "pamsil" in c or "timeseriespamsil" in c:
        return KmedoidsPackage(
            max_iter=50,
            n_init=10,
            init_algorithm=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            method="pamsil",
            **kwargs,
        )
    elif "pammedsil" in c or "timeseriespammedsil" in c:
        return KmedoidsPackage(
            max_iter=50,
            n_init=10,
            init_algorithm=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            method="pammedsil",
            **kwargs,
        )
    elif "fasterpam" in c or "timeseriesfasterpam" in c:
        return KmedoidsPackage(
            max_iter=50,
            n_init=10,
            init_algorithm=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            method="fasterpam",
            **kwargs,
        )
    elif "pam" in c or "timeseriespam" in c:
        return TimeSeriesKMedoids(
            max_iter=50,
            n_init=10,
            init_algorithm=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            method="pam",
            **kwargs,
        )
    elif "clarans" in c or "timeseriesclarans" in c:
        return TimeSeriesCLARANS(
            n_init=10,
            init_algorithm=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            **kwargs,
        )
    elif "clara" in c or "timeseriesclara" in c:
        return TimeSeriesCLARA(
            max_iter=50,
            init_algorithm=init_algorithm,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            **kwargs,
        )
    elif "HDBSCAN" in c or "timeserieshdbscan" in c:
        return TimeSeriesHDBScan(
            min_cluster_size=5,
            min_samples=None,
            cluster_selection_epsilon=0.0,
            max_cluster_size=None,
            distance=distance,
            distance_params=distance_params,
            precomputed_distances=precomputed_distances,
            alpha=1.0,
            algorithm="auto",
            leaf_size=40,
            cluster_selection_method="eom",
            allow_single_cluster=False,
            store_centers=None,
            copy=False,
            **kwargs,
        )
    elif "DBSCAN" in c or "timeseriesdbscan" in c:
        return TimeSeriesDBScan(
            eps=0.5,
            min_samples=5,
            distance=distance,
            distance_params=distance_params,
            precomputed_distances=precomputed_distances,
            algorithm="auto",
            leaf_size=30,
            **kwargs,
        )
    elif "agglomerative" in c or "timeseriesagglomerative" in c:
        return TimeSeriesAgglomerative(
            distance=distance,
            distance_params=distance_params,
            precomputed_distances=precomputed_distances,
            memory=None,
            connectivity=None,
            compute_full_tree="auto",
            linkage="ward",
            distance_threshold=None,
            compute_distances=False,
            **kwargs,
        )
    elif "som" in c:
        return ElasticSOM(
            sigma=1.0,
            learning_rate=0.5,
            distance=distance,
            distance_params=distance_params,
            random_state=random_state,
            **kwargs,
        )
    elif "ksc" in c or "kspectral-centroid" in c or "kspectralcentroid" in c:
        return KSpectralCentroid(
            # Max shift set to n_timepoints when max_shift is None
            max_shift=None,
            max_iter=50,
            init_algorithm=init_algorithm,
            random_state=random_state,
            **kwargs,
        )
    elif "kshapes" in c or "timeserieskshapes" in c:
        return TimeSeriesKShapes(
            init_algorithm=init_algorithm,
            max_iter=50,
            n_init=10,
            tol=1e-06,
            random_state=random_state,
            **kwargs,
        )

    return None


def _get_distance_default_params(
    dist_name: str, data_vars: list, row_normalise: bool
) -> dict:
    if dist_name == "dtw" or dist_name == "ddtw":
        return {"window": 0.2}
    if dist_name == "lcss":
        return {"epsilon": 1.0}
    if dist_name == "erp":
        # load dataset to get std if available
        if data_vars is not None:
            X_train, _, _, _, _ = load_experiment_data(*data_vars)

            # cant handle unequal length series
            if isinstance(X_train, np.ndarray):
                if row_normalise:
                    scaler = TimeSeriesScaler()
                    X_train = scaler.fit_transform(X_train)

                return {"g": X_train.std(axis=0).sum()}
            elif not isinstance(X_train, list):
                raise ValueError("Unknown data type in _get_distance_default_params")
        return {"g": 0.05}
    if dist_name == "msm":
        return {"c": 1.0, "independent": True}
    if dist_name == "edr":
        return {"epsilon": None}
    if dist_name == "twe":
        return {"nu": 0.001, "lmbda": 1.0}
    if dist_name == "psi_dtw":
        return {"r": 0.5}
    if dist_name == "adtw":
        return {"warp_penalty": 1.0}
    if dist_name == "shape_dtw":
        return {"descriptor": "identity", "reach": 30}
    return {}


def _set_clusterer_feature_based(
    c, random_state, n_jobs, fit_contract, checkpoint, kwargs
):
    if c == "catch22" or c == "catch22clusterer":
        from aeon.clustering.feature_based import Catch22Clusterer

        return Catch22Clusterer(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "tsfresh" or c == "tsfreshclusterer":
        from aeon.clustering.feature_based import TSFreshClusterer

        return TSFreshClusterer(random_state=random_state, n_jobs=n_jobs, **kwargs)
    elif c == "summary" or c == "summaryclusterer":
        from aeon.clustering.feature_based import SummaryClusterer

        return SummaryClusterer(random_state=random_state, n_jobs=n_jobs, **kwargs)


def _set_clusterer_other(c, random_state, n_jobs, fit_contract, checkpoint, kwargs):
    if c == "dummyclusterer" or c == "dummy" or c == "dummyclusterer-tsml":
        from tsml.dummy import DummyClusterer

        return DummyClusterer(
            strategy="random", n_clusters=1, random_state=random_state, **kwargs
        )
    elif c == "dummyclusterer-aeon":
        return TimeSeriesKMeans(
            n_clusters=1,
            n_init=1,
            init_algorithm="random",
            distance="euclidean",
            max_iter=1,
            random_state=random_state,
            **kwargs,
        )
    elif c == "dummyclusterer-sklearn":
        return KMeans(
            n_clusters=1,
            n_init=1,
            init="random",
            max_iter=1,
            random_state=random_state,
            **kwargs,
        )


def _set_clusterer_vector(c, random_state, n_jobs, fit_contract, checkpoint, kwargs):
    if c == "kmeans" or c == "kmeans-sklearn":
        from sklearn.cluster import KMeans

        return KMeans(random_state=random_state, **kwargs)
    elif c == "dbscan":
        from sklearn.cluster import DBSCAN

        return DBSCAN(n_jobs=n_jobs, **kwargs)
