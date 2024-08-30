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
    init_experiment = [
        "greedy-kmeans++",
        "random-init",
        "forgy-init",
        "random-init-10-restarts",
        "forgy-init-10-restarts",
        "forgy-init-10-restarts-average-number-iterations",
    ]
    if c in init_experiment:
        if c == "greedy-kmeans++":
            return TimeSeriesKMeans(
                max_iter=50,
                n_init=1,
                init_algorithm="kmeans++",
                distance="squared",
                random_state=random_state,
                averaging_method="mean",
                **kwargs,
            )
        elif "forgy-init" == c:
            return TimeSeriesKMeans(
                max_iter=50,
                n_init=1,
                init_algorithm="random",
                distance="squared",
                random_state=random_state,
                averaging_method="mean",
                **kwargs,
            )
        elif "forgy-init-10-restarts" == c:
            return TimeSeriesKMeans(
                max_iter=50,
                n_init=10,
                init_algorithm="random",
                distance="squared",
                random_state=random_state,
                averaging_method="mean",
                **kwargs,
            )
        elif "random-init" == c:
            return TimeSeriesKMeans(
                max_iter=50,
                n_init=1,
                init_algorithm="random_old",
                distance="squared",
                random_state=random_state,
                averaging_method="mean",
                **kwargs,
            )
        elif "random-init-10-restarts" == c:
            return TimeSeriesKMeans(
                max_iter=50,
                n_init=10,
                init_algorithm="random_old",
                distance="squared",
                random_state=random_state,
                averaging_method="mean",
                **kwargs,
            )
        elif "forgy-init-10-restarts-average-number-iterations" == c:
            return TimeSeriesKMeans(
                max_iter=300,
                n_init=10,
                init_algorithm="random",
                distance="squared",
                random_state=random_state,
                averaging_method="mean",
                **kwargs,
            )

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

    kmeans_full_window = [
        "k-means-full-window-dtw",
        "k-means-full-window-ddtw",
        "k-means-ba-full-window-dtw",
        "k-means-ba-full-window-ddtw",
    ]
    if c in kmeans_full_window:

        if "k-means" in c:

            if "ssg" in c:
                # Sets to use subgradient BA
                average_params = {
                    "method": "subgradient",
                }
                return TimeSeriesKMeans(
                    max_iter=50,
                    n_init=10,
                    init_algorithm=init_algorithm,
                    distance=distance,
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
                    random_state=random_state,
                    averaging_method="ba",
                    **kwargs,
                )
            else:
                return TimeSeriesKMeans(
                    max_iter=50,
                    n_init=10,
                    init_algorithm=init_algorithm,
                    distance=distance,
                    random_state=random_state,
                    averaging_method="mean",
                    **kwargs,
                )

    if "window" in c:
        distance_params = {**distance_params, "window": 0.2}
    average_params = {"distance": distance, **distance_params.copy()}

    if "faster" in c:
        average_params = {
            **average_params,
            "method": "holdit",
        }
    elif "proper-stopping" in c:
        average_params = {**average_params, "method": "holdit_stopping"}
    elif "approx-stopping" in c:
        average_params = {
            **average_params,
            "method": "holdit_stopping_approx",
        }
    elif "avg-change-stopping" in c:
        average_params = {
            **average_params,
            "method": "holdit_stopping_avg_change",
        }
    else:
        average_params = {
            **average_params,
            "method": "subgradient",
        }

    potential_size_arg = ["50", "40", "30", "20", "10"]
    if any(arg in c for arg in potential_size_arg):
        size = int(c.split("-")[0])
        average_params = {
            **average_params,
            "holdit_num_ts_to_use_percentage": size / 100,
        }

    return TimeSeriesKMeans(
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
