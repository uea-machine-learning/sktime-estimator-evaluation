import os
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from aeon.transformations.collection import TimeSeriesScaler
from tsml.datasets import load_from_ts_file

from tsml_eval.estimators.clustering.consensus import (
    CSPAFromFile,
    ElasticEnsembleClustererFromFile,
    FromFileIterativeVotingClustering,
    FromFileSimpleVote,
    HBGFFromFile,
    HGPAFromFile,
    MCLAFromFile,
    NMFFromFile,
)
from tsml_eval.estimators.clustering.consensus.experiment_utils import (
    create_symlink_temp_experiment,
    get_dataset_list_for_model_dir,
    get_output_path,
)
from tsml_eval.experiments import run_clustering_experiment
from tsml_eval.utils.datasets import load_experiment_data

DATASET_PATH = "/home/chris/Documents/Univariate_ts"


def _get_model(ensemble_model_name: str, clusterers: list[str]):
    """Get the ensemble model from the name

    Parameters
    ----------
    ensemble_model_name : str
        The name of the ensemble model to get
    clusterers : list[str]
        The list of clusterers to use

    Returns
    -------
    BaseEstimator
        The ensemble model
    """
    if "iterative-voting" in ensemble_model_name:
        return FromFileIterativeVotingClustering(clusterers=clusterers, random_state=0)
    elif "simple-voting" in ensemble_model_name:
        return FromFileSimpleVote(clusterers=clusterers, random_state=0)
    elif "cspa" in ensemble_model_name:
        return CSPAFromFile(clusterers=clusterers, random_state=0)
    elif "hgpa" in ensemble_model_name:
        return HGPAFromFile(clusterers=clusterers, random_state=0)
    elif "mcla" in ensemble_model_name:
        return MCLAFromFile(clusterers=clusterers, random_state=0)
    elif "hbgf" in ensemble_model_name:
        return HBGFFromFile(clusterers=clusterers, random_state=0)
    elif "nmf" in ensemble_model_name:
        return NMFFromFile(clusterers=clusterers, random_state=0)
    elif "EE-davies-bouldin-twe" in ensemble_model_name:
        distance_params = {"nu": 0.001, "lmbda": 1.0}
        return ElasticEnsembleClustererFromFile(
            clusterers=clusterers,
            random_state=0,
            evaluation_metric="davies_bouldin_score",
            distances_to_average_over="twe",
            distances_to_average_over_params=distance_params,
        )
    elif "EE-davies-bouldin-euclidean" in ensemble_model_name:
        return ElasticEnsembleClustererFromFile(
            clusterers=clusterers,
            random_state=0,
            evaluation_metric="davies_bouldin_score",
            distances_to_average_over="euclidean",
        )
    elif "EE-davies-bouldin-twe" in ensemble_model_name:
        distance_params = {"nu": 0.001, "lmbda": 1.0, "window": 0.5}
        return ElasticEnsembleClustererFromFile(
            clusterers=clusterers,
            random_state=0,
            evaluation_metric="davies_bouldin_score",
            distances_to_average_over="twe",
            distances_to_average_over_params=distance_params,
        )
    elif "EE-calinski-harabasz-euclidean" in ensemble_model_name:
        return ElasticEnsembleClustererFromFile(
            clusterers=clusterers,
            random_state=0,
            evaluation_metric="calinski_harabasz_score",
            distances_to_average_over="euclidean",
        )
    elif "EE-calinski-harabasz-twe" in ensemble_model_name:
        distance_params = {"nu": 0.001, "lmbda": 1.0, "window": 0.5}
        return ElasticEnsembleClustererFromFile(
            clusterers=clusterers,
            random_state=0,
            evaluation_metric="calinski_harabasz_score",
            distances_to_average_over="twe",
            distances_to_average_over_params=distance_params,
        )
    elif "EE-calinski-harabasz-msm" in ensemble_model_name:
        distance_params = {"c": 1.0, "independent": True, "window": 0.5}
        return ElasticEnsembleClustererFromFile(
            clusterers=clusterers,
            random_state=0,
            evaluation_metric="calinski_harabasz_score",
            distances_to_average_over="twe",
            distances_to_average_over_params=distance_params,
        )
    elif "EE-davies-bouldin-msm" in ensemble_model_name:
        distance_params = {"c": 1.0, "independent": True, "window": 0.5}
        return ElasticEnsembleClustererFromFile(
            clusterers=clusterers,
            random_state=0,
            evaluation_metric="davies_bouldin_score",
            distances_to_average_over="twe",
            distances_to_average_over_params=distance_params,
        )
    else:
        raise ValueError(f"Unknown ensemble model: {ensemble_model_name}")


def process_dataset(
    dataset: str,
    result_path: str,
    model_names: list[str],
    err_datasets: list[str],
    ensemble_model_name: str,
    result_model_name: str,
    clustering_results_dir_name: str,
    count: int = 0,
    total: int = 0,
    test_train_split: bool = True,
    normalise: bool = True,
):
    """Process a dataset.

    This logic has been separated out so that it can be run in parallel with the
    ThreadPoolExecutor.

    Parameters
    ----------
    dataset : str
        The name of the dataset to process
    model_names : list[str]
        The list of model names to use
    err_datasets : list[str]
        The list of datasets that have errors. As this function is used it thread
        it is passed here
    ensemble_model_name : str
        The ensemble model to use. This can be any tsml support. See function _get_model
        above for the list of supported models.
    result_model_name : str
        The name of the model to use. This is the name of the model that will be saved
        in the results directory. For example if you are only using 3 pam distances you
        may call it pam-top-3. If you are using all the pam distances you may call it
        pam-all.
    clustering_results_dir_name : str
        The directory of clustering results to use.
        Name of directory results are in so can be "pam" or "kmeans". But if you create
        a custom one for example with 2 of pam and 3 of kmeans, then whatever the dir
        name they sit in is what you should use here.
    count : int
        The count of the dataset. This is used for logging.
    total : int
        The total number of datasets. This is used for logging.
    """
    resample_id = 0  # Move to param later
    build_test_file = True
    build_train_file = True
    if os.path.exists(
        f"{result_path}/{ensemble_model_name}/{result_model_name}/Predictions/{dataset}/testResample{resample_id}.csv"
    ):
        build_test_file = False
    if os.path.exists(
        f"{result_path}/{ensemble_model_name}/{result_model_name}/Predictions/{dataset}/trainResample{resample_id}.csv"
    ):
        build_train_file = False

    if not build_test_file and not build_train_file:
        print(f"Skipping {dataset} results as they already exists")
        return
    clusterers = [
        f"{result_path}/{clustering_results_dir_name}/{model_name}/Predictions/{dataset}/"
        for model_name in model_names
    ]
    print(
        f"Running {count}/{total}: {dataset} with {ensemble_model_name} and {result_model_name}"
    )

    c = _get_model(ensemble_model_name, clusterers)

    X_train, y_train, X_test, y_test, resample = load_experiment_data(
        DATASET_PATH, dataset, 0, False
    )

    # Normalise
    if normalise:
        scaler = TimeSeriesScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    if not test_train_split:
        y_train = np.concatenate((y_train, y_test), axis=None)
        X_train = (
            np.concatenate([X_train, X_test], axis=0)
            if isinstance(X_train, np.ndarray)
            else X_train + X_test
        )
        X_test = None
        y_test = None
        build_test_file = False

    try:
        run_clustering_experiment(
            X_train,
            y_train,
            c,
            f"{result_path}/{ensemble_model_name}",
            X_test=X_test,
            y_test=y_test,
            n_clusters=-1,
            dataset_name=dataset,
            resample_id=0,
            build_test_file=build_test_file,
            build_train_file=build_train_file,
            clusterer_name=result_model_name,
        )
    except ValueError as e:
        print(f"======== Error in {dataset} ========")
        err_datasets.append(dataset)
        print(e)
    print(
        f"Finished {count}/{total}: {dataset} with {ensemble_model_name} and {result_model_name}"
    )
    return err_datasets


def run_experiment_for_model(
    clustering_results_dir_name: str,
    result_path: str,
    ensemble_model_name: str,
    result_model_name: str,
    thread=False,
    test_train_split: bool = True,
):
    """Run an ensemble experiment with a directory of results

    Parameters
    ----------
    clustering_results_dir_name : str
        The directory of clustering results to use.
        Name of directory results are in so can be "pam" or "kmeans". But if you create
        a custom one for example with 2 of pam and 3 of kmeans, then whatever the dir
        name they sit in is what you should use here.
    ensemble_model_name : str
        The ensemble model to use. This can be any tsml support. See function _get_model
        above for the list of supported models.
    result_model_name : str
        The name of the model to use. This is the name of the model that will be saved
        in the results directory. For example if you are only using 3 pam distances you
        may call it pam-top-3. If you are using all the pam distances you may call it
        pam-all.
    """
    model_path = f"{result_path}/{clustering_results_dir_name}"
    valid_datasets, model_names, missing = get_dataset_list_for_model_dir(
        model_path, test_train_split
    )
    
    valid_datasets = ['ACSF1', 'ArrowHead', 'BME', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car', 'Chinatown', 'ChlorineConcentration', 'CinCECGTorso', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'ECG200', 'ECG5000', 'ECGFiveDays', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'Earthquakes', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain', 'GunPoint', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'MedicalImages', 'MiddlePhalanxOutlineCorrect', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'MoteStrain', 'NonInvasiveFetalECGThorax2', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineCorrect', 'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'Wafer', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']

    if ensemble_model_name not in result_model_name:
        result_model_name = f"{ensemble_model_name}-{result_model_name}"

    # Check if results exist and raise error so we don't overwrite
    err_datasets = []
    num_datasets = len(valid_datasets)

    # For testing
    if thread:
        dataset_queue = queue.Queue()
        for i, dataset in enumerate(valid_datasets):
            dataset_queue.put((i, dataset))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {}

            while not dataset_queue.empty() or futures:
                while not dataset_queue.empty() and len(futures) < 20:
                    i, dataset = dataset_queue.get()
                    future = executor.submit(
                        process_dataset,
                        dataset,
                        result_path,
                        model_names,
                        err_datasets,
                        ensemble_model_name,
                        result_model_name,
                        clustering_results_dir_name,
                        i,
                        num_datasets,
                        test_train_split,
                    )
                    futures[future] = dataset

                for future in as_completed(futures):
                    dataset = futures.pop(future)
                    try:
                        err_datasets = future.result()
                    except Exception as e:
                        print(f"Exception for {dataset}: {e}")
    else:
        for i, dataset in enumerate(valid_datasets):
            process_dataset(
                dataset,
                result_path,
                model_names,
                err_datasets,
                ensemble_model_name,
                result_model_name,
                clustering_results_dir_name,
                i,
                num_datasets,
                test_train_split,
            )

    print("All datasets processed")
    print("Error datasets:", err_datasets)


import shutil

if __name__ == "__main__":
    # All the supported ensemble models
    ensemble_models = [
        # "simple-voting",
        # "iterative-voting",
        # "cspa",
        # "mcla",
        # "hbgf",
        # "nmf",
        # "elastic-ensemble"
        # "EE-davies-bouldin-euclidean",
        "EE-davies-bouldin-msm",
        # "EE-davies-bouldin-twe"
        # "EE-davies-bouldin-msm",
        # "EE-calinski-harabasz-euclidean"
        "EE-calinski-harabasz-msm"
        # "EE-calinski-harabasz-msm"
    ]

    # Name of various configurations. So pam-all I use when clustering_result_dir_name
    # is pam to say I want to use all the models in the pam directory. pam-top-5 would
    # be the top 5 models in the pam directory. pam-top-3 would be the top 3 models in
    # the pam directory. But I have to specify which models are the top 5 or top 3 in
    # the specific_models dictionary.
    specific_models = {
        # "pam-all": [],
        # "pam-top-5": ["pam-twe", "pam-msm", "pam-adtw", "pam-soft-dtw", "pam-shape-dtw"],
        # "pam-top-3": ["pam-twe", "pam-msm", "pam-adtw"],
        # "k-means-ba-all": [],
        # "k-means-ba-top-5": ["kmeans-ba-twe", "kmeans-ba-msm", "kmeans-ba-adtw", "kmeans-ba-soft-dtw", "kmeans-ba-shape-dtw"],
        # "mixed": ["pam-twe", "pam-msm", "pam-adtw", "pam-soft-dtw", "pam-shape-dtw", "kmeans-ba-twe", "kmeans-ba-msm", "kmeans-ba-adtw", "kmeans-ba-soft-dtw", "kmeans-ba-shape-dtw"],
        # "k-means-ba": ["k-means-ba-dtw", "k-means-ba-msm", "k-means-ba-twe", "k-means-ba-erp", "k-means-ba-wdtw", "k-means-ba-adtw"],
        "pam": [
            "pam-dtw",
            "pam-msm",
            "pam-twe",
            "pam-erp",
            "pam-wdtw",
            "pam-adtw",
            "pam-ddtw",
            "pam-wddtw",
            "pam-edr",
            "pam-lcss",
        ]
    }
    model_name = "pam"

    # models_to_use = ["k-means-ba-dtw", "k-means-ba-msm", "k-means-ba-twe", "k-means-ba-erp", "k-means-ba-wdtw", "k-means-ba-adtw", "k-means-euclidean"],
    RESULT_PATH = "/home/chris/Documents/phd-results/31-aug-results/normalised"
    for test_train_split in [False]:
        if not test_train_split:
            result_path = f"{RESULT_PATH}/combine-test-train-split"
        else:
            result_path = f"{RESULT_PATH}/test-train-split"

        TEMP_EXPERIMENT_PATH = os.path.join(result_path, "temp-ensemble-experiment")
        # check if exists and delete
        if os.path.exists(TEMP_EXPERIMENT_PATH):
            shutil.rmtree(TEMP_EXPERIMENT_PATH)
        create_symlink_temp_experiment(
            TEMP_EXPERIMENT_PATH, result_path, specific_models
        )

        for ensemble_model_name in ensemble_models:
            print(f"Running {ensemble_model_name} with {model_name}")
            # Check if it exists

            run_experiment_for_model(
                clustering_results_dir_name="temp-ensemble-experiment",
                result_path=result_path,
                ensemble_model_name=ensemble_model_name,
                result_model_name=model_name,
                thread=True,
                test_train_split=test_train_split,
            )
        shutil.rmtree(TEMP_EXPERIMENT_PATH)
