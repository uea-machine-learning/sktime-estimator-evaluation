import argparse
import os
import sys

import numpy as np
from aeon.transformations.collection import TimeSeriesScaler

from tsml_eval.estimators.clustering.consensus.run_experiment import _get_model
from tsml_eval.experiments import run_clustering_experiment
from tsml_eval.utils.datasets import load_experiment_data

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model on the dataset")
    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument(
        "running_on_cluster",
        type=lambda x: (str(x).lower() == "true"),
        help="Whether the job runs on the cluster (true/false)",
    )
    parser.add_argument(
        "combine_test_train_split",
        type=lambda x: (str(x).lower() == "true"),
        help="Whether to combine the test and train split (true/false)",
    )
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    running_on_cluster = args.running_on_cluster
    combine_test_train_split = args.combine_test_train_split
    # model_name = "EE-calinski-harabasz-euclidean"
    # dataset_name = "ACSF1"
    # running_on_cluster = True
    # combine_test_train_split = True
    print(f"Running {model_name} with {dataset_name}")
    print(f"Running on cluster {running_on_cluster}")
    print(f"Combining test and train split {combine_test_train_split}")

    dataset_path = "/home/chris/Documents/Univariate_ts"
    result_path = "/home/chris/Documents/phd-results/ee-test-results/temp-ee-results"
    if running_on_cluster:
        dataset_path = "/gpfs/home/ajb/Data"
        result_path = "/gpfs/home/eej17ucu/ee-test-results/temp-ee-results"

    if combine_test_train_split:
        result_path += "/combine-test-train-split"
    else:
        result_path += "/test-train-split"

    if running_on_cluster:
        models_to_ensemble_path = f"{result_path}/pam"
    else:
        models_to_ensemble_path = os.path.join(result_path, "pam")

    print(
        f"Running {model_name} with {models_to_ensemble_path}.\nResult output path: {result_path}"
    )
    pam_models_to_use = [
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
    err_datasets = []

    result_output_path = f"{result_path}/{model_name}"

    resample_id = 0  # Move to param later
    build_test_file = True
    build_train_file = True
    if os.path.exists(
        f"{result_path}/{model_name}/Predictions/{dataset_name}/testResample{resample_id}.csv"
    ):
        build_test_file = False
    if os.path.exists(
        f"{result_path}/{model_name}/Predictions/{dataset_name}/trainResample{resample_id}.csv"
    ):
        build_train_file = False

    if not build_test_file and not build_train_file:
        print(f"Skipping {dataset_name} results as they already exists")
        sys.exit(0)

    clusterers = [
        f"{models_to_ensemble_path}/{name}/Predictions/{dataset_name}/"
        for name in pam_models_to_use
    ]

    print(clusterers)

    c = _get_model(model_name, clusterers)

    X_train, y_train, X_test, y_test, resample = load_experiment_data(
        dataset_path, dataset_name, 0, False
    )

    # Normalise
    scaler = TimeSeriesScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    if combine_test_train_split:
        y_train = np.concatenate((y_train, y_test), axis=None)
        X_train = (
            np.concatenate([X_train, X_test], axis=0)
            if isinstance(X_train, np.ndarray)
            else X_train + X_test
        )
        X_test = None
        y_test = None
        build_test_file = False

    run_clustering_experiment(
        X_train,
        y_train,
        c,
        f"{result_path}/{model_name}",
        X_test=X_test,
        y_test=y_test,
        n_clusters=-1,
        dataset_name=dataset_name,
        resample_id=0,
        build_test_file=build_test_file,
        build_train_file=build_train_file,
        clusterer_name=model_name,
    )
