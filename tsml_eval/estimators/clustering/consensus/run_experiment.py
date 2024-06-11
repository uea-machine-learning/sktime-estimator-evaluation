from aeon.datasets import tsc_datasets
from tsml.datasets import load_from_ts_file

from tsml_eval.estimators.clustering.consensus.ivc_from_file import \
    FromFileIterativeVotingClustering
from tsml_eval.experiments import run_clustering_experiment
from tsml_eval.utils.experiments import _results_present

PATH_TO_TSCL_RESULTS = "/home/chris/projects/tscl-results"
PATH_TO_RESULTS = f"{PATH_TO_TSCL_RESULTS}/results/test-train-results/normalised/PAM"
SKIP_DATASETS = ["Adiac", "OliveOil", "PigAirwayPressure", "PigArtPressure", "PigCVP"]

DATASET_PATH = "/home/chris/Documents/Univariate_ts"
RESULT_PATH = "/home/chris/Documents/phd/ensemble-results/normalised/PAM"

if __name__ == "__main__":
    datasets = sorted(list(tsc_datasets.univariate_equal_length))
    distances = [
        "adtw",
        "ddtw",
        "dtw",
        "edr",
        "erp",
        "lcss",
        "msm",
        "euclidean",
        "twe",
        "wddtw",
        "wdtw",
    ]
    err_datasets = []
    for val in SKIP_DATASETS:
        err_datasets.append(val)

    for dataset in datasets:
        clusterers = [
            f"{PATH_TO_RESULTS}/pam-{distance}/Predictions/{dataset}/" for distance in distances
        ]

        print(f"======== Running {dataset} ========")
        c = FromFileIterativeVotingClustering(clusterers=clusterers, random_state=0)
        # c = FromFileSimpleVote(clusterers=clusterers, random_state=0)

        if dataset in SKIP_DATASETS or _results_present(
            PATH_TO_RESULTS,
            type(c).__name__,
            dataset,
            resample_id=0,
            split="BOTH",
        ):
            continue

        X_train, y_train = load_from_ts_file(f"{DATASET_PATH}/{dataset}/{dataset}_TRAIN.ts")
        X_test, y_test = load_from_ts_file(f"{DATASET_PATH}/{dataset}/{dataset}_TEST.ts")
        try:
            run_clustering_experiment(X_train, y_train, c, RESULT_PATH, X_test=X_test, y_test=y_test, n_clusters=-1, dataset_name=dataset, resample_id=0, build_test_file=True)
        except ValueError as e:
            print(f"======== Error in {dataset} ========")
            err_datasets.append(dataset)
            print(e)
            continue

        print(f"======== Finished {dataset} ========")
        print(err_datasets)