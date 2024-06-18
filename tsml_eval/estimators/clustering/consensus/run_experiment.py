import os
from tsml.datasets import load_from_ts_file
from tsml_eval.estimators.clustering.consensus.ivc_from_file import FromFileIterativeVotingClustering
from tsml_eval.estimators.clustering.consensus.simple_vote_from_file import FromFileSimpleVote
from tsml_eval.experiments import run_clustering_experiment
from tsml_eval.estimators.clustering.consensus.experiment_utils import get_dataset_list_for_model_dir
from concurrent.futures import ThreadPoolExecutor, as_completed

RESULT_PATH = "/home/chris/Documents/phd-results/normalised"
MODEL = "pam"
DATASET_PATH = "/home/chris/Documents/Univariate_ts"

# ENSEMBLE_MODEL = "simple-voting"
ENSEMBLE_MODEL = "iterative-voting"
RESULT_MODEL_NAME = "pam-top-3"
USE_SPECIFIC_MODELS = ["pam-twe", "pam-msm", "pam-adtw"]


def process_dataset(dataset, model_names, err_datasets):
    clusterers = [f"{RESULT_PATH}/{MODEL}/{model_name}/Predictions/{dataset}/" for
                  model_name in model_names]

    print(f"======== Running {dataset} ========")
    if ENSEMBLE_MODEL == "iterative-voting":
        c = FromFileIterativeVotingClustering(clusterers=clusterers, random_state=0)
    elif ENSEMBLE_MODEL == "simple-voting":
        c = FromFileSimpleVote(clusterers=clusterers, random_state=0)
    else:
        raise ValueError(f"Unknown ensemble model: {ENSEMBLE_MODEL}")

    X_train, y_train = load_from_ts_file(f"{DATASET_PATH}/{dataset}/{dataset}_TRAIN.ts")
    X_test, y_test = load_from_ts_file(f"{DATASET_PATH}/{dataset}/{dataset}_TEST.ts")

    try:
        run_clustering_experiment(
            X_train,
            y_train,
            c,
            f"{RESULT_PATH}/{ENSEMBLE_MODEL}",
            X_test=X_test,
            y_test=y_test,
            n_clusters=-1,
            dataset_name=dataset,
            resample_id=0,
            build_test_file=True,
            clusterer_name=RESULT_MODEL_NAME,
        )
    except ValueError as e:
        # print(f"======== Error in {dataset} ========")
        err_datasets.append(dataset)
        print(e)
    # print(f"======== Finished {dataset} ========")
    return err_datasets

if __name__ == "__main__":
    model_path = f"{RESULT_PATH}/{MODEL}"
    valid_datasets, model_names = get_dataset_list_for_model_dir(model_path)

    if ENSEMBLE_MODEL not in RESULT_MODEL_NAME:
        RESULT_MODEL_NAME = f"{ENSEMBLE_MODEL}-{RESULT_MODEL_NAME}"

    if len(USE_SPECIFIC_MODELS) > 1:
        temp_model_names = [model_name for model_name in model_names if model_name in USE_SPECIFIC_MODELS]
        model_names = temp_model_names

    # Check if results exist and raise error so we don't overwrite
    if os.path.exists(f"{RESULT_PATH}/{ENSEMBLE_MODEL}/{RESULT_MODEL_NAME}"):
        raise ValueError(f"Results directory already exists: {RESULT_PATH}/{ENSEMBLE_MODEL}/{RESULT_MODEL_NAME}")

    err_datasets = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_dataset, dataset, model_names, err_datasets): dataset for dataset in valid_datasets}

        for future in as_completed(futures):
            dataset = futures[future]
            try:
                err_datasets = future.result()
            except Exception as e:
                print(f"Exception for {dataset}: {e}")

    print("All datasets processed")
    print("Error datasets:", err_datasets)