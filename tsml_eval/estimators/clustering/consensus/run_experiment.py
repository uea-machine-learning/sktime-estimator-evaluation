import os
from tsml.datasets import load_from_ts_file
from tsml_eval.experiments import run_clustering_experiment
from tsml_eval.estimators.clustering.consensus.experiment_utils import \
    get_dataset_list_for_model_dir
from concurrent.futures import ThreadPoolExecutor, as_completed
from tsml_eval.estimators.clustering.consensus import (
    FromFileSimpleVote,
    FromFileIterativeVotingClustering,
    CSPAFromFile,
    HGPAFromFile,
    MCLAFromFile,
    HBGFFromFile,
    NMFFromFile,
)
import queue

RESULT_PATH = "/home/chris/Documents/phd-results/normalised"
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
    else:
        raise ValueError(f"Unknown ensemble model: {ensemble_model_name}")

def process_dataset(
        dataset: str,
        model_names: list[str],
        err_datasets: list[str],
        ensemble_model_name: str,
        result_model_name: str,
        clustering_results_dir_name: str,
        count: int = 0,
        total: int = 0
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
    resample_id = 0 # Move to param later
    build_test_file = True
    build_train_file = True
    if os.path.exists(f"{RESULT_PATH}/{ensemble_model_name}/{result_model_name}/Predictions/{dataset}/testResample{resample_id}.csv"):
        build_test_file = False
    if os.path.exists(f"{RESULT_PATH}/{ensemble_model_name}/{result_model_name}/Predictions/{dataset}/trainResample{resample_id}.csv"):
        build_train_file = False

    if not build_test_file and not build_train_file:
        print(f"Skipping {dataset} results as they already exists")
        return
    clusterers = [
        f"{RESULT_PATH}/{clustering_results_dir_name}/{model_name}/Predictions/{dataset}/"
        for
        model_name in model_names]
    print(
        f"Running {count}/{total}: {dataset} with {ensemble_model_name} and {result_model_name}")

    c = _get_model(ensemble_model_name, clusterers)

    X_train, y_train = load_from_ts_file(f"{DATASET_PATH}/{dataset}/{dataset}_TRAIN.ts")
    X_test, y_test = load_from_ts_file(f"{DATASET_PATH}/{dataset}/{dataset}_TEST.ts")

    try:
        run_clustering_experiment(
            X_train,
            y_train,
            c,
            f"{RESULT_PATH}/{ensemble_model_name}",
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
        # print(f"======== Error in {dataset} ========")
        err_datasets.append(dataset)
        print(e)
    print(f"Finished {count}/{total}: {dataset} with {ensemble_model_name} and {result_model_name}")
    return err_datasets


def run_experiment_for_model(
        clustering_results_dir_name: str,
        ensemble_model_name: str,
        result_model_name: str,
        use_specific_models: list[str],
        thread = False
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
    use_specific_models : list[str]
        The list of specific models to use. This is used when you want to use only a
        subset of the models in the directory. For example if you have 5 models in the
        pam directory and you only want to use 3 of them, you would specify the names
        of the 3 models here.
    """
    model_path = f"{RESULT_PATH}/{clustering_results_dir_name}"
    valid_datasets, model_names = get_dataset_list_for_model_dir(model_path)

    if ensemble_model_name not in result_model_name:
        result_model_name = f"{ensemble_model_name}-{result_model_name}"

    if len(use_specific_models) > 1:
        temp_model_names = [model_name for model_name in model_names if
                            model_name in use_specific_models]
        model_names = temp_model_names

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
                        model_names,
                        err_datasets,
                        ensemble_model_name,
                        result_model_name,
                        clustering_results_dir_name,
                        i,
                        num_datasets
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
                model_names,
                err_datasets,
                ensemble_model_name,
                result_model_name,
                clustering_results_dir_name,
                i,
                num_datasets
            )


    print("All datasets processed")
    print("Error datasets:", err_datasets)


if __name__ == "__main__":
    # Name of directory results are in so can be "pam" or "kmeans". But if you create
    # a custom one for example with 2 of pam and 3 of kmeans, then whatever the dir
    # name they sit in is what you should use here.
    clustering_results_dir_name = "pam"

    # All the supported ensemble models
    ensemble_models = [
        # "simple-voting",
        # "iterative-voting",
        # "cspa",
        # "mcla",
        # "hbgf",
        # "nmf",
    ]

    # Name of various configurations. So pam-all I use when clustering_result_dir_name
    # is pam to say I want to use all the models in the pam directory. pam-top-5 would
    # be the top 5 models in the pam directory. pam-top-3 would be the top 3 models in
    # the pam directory. But I have to specify which models are the top 5 or top 3 in
    # the specific_models dictionary.
    specific_models = {
        # "pam-all": [],
        "pam-top-5": ["pam-twe", "pam-msm", "pam-adtw", "pam-edr", "pam-wdtw"],
        "pam-top-3": ["pam-twe", "pam-msm", "pam-adtw"],
    }

    for ensemble_model_name in ensemble_models:
        for result_model_name, use_specific_models in specific_models.items():
            print(f"Running {ensemble_model_name} with {result_model_name}")
            # Check if it exists

            run_experiment_for_model(
                clustering_results_dir_name=clustering_results_dir_name,
                ensemble_model_name=ensemble_model_name,
                result_model_name=result_model_name,
                use_specific_models=use_specific_models,
                thread=False
            )
