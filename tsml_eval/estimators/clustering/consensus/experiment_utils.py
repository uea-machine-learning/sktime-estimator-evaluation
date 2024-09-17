import os
from typing import Any, List, Set, Tuple

from aeon.datasets import tsc_datasets


def check_results(
    result_path: str,
    expected_dataset_list: list,
    resample: int = 0,
    test_train_split: bool = True,
) -> dict:
    files = os.listdir(result_path)

    present_test_datasets = {}
    present_train_datasets = {}

    missing_datasets = {}

    # Get all the datasets
    models = [file.split("_")[0] for file in files]
    # Remove any key that starts with a .
    models = [model for model in models if model[0] != "."]
    for model in models:
        model_path = os.path.join(result_path, model, "Predictions")
        missing, present_test, present_train = _get_datasets_for_specific_model(
            model_path, expected_dataset_list, resample, test_train_split
        )
        missing_datasets[model] = set(missing)
        present_test_datasets[model] = set(present_test)
        present_train_datasets[model] = set(present_train)

    # Asset present and missing datasets dont overlap
    # for model in models:
    #     assert len(missing_datasets[model].intersection(present_test_datasets[model])) == 0
    #     assert len(missing_datasets[model].intersection(present_train_datasets[model])) == 0

    if isinstance(present_test_datasets, str):
        present_test_datasets = [present_test_datasets]
    if isinstance(present_train_datasets, str):
        present_train_datasets = [present_train_datasets]
    if isinstance(missing_datasets, str):
        missing_datasets = [missing_datasets]

    new_dict_alphabetically_orders = {}
    for key in sorted(present_test_datasets.keys()):
        new_dict_alphabetically_orders[key] = present_test_datasets[key]

    return {
        "present_train": present_train_datasets,
        "present_test": present_test_datasets,
        "missing": missing_datasets,
    }


def _get_datasets_for_specific_model(
    model_path: str,
    expected_dataset_list: list,
    resample: int = 0,
    test_train_split: bool = True,
) -> tuple[set[str], list[str], list[str]]:

    # Check model_path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist")
    missing_datasets = set()

    present_test_datasets = expected_dataset_list.copy()
    present_train_datasets = expected_dataset_list.copy()

    for dataset in expected_dataset_list:
        dataset_path = os.path.join(model_path, dataset)

        try:
            dataset_results = os.listdir(dataset_path)
        except FileNotFoundError:
            missing_datasets.add(dataset)
            try:
                present_train_datasets.remove(dataset)
            except ValueError:
                pass
            try:
                present_test_datasets.remove(dataset)
            except ValueError:
                pass
            continue
        train_result_present = False
        test_result_present = False
        expected_train_path = f"testResample{resample}.csv"
        expected_test_path = f"trainResample{resample}.csv"
        for result in dataset_results:
            if result == expected_train_path:
                train_result_present = True
            if result == expected_test_path:
                test_result_present = True

        if not train_result_present and test_train_split:
            missing_datasets.add(dataset)
            try:
                present_test_datasets.remove(dataset)
            except ValueError:
                pass
        if not test_result_present:
            try:
                present_train_datasets.remove(dataset)
            except ValueError:
                pass
    return missing_datasets, present_test_datasets, present_train_datasets


def get_dataset_list_for_model_dir(model_dir: str, test_train_split: bool = True):
    """Checks the model directory for the datasets that have been completed.

    Parameters
    ----------
    model_dir : str
        Path to the model directory.

    Returns
    -------
    list
        List of datasets that have been completed.

    list
        List of model names
    """
    missing = check_results(
        model_dir,
        tsc_datasets.univariate_equal_length,
        resample=0,
        test_train_split=test_train_split,
    )["missing"]
    missing_copy = missing.copy()

    missing_set = set()

    for key in missing.keys():
        for dataset in missing[key]:
            missing_set.add(dataset)

    full_dataset_set = set(tsc_datasets.univariate_equal_length)
    for missing in missing_set:
        full_dataset_set.remove(missing)

    return list(full_dataset_set), list(missing_copy.keys()), missing_copy


if __name__ == "__main__":
    from aeon.datasets import tsc_datasets

    RESULT_PATH = "/Users/chris/Documents/phd-results/normalised"

    missing_results = {}
    for dir in os.listdir(RESULT_PATH):
        path = os.path.join(RESULT_PATH, dir)
        if os.path.isdir(path):
            temp = check_results(path, tsc_datasets.univariate_equal_length, resample=0)
            missing_results[dir] = temp["missing"]

    stop = ""
#
