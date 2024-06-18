import os
from aeon.datasets import tsc_datasets


def check_results(result_path: str, expected_dataset_list: list,
                  resample: int = 0) -> dict:
    files = os.listdir(result_path)

    present_test_datasets = {}
    present_train_datasets = {}

    missing_datasets = {}

    # Get all the datasets
    models = [file.split('_')[0] for file in files]
    for model in models:
        missing_datasets[model] = set()
        present_test_datasets[model] = expected_dataset_list.copy()
        present_train_datasets[model] = expected_dataset_list.copy()
        model_path = os.path.join(result_path, model, "Predictions")

        for dataset in expected_dataset_list:
            dataset_path = os.path.join(model_path, dataset)

            try:
                dataset_results = os.listdir(dataset_path)
            except FileNotFoundError:
                missing_datasets[model].add(dataset)
                try:
                    present_train_datasets[model].remove(dataset)
                except ValueError:
                    pass
                try:
                    present_test_datasets[model].remove(dataset)
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

            if not train_result_present:
                missing_datasets[model].add(dataset)
                try:
                    present_test_datasets[model].remove(dataset)
                except ValueError:
                    pass
            if not test_result_present:
                try:
                    present_train_datasets[model].remove(dataset)
                except ValueError:
                    pass

    # Asset present and missing datasets dont overlap
    for model in models:
        assert len(
            missing_datasets[model].intersection(present_test_datasets[model])) == 0
        assert len(
            missing_datasets[model].intersection(present_train_datasets[model])) == 0

    if isinstance(present_test_datasets, str):
        present_test_datasets = [present_test_datasets]
    if isinstance(present_train_datasets, str):
        present_train_datasets = [present_train_datasets]
    if isinstance(missing_datasets, str):
        missing_datasets = [missing_datasets]

    return {
        "present_train": present_train_datasets,
        "present_test": present_test_datasets,
        "missing": missing_datasets
    }


def get_dataset_list_for_model_dir(model_dir: str):
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
        model_dir, tsc_datasets.univariate_equal_length, resample=0
    )["missing"]
    missing_copy = missing.copy()

    missing_set = set()

    for key in missing.keys():
        for dataset in missing[key]:
            missing_set.add(dataset)

    full_dataset_set = set(tsc_datasets.univariate_equal_length)
    for missing in missing_set:
        full_dataset_set.remove(missing)

    return list(full_dataset_set), list(missing_copy.keys())
