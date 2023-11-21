"""Functions for evaluating multiple estimators on multiple datasets."""

import os
import pickle
from datetime import datetime

import numpy as np
from aeon.benchmarking import plot_critical_difference
from aeon.benchmarking.results_plotting import plot_scatter
from matplotlib import pyplot as plt

from tsml_eval.evaluation.storage import (
    ClassifierResults,
    ClustererResults,
    ForecasterResults,
    RegressorResults,
)
from tsml_eval.utils.functions import rank_array, time_to_milliseconds


def evaluate_classifiers(
    classifier_results,
    save_path,
    error_on_missing=True,
    eval_name=None,
    estimator_names=None,
):
    """
    Evaluate multiple classifiers on multiple datasets.

    Writes multiple csv files and figures to save_path, one for each statistic
    evaluated. Provides a summary csv file with the average statistic and
    average rank for each classifier.

    Parameters
    ----------
    classifier_results : list of ClassifierResults
        The results to evaluate.
    save_path : str
        The path to save the evaluation results to.
    error_on_missing : bool, default=True
        Whether to raise an error if results are missing.
    eval_name : str, default=None
        The name of the evaluation, used in save_path.
    estimator_names : list of str, default=None
        The names of the estimator for each classifier result. If None, uses
        the estimator_name attribute of each classifier result.
    """
    _evaluate_estimators(
        classifier_results,
        ClassifierResults.statistics,
        save_path,
        error_on_missing,
        eval_name,
        estimator_names,
    )


def evaluate_classifiers_from_file(
    load_paths,
    save_path,
    error_on_missing=True,
    eval_name=None,
    verify_results=True,
    estimator_names=None,
):
    """
    Evaluate multiple classifiers on multiple datasets from file.

    Writes multiple csv files and figures to save_path, one for each statistic
    evaluated. Provides a summary csv file with the average statistic and
    average rank for each classifier.

    Parameters
    ----------
    load_paths : list of str
        The paths to the classifier result files to evaluate.
    save_path : str
        The path to save the evaluation results to.
    error_on_missing : bool, default=True
        Whether to raise an error if results are missing.
    eval_name : str, default=None
        The name of the evaluation, used in save_path.
    verify_results : bool, default=True
        If the verification should be performed on the loaded results values.
    estimator_names : list of str, default=None
        The names of the estimator for each classifier result. If None, uses
        the estimator_name attribute of each classifier result.
    """
    classifier_results = []
    for load_path in load_paths:
        try:
            classifier_results.append(
                ClassifierResults().load_from_file(
                    load_path, verify_values=verify_results
                )
            )
        except FileNotFoundError:
            if error_on_missing:
                raise FileNotFoundError(f"Results for {load_path} not found.")

    evaluate_classifiers(
        classifier_results,
        save_path,
        error_on_missing=error_on_missing,
        eval_name=eval_name,
        estimator_names=estimator_names,
    )


def evaluate_classifiers_by_problem(
    load_path,
    classifier_names,
    dataset_names,
    save_path,
    resamples=None,
    load_train_results=False,
    error_on_missing=True,
    eval_name=None,
    verify_results=True,
):
    """
    Evaluate multiple classifiers on multiple datasets from file using standard paths.

    Finds files using classifier, dataset and resample names. It is expected the
    common tsml-eval file structure of
    {classifier}/Predictions/{dataset}/{split}Resample{resample}.csv is followed.

    Writes multiple csv files and figures to save_path, one for each statistic
    evaluated. Provides a summary csv file with the average statistic and
    average rank for each classifier.

    Parameters
    ----------
    load_path : str or list of str
        The path to the collection of classifier result files to evaluate.
        If load_path is a list, it will load results from each path in the list. It
        is expected that classifier_names and dataset_names are lists of lists with
        the same length as load_path.
    classifier_names : list of str, list of tuple or list of list
        The names of the classifiers to evaluate.
        A length 2 tuple containing strings can be used to specify a classifier name to
        load from in the first item and a classifier name to use in the evaluation
        results in the second.
        If load_path is a list, classifier_names must be a list of lists with the same
        length as load_path.
    dataset_names : list of str or list of list
        The names of the datasets to evaluate.
        If load_path is a list, dataset_names must be a list of lists with the same
        length as load_path.
    save_path : str
        The path to save the evaluation results to.
    resamples : int or list of int, default=None
        The resamples to evaluate. If int, evaluates resamples 0 to resamples-1.
    load_train_results : bool, default=False
        Whether to load train results as well as test results.
    error_on_missing : bool, default=True
        Whether to raise an error if results are missing.
    eval_name : str, default=None
        The name of the evaluation, used in save_path.
    verify_results : bool, default=True
        If the verification should be performed on the loaded results values.
    """
    load_path, classifier_names, dataset_names, resamples = _evaluate_by_problem_init(
        "classifier",
        load_path,
        classifier_names,
        dataset_names,
        resamples,
    )

    if load_train_results:
        splits = ["test", "train"]
    else:
        splits = ["test"]

    classifier_results = []
    estimator_eval_names = []
    names = []
    for i, path in enumerate(load_path):
        found_datasets = np.zeros(len(dataset_names[i]), dtype=bool)

        for classifier_name in classifier_names[i]:
            found_estimator = False

            if isinstance(classifier_name, tuple):
                classifier_eval_name = classifier_name[1]
                classifier_name = classifier_name[0]
            else:
                classifier_eval_name = classifier_name

            if classifier_eval_name not in estimator_eval_names:
                estimator_eval_names.append(classifier_eval_name)
            else:
                raise ValueError(
                    f"Duplicate evaluation name {classifier_eval_name} found."
                )

            for n, dataset_name in enumerate(dataset_names[i]):
                for resample in resamples:
                    for split in splits:
                        try:
                            classifier_results.append(
                                ClassifierResults().load_from_file(
                                    f"{path}/{classifier_name}/Predictions/"
                                    f"{dataset_name}/{split}Resample{resample}.csv",
                                    verify_values=verify_results,
                                )
                            )
                            names.append(classifier_eval_name)
                            found_estimator = True
                            found_datasets[n] = True
                        except FileNotFoundError:
                            if error_on_missing:
                                raise FileNotFoundError(
                                    f"Results for {classifier_eval_name} on "
                                    f"{dataset_name} {split} resample {resample} not "
                                    "found."
                                )

            if not found_estimator:
                print(f"Classifier {classifier_eval_name} not found.")  # noqa: T201

        missing_datasets = [
            dataset
            for dataset, found in zip(dataset_names[i], found_datasets)
            if not found
        ]
        if missing_datasets:
            msg = f"Files for datasets {missing_datasets} not found."
            if error_on_missing:
                raise FileNotFoundError(msg)
            else:
                print("\n\n" + msg)  # noqa: T201

    evaluate_classifiers(
        classifier_results,
        save_path,
        error_on_missing=error_on_missing,
        eval_name=eval_name,
        estimator_names=names,
    )


def evaluate_clusterers(
    clusterer_results,
    save_path,
    error_on_missing=True,
    eval_name=None,
    estimator_names=None,
):
    """
    Evaluate multiple clusterers on multiple datasets.

    Writes multiple csv files and figures to save_path, one for each statistic
    evaluated. Provides a summary csv file with the average statistic and
    average rank for each clusterer.

    Parameters
    ----------
    clusterer_results : list of ClustererResults
        The results to evaluate.
    save_path : str
        The path to save the evaluation results to.
    error_on_missing : bool, default=True
        Whether to raise an error if results are missing.
    eval_name : str, default=None
        The name of the evaluation, used in save_path.
    estimator_names : list of str, default=None
        The names of the estimator for each clusterer result. If None, uses
        the estimator_name attribute of each clusterer result.
    """
    _evaluate_estimators(
        clusterer_results,
        ClustererResults.statistics,
        save_path,
        error_on_missing,
        eval_name,
        estimator_names,
    )


def evaluate_clusterers_from_file(
    load_paths,
    save_path,
    error_on_missing=True,
    eval_name=None,
    verify_results=True,
    estimator_names=None,
):
    """
    Evaluate multiple clusterers on multiple datasets from file.

    Writes multiple csv files and figures to save_path, one for each statistic
    evaluated. Provides a summary csv file with the average statistic and
    average rank for each clusterer.

    Parameters
    ----------
    load_paths : list of str
        The paths to the clusterer result files to evaluate.
    save_path : str
        The path to save the evaluation results to.
    error_on_missing : bool, default=True
        Whether to raise an error if results are missing.
    eval_name : str, default=None
        The name of the evaluation, used in save_path.
    verify_results : bool, default=True
        If the verification should be performed on the loaded results values.
    estimator_names : list of str, default=None
        The names of the estimator for each clusterer result. If None, uses
        the estimator_name attribute of each clusterer result.
    """
    clusterer_results = []
    for load_path in load_paths:
        try:
            clusterer_results.append(
                ClustererResults().load_from_file(
                    load_path, verify_values=verify_results
                )
            )
        except FileNotFoundError:
            if error_on_missing:
                raise FileNotFoundError(f"Results for {load_path} not found.")

    evaluate_clusterers(
        clusterer_results,
        save_path,
        error_on_missing=error_on_missing,
        eval_name=eval_name,
        estimator_names=estimator_names,
    )


def evaluate_clusterers_by_problem(
    load_path,
    clusterer_names,
    dataset_names,
    save_path,
    resamples=None,
    load_test_results=True,
    error_on_missing=True,
    eval_name=None,
    verify_results=True,
):
    """
    Evaluate multiple clusterers on multiple datasets from file using standard paths.

    Finds files using clusterer, dataset and resample names. It is expected the
    common tsml-eval file structure of
    {clusterer}/Predictions/{dataset}/{split}Resample{resample}.csv is followed.

    Writes multiple csv files and figures to save_path, one for each statistic
    evaluated. Provides a summary csv file with the average statistic and
    average rank for each clusterer.

    Parameters
    ----------
    load_path : str or list of str
        The path to the collection of clusterer result files to evaluate.
        If load_path is a list, it will load results from each path in the list. It
        is expected that clusterer_names and dataset_names are lists of lists with
        the same length as load_path.
    clusterer_names : list of str, list of tuple or list of list
        The names of the clusterers to evaluate.
        A length 2 tuple containing strings can be used to specify a clusterer name to
        load from in the first item and a clusterer name to use in the evaluation
        results in the second.
        If load_path is a list, clusterer_names must be a list of lists with the same
        length as load_path.
    dataset_names : list of str or list of list
        The names of the datasets to evaluate.
        If load_path is a list, dataset_names must be a list of lists with the same
        length as load_path.
    save_path : str
        The path to save the evaluation results to.
    resamples : int or list of int, default=None
        The resamples to evaluate. If int, evaluates resamples 0 to resamples-1.
    load_test_results : bool, default=True
        Whether to load test results as well as train results.
    error_on_missing : bool, default=True
        Whether to raise an error if results are missing.
    eval_name : str, default=None
        The name of the evaluation, used in save_path.
    verify_results : bool, default=True
        If the verification should be performed on the loaded results values.
    """
    load_path, clusterer_names, dataset_names, resamples = _evaluate_by_problem_init(
        "clusterer",
        load_path,
        clusterer_names,
        dataset_names,
        resamples,
    )

    if load_test_results:
        splits = ["test", "train"]
    else:
        splits = ["train"]

    clusterer_results = []
    estimator_eval_names = []
    names = []
    for i, path in enumerate(load_path):
        found_datasets = np.zeros(len(dataset_names[i]), dtype=bool)

        for clusterer_name in clusterer_names[i]:
            found_estimator = False

            if isinstance(clusterer_name, tuple):
                clusterer_eval_name = clusterer_name[1]
                clusterer_name = clusterer_name[0]
            else:
                clusterer_eval_name = clusterer_name

            if clusterer_eval_name not in estimator_eval_names:
                estimator_eval_names.append(clusterer_eval_name)
            else:
                raise ValueError(
                    f"Duplicate evaluation name {clusterer_eval_name} found."
                )

            for n, dataset_name in enumerate(dataset_names[i]):
                for resample in resamples:
                    for split in splits:
                        try:
                            clusterer_results.append(
                                ClustererResults().load_from_file(
                                    f"{path}/{clusterer_name}/Predictions/"
                                    f"{dataset_name}/{split}Resample{resample}.csv",
                                    verify_values=verify_results,
                                )
                            )
                            names.append(clusterer_eval_name)
                            found_estimator = True
                            found_datasets[n] = True
                        except FileNotFoundError:
                            if error_on_missing:
                                raise FileNotFoundError(
                                    f"Results for {clusterer_eval_name} on "
                                    f"{dataset_name} {split} resample {resample} not "
                                    "found."
                                )

            if not found_estimator:
                print(f"Clusterer {clusterer_eval_name} not found.")  # noqa: T201

        missing_datasets = [
            dataset
            for dataset, found in zip(dataset_names[i], found_datasets)
            if not found
        ]
        if missing_datasets:
            msg = f"Files for datasets {missing_datasets} not found."
            if error_on_missing:
                raise FileNotFoundError(msg)
            else:
                print("\n\n" + msg)  # noqa: T201

    evaluate_clusterers(
        clusterer_results,
        save_path,
        error_on_missing=error_on_missing,
        eval_name=eval_name,
        estimator_names=names,
    )


def evaluate_regressors(
    regressor_results,
    save_path,
    error_on_missing=True,
    eval_name=None,
    estimator_names=None,
):
    """
    Evaluate multiple regressors on multiple datasets.

    Writes multiple csv files and figures to save_path, one for each statistic
    evaluated. Provides a summary csv file with the average statistic and
    average rank for each regressor.

    Parameters
    ----------
    regressor_results : list of RegressorResults
        The results to evaluate.
    save_path : str
        The path to save the evaluation results to.
    error_on_missing : bool, default=True
        Whether to raise an error if results are missing.
    eval_name : str, default=None
        The name of the evaluation, used in save_path.
    estimator_names : list of str, default=None
        The names of the estimator for each regressor result. If None, uses
        the estimator_name attribute of each regressor result.
    """
    _evaluate_estimators(
        regressor_results,
        RegressorResults.statistics,
        save_path,
        error_on_missing,
        eval_name,
        estimator_names,
    )


def evaluate_regressors_from_file(
    load_paths,
    save_path,
    error_on_missing=True,
    eval_name=None,
    verify_results=True,
    estimator_names=None,
):
    """
    Evaluate multiple regressors on multiple datasets from file.

    Writes multiple csv files and figures to save_path, one for each statistic
    evaluated. Provides a summary csv file with the average statistic and
    average rank for each regressor.

    Parameters
    ----------
    load_paths : list of str
        The paths to the regressor result files to evaluate.
    save_path : str
        The path to save the evaluation results to.
    error_on_missing : bool, default=True
        Whether to raise an error if results are missing.
    eval_name : str, default=None
        The name of the evaluation, used in save_path.
    verify_results : bool, default=True
        If the verification should be performed on the loaded results values.
    estimator_names : list of str, default=None
        The names of the estimator for each regressor result. If None, uses
        the estimator_name attribute of each regressor result.
    """
    regressor_results = []
    for load_path in load_paths:
        try:
            regressor_results.append(
                RegressorResults().load_from_file(
                    load_path, verify_values=verify_results
                )
            )
        except FileNotFoundError:
            if error_on_missing:
                raise FileNotFoundError(f"Results for {load_path} not found.")

    evaluate_regressors(
        regressor_results,
        save_path,
        error_on_missing=error_on_missing,
        eval_name=eval_name,
        estimator_names=estimator_names,
    )


def evaluate_regressors_by_problem(
    load_path,
    regressor_names,
    dataset_names,
    save_path,
    resamples=None,
    load_train_results=False,
    error_on_missing=True,
    eval_name=None,
    verify_results=True,
):
    """
    Evaluate multiple regressors on multiple datasets from file using standard paths.

    Finds files using regressor, dataset and resample names. It is expected the
    common tsml-eval file structure of
    {regressor}/Predictions/{dataset}/{split}Resample{resample}.csv is followed.

    Writes multiple csv files and figures to save_path, one for each statistic
    evaluated. Provides a summary csv file with the average statistic and
    average rank for each regressor.

    Parameters
    ----------
    load_path : str or list of str
        The path to the collection of regressor result files to evaluate.
        If load_path is a list, it will load results from each path in the list. It
        is expected that regressor_names and dataset_names are lists of lists with
        the same length as load_path.
    regressor_names : list of str, list of tuple or list of list
        The names of the regressors to evaluate.
        A length 2 tuple containing strings can be used to specify a regressor name to
        load from in the first item and a regressor name to use in the evaluation
        results in the second.
        If load_path is a list, regressor_names must be a list of lists with the same
        length as load_path.
    dataset_names : list of str or list of list
        The names of the datasets to evaluate.
        If load_path is a list, dataset_names must be a list of lists with the same
        length as load_path.
    save_path : str
        The path to save the evaluation results to.
    resamples : int or list of int, default=None
        The resamples to evaluate. If int, evaluates resamples 0 to resamples-1.
    load_train_results : bool, default=False
        Whether to load train results as well as test results.
    error_on_missing : bool, default=True
        Whether to raise an error if results are missing.
    eval_name : str, default=None
        The name of the evaluation, used in save_path.
    verify_results : bool, default=True
        If the verification should be performed on the loaded results values.
    """
    load_path, regressor_names, dataset_names, resamples = _evaluate_by_problem_init(
        "regressor",
        load_path,
        regressor_names,
        dataset_names,
        resamples,
    )

    if load_train_results:
        splits = ["test", "train"]
    else:
        splits = ["test"]

    regressor_results = []
    estimator_eval_names = []
    names = []
    for i, path in enumerate(load_path):
        found_datasets = np.zeros(len(dataset_names[i]), dtype=bool)

        for regressor_name in regressor_names[i]:
            found_estimator = False

            if isinstance(regressor_name, tuple):
                regressor_eval_name = regressor_name[1]
                regressor_name = regressor_name[0]
            else:
                regressor_eval_name = regressor_name

            if regressor_eval_name not in estimator_eval_names:
                estimator_eval_names.append(regressor_eval_name)
            else:
                raise ValueError(
                    f"Duplicate evaluation name {regressor_eval_name} found."
                )

            for n, dataset_name in enumerate(dataset_names[i]):
                for resample in resamples:
                    for split in splits:
                        try:
                            regressor_results.append(
                                RegressorResults().load_from_file(
                                    f"{path}/{regressor_name}/Predictions/"
                                    f"{dataset_name}/{split}Resample{resample}.csv",
                                    verify_values=verify_results,
                                )
                            )
                            names.append(regressor_eval_name)
                            found_estimator = True
                            found_datasets[n] = True
                        except FileNotFoundError:
                            if error_on_missing:
                                raise FileNotFoundError(
                                    f"Results for {regressor_eval_name} on "
                                    f"{dataset_name} {split} resample {resample} not "
                                    f"found."
                                )

            if not found_estimator:
                print(f"Regressor {regressor_eval_name} not found.")  # noqa: T201

        missing_datasets = [
            dataset
            for dataset, found in zip(dataset_names[i], found_datasets)
            if not found
        ]
        if missing_datasets:
            msg = f"Files for datasets {missing_datasets} not found."
            if error_on_missing:
                raise FileNotFoundError(msg)
            else:
                print("\n\n" + msg)  # noqa: T201

    evaluate_regressors(
        regressor_results,
        save_path,
        error_on_missing=error_on_missing,
        eval_name=eval_name,
        estimator_names=names,
    )


def evaluate_forecasters(
    forecaster_results,
    save_path,
    error_on_missing=True,
    eval_name=None,
    estimator_names=None,
):
    """
    Evaluate multiple forecasters on multiple datasets.

    Writes multiple csv files and figures to save_path, one for each statistic
    evaluated. Provides a summary csv file with the average statistic and
    average rank for each forecaster.

    Parameters
    ----------
    forecaster_results : list of ForecasterResults
        The results to evaluate.
    save_path : str
        The path to save the evaluation results to.
    error_on_missing : bool, default=True
        Whether to raise an error if results are missing.
    eval_name : str, default=None
        The name of the evaluation, used in save_path.
    estimator_names : list of str, default=None
        The names of the estimator for each forecaster result. If None, uses
        the estimator_name attribute of each forecaster result.
    """
    _evaluate_estimators(
        forecaster_results,
        ForecasterResults.statistics,
        save_path,
        error_on_missing,
        eval_name,
        estimator_names,
    )


def evaluate_forecasters_from_file(
    load_paths,
    save_path,
    error_on_missing=True,
    eval_name=None,
    verify_results=True,
    estimator_names=None,
):
    """
    Evaluate multiple forecasters on multiple datasets from file.

    Writes multiple csv files and figures to save_path, one for each statistic
    evaluated. Provides a summary csv file with the average statistic and
    average rank for each forecaster.

    Parameters
    ----------
    load_paths : list of str
        The paths to the forecaster result files to evaluate.
    save_path : str
        The path to save the evaluation results to.
    error_on_missing : bool, default=True
        Whether to raise an error if results are missing.
    eval_name : str, default=None
        The name of the evaluation, used in save_path.
    verify_results : bool, default=True
        If the verification should be performed on the loaded results values.
    estimator_names : list of str, default=None
        The names of the estimator for each forecaster result. If None, uses
        the estimator_name attribute of each forecaster result.
    """
    forecaster_results = []
    for load_path in load_paths:
        try:
            forecaster_results.append(
                ForecasterResults().load_from_file(
                    load_path, verify_values=verify_results
                )
            )
        except FileNotFoundError:
            if error_on_missing:
                raise FileNotFoundError(f"Results for {load_path} not found.")

    evaluate_forecasters(
        forecaster_results,
        save_path,
        error_on_missing=error_on_missing,
        eval_name=eval_name,
        estimator_names=estimator_names,
    )


def evaluate_forecasters_by_problem(
    load_path,
    forecaster_names,
    dataset_names,
    save_path,
    resamples=None,
    error_on_missing=True,
    eval_name=None,
    verify_results=True,
):
    """
    Evaluate multiple forecasters on multiple datasets from file using standard paths.

    Finds files using forecaster, dataset and resample names. It is expected the
    common tsml-eval file structure of
    {forecaster}/Predictions/{dataset}/{split}Resample{resample}.csv is followed.

    Writes multiple csv files and figures to save_path, one for each statistic
    evaluated. Provides a summary csv file with the average statistic and
    average rank for each forecaster.

    Parameters
    ----------
    load_path : str or list of str
        The path to the collection of forecaster result files to evaluate.
        If load_path is a list, it will load results from each path in the list. It
        is expected that forecaster_names and dataset_names are lists of lists with
        the same length as load_path.
    forecaster_names : list of str, list of tuple or list of list
        The names of the forecasters to evaluate.
        A length 2 tuple containing strings can be used to specify a forecaster name to
        load from in the first item and a forecaster name to use in the evaluation
        results in the second.
        If load_path is a list, regressor_names must be a list of lists with the same
        length as load_path.
    dataset_names : list of str or list of list
        The names of the datasets to evaluate.
        If load_path is a list, dataset_names must be a list of lists with the same
        length as load_path..
    save_path : str
        The path to save the evaluation results to.
    resamples : int or list of int, default=None
        The resamples to evaluate. If int, evaluates resamples 0 to resamples-1.
    error_on_missing : bool, default=True
        Whether to raise an error if results are missing.
    eval_name : str, default=None
        The name of the evaluation, used in save_path.
    verify_results : bool, default=True
        If the verification should be performed on the loaded results values.
    """
    load_path, forecaster_names, dataset_names, resamples = _evaluate_by_problem_init(
        "forecaster",
        load_path,
        forecaster_names,
        dataset_names,
        resamples,
    )

    forecaster_results = []
    estimator_eval_names = []
    names = []
    for i, path in enumerate(load_path):
        found_datasets = np.zeros(len(dataset_names[i]), dtype=bool)

        for forecaster_name in forecaster_names[i]:
            found_estimator = False

            if isinstance(forecaster_name, tuple):
                forecaster_eval_name = forecaster_name[1]
                forecaster_name = forecaster_name[0]
            else:
                forecaster_eval_name = forecaster_name

            if forecaster_eval_name not in estimator_eval_names:
                estimator_eval_names.append(forecaster_eval_name)
            else:
                raise ValueError(
                    f"Duplicate evaluation name {forecaster_eval_name} found."
                )

            for n, dataset_name in enumerate(dataset_names[i]):
                for resample in resamples:
                    try:
                        forecaster_results.append(
                            ForecasterResults().load_from_file(
                                f"{path}/{forecaster_name}/Predictions/"
                                f"{dataset_name}/testResample{resample}.csv",
                                verify_values=verify_results,
                            )
                        )
                        names.append(forecaster_eval_name)
                        found_estimator = True
                        found_datasets[n] = True
                    except FileNotFoundError:
                        if error_on_missing:
                            raise FileNotFoundError(
                                f"Results for {forecaster_eval_name} on {dataset_name} "
                                f"resample {resample} not found."
                            )

            if not found_estimator:
                print(f"Forecaster {forecaster_eval_name} not found.")  # noqa: T201

        missing_datasets = [
            dataset
            for dataset, found in zip(dataset_names[i], found_datasets)
            if not found
        ]
        if missing_datasets:
            msg = f"Files for datasets {missing_datasets} not found."
            if error_on_missing:
                raise FileNotFoundError(msg)
            else:
                print("\n\n" + msg)  # noqa: T201

    evaluate_forecasters(
        forecaster_results,
        save_path,
        error_on_missing=error_on_missing,
        eval_name=eval_name,
        estimator_names=names,
    )


def _evaluate_by_problem_init(
    type, load_path, estimator_names, dataset_names, resamples
):
    if isinstance(load_path, str):
        load_path = [load_path]
    elif not isinstance(load_path, list):
        raise TypeError("load_path must be a str or list of str.")

    if isinstance(estimator_names[0], (str, tuple)):
        estimator_names = [estimator_names]
    elif not isinstance(estimator_names[0], list):
        raise TypeError(f"{type}_names must be a str, tuple or list of str or tuple.")

    if isinstance(dataset_names[0], str):
        dataset_names = [dataset_names]
    elif not isinstance(dataset_names[0], list):
        raise TypeError("dataset_names must be a str or list of str.")

    if len(load_path) != len(estimator_names) or len(load_path) != len(dataset_names):
        raise ValueError(
            f"load_path, {type}_names and dataset_names must be the same length."
        )

    if resamples is None:
        resamples = [""]
    elif isinstance(resamples, int):
        resamples = [str(i) for i in range(resamples)]
    else:
        resamples = [str(resample) for resample in resamples]

    return load_path, estimator_names, dataset_names, resamples


def _evaluate_estimators(
    estimator_results,
    statistics,
    save_path,
    error_on_missing,
    eval_name,
    estimator_names,
):
    save_path = save_path + "/" + eval_name + "/"

    estimators = set()
    datasets = set()
    resamples = set()
    has_test = False
    has_train = False

    results_dict = _create_results_dictionary(estimator_results, estimator_names)

    if eval_name is None:
        dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        eval_name = f"{estimator_results[0].__class__.__name__}Evaluation {dt}"

    for estimator_name in results_dict:
        estimators.add(estimator_name)
        for dataset_name in results_dict[estimator_name]:
            datasets.add(dataset_name)
            for split in results_dict[estimator_name][dataset_name]:
                split_fail = False
                if split == "train":
                    has_train = True
                elif split == "test":
                    has_test = True
                else:
                    split_fail = True

                for resample in results_dict[estimator_name][dataset_name][split]:
                    if split_fail:
                        raise ValueError(
                            "Results must have a split of either 'train' or 'test' "
                            f"to be evaluated. Missing for {estimator_name} on "
                            f"{dataset_name} resample {resample}."
                        )

                    if resample is not None:
                        resamples.add(resample)
                    else:
                        raise ValueError(
                            "Results must have a resample_id to be evaluated. "
                            f"Missing for {estimator_name} on {dataset_name} "
                            f"{split} resample {resample}."
                        )

    estimators = sorted(list(estimators))
    datasets = sorted(list(datasets))
    resamples = sorted(list(resamples))
    has_dataset_train = np.zeros(
        (len(estimators), len(datasets), len(resamples)), dtype=bool
    )
    has_dataset_test = np.zeros(
        (len(estimators), len(datasets), len(resamples)), dtype=bool
    )

    for estimator_name in results_dict:
        for dataset_name in results_dict[estimator_name]:
            for split in results_dict[estimator_name][dataset_name]:
                for resample in results_dict[estimator_name][dataset_name][split]:
                    if split == "train":
                        has_dataset_train[estimators.index(estimator_name)][
                            datasets.index(dataset_name)
                        ][resamples.index(resample)] = True
                    elif split == "test":
                        has_dataset_test[estimators.index(estimator_name)][
                            datasets.index(dataset_name)
                        ][resamples.index(resample)] = True

    msg = "\n\n"
    missing = False
    splits = []

    if has_train:
        splits.append("train")
        for (i, n, j), present in np.ndenumerate(has_dataset_train):
            if not present:
                msg += (
                    f"Estimator {estimators[i]} is missing train results for "
                    f"{datasets[n]} resample {resamples[j]}.\n"
                )
                missing = True

    if has_test:
        splits.append("test")
        for (i, n, j), present in np.ndenumerate(has_dataset_test):
            if not present:
                msg += (
                    f"Estimator {estimators[i]} is missing test results for "
                    f"{datasets[n]} resample {resamples[j]}.\n"
                )
                missing = True

    if missing:
        if error_on_missing:
            print(msg + "\n")  # noqa: T201
            raise ValueError("Missing results, exiting evaluation.")
        else:
            if has_test and has_train:
                has_both = has_dataset_train.all(axis=(0, 2)) & has_dataset_test.all(
                    axis=(0, 2)
                )
                datasets = [dataset for dataset, has in zip(datasets, has_both) if has]
            elif has_test:
                datasets = [
                    dataset
                    for dataset, has in zip(datasets, has_dataset_test.all(axis=(0, 2)))
                    if has
                ]
            else:
                datasets = [
                    dataset
                    for dataset, has in zip(
                        datasets, has_dataset_train.all(axis=(0, 2))
                    )
                    if has
                ]

            msg += "\nMissing results, continuing evaluation with available datasets.\n"
            print(msg)  # noqa: T201
    else:
        msg += "All results present, continuing evaluation.\n"
        print(msg)  # noqa: T201

    print(f"Estimators ({len(estimators)}): {estimators}\n")  # noqa: T201
    print(f"Datasets ({len(datasets)}): {datasets}\n")  # noqa: T201
    print(f"Resamples ({len(resamples)}): {resamples}\n")  # noqa: T201

    stats = []
    for var, (stat, ascending, time) in statistics.items():
        for split in splits:
            average, rank = _create_directory_for_statistic(
                estimators,
                datasets,
                resamples,
                split,
                results_dict,
                stat,
                ascending,
                time,
                var,
                save_path,
            )
            stats.append((average, rank, stat, ascending, split))

    _summary_evaluation(stats, estimators, save_path, eval_name)


def _create_results_dictionary(estimator_results, estimator_names):
    results_dict = {}

    for i, estimator_result in enumerate(estimator_results):
        name = (
            estimator_result.estimator_name
            if estimator_names is None
            else estimator_names[i]
        )

        if results_dict.get(name) is None:
            results_dict[name] = {}

        if results_dict[name].get(estimator_result.dataset_name) is None:
            results_dict[name][estimator_result.dataset_name] = {}

        if (
            results_dict[name][estimator_result.dataset_name].get(
                estimator_result.split.lower()
            )
            is None
        ):
            results_dict[name][estimator_result.dataset_name][
                estimator_result.split.lower()
            ] = {}

        results_dict[name][estimator_result.dataset_name][
            estimator_result.split.lower()
        ][estimator_result.resample_id] = estimator_result

    return results_dict


def _create_directory_for_statistic(
    estimators,
    datasets,
    resamples,
    split,
    results_dict,
    statistic_name,
    higher_better,
    is_timing,
    variable_name,
    save_path,
):
    os.makedirs(f"{save_path}/{statistic_name}/all_resamples/", exist_ok=True)

    average_stats = np.zeros((len(datasets), len(estimators)))

    for i, estimator_name in enumerate(estimators):
        est_stats = np.zeros((len(datasets), len(resamples)))

        for n, dataset_name in enumerate(datasets):
            for j, resample in enumerate(resamples):
                er = results_dict[estimator_name][dataset_name][split][resample]
                er.calculate_statistics()
                est_stats[n, j] = (
                    er.__dict__[variable_name]
                    if not is_timing
                    else (
                        time_to_milliseconds(er.__dict__[variable_name], er.time_unit)
                    )
                )

            average_stats[n, i] = np.mean(est_stats[n, :])

        with open(
            f"{save_path}/{statistic_name}/all_resamples/{estimator_name}_"
            f"{statistic_name}.csv",
            "w",
        ) as file:
            file.write(f",{','.join([str(j) for j in resamples])}\n")
            for n, dataset_name in enumerate(datasets):
                file.write(
                    f"{dataset_name},{','.join([str(j) for j in est_stats[n]])}\n"
                )

    with open(f"{save_path}/{statistic_name}/{statistic_name}_mean.csv", "w") as file:
        file.write(f",{','.join(estimators)}\n")
        for i, dataset_name in enumerate(datasets):
            file.write(
                f"{dataset_name},{','.join([str(n) for n in average_stats[i]])}\n"
            )

    ranks = np.apply_along_axis(
        lambda x: rank_array(x, higher_better=higher_better), 1, average_stats
    )

    with open(f"{save_path}/{statistic_name}/{statistic_name}_ranks.csv", "w") as file:
        file.write(f",{','.join(estimators)}\n")
        for i, dataset_name in enumerate(datasets):
            file.write(f"{dataset_name},{','.join([str(n) for n in ranks[i]])}\n")

    _figures_for_statistic(
        average_stats, estimators, statistic_name, higher_better, save_path
    )

    return average_stats, ranks


def _figures_for_statistic(
    scores, estimators, statistic_name, higher_better, save_path
):
    os.makedirs(f"{save_path}/{statistic_name}/figures/", exist_ok=True)

    cd = plot_critical_difference(scores, estimators, errors=not higher_better)
    cd.savefig(
        f"{save_path}/{statistic_name}/figures/"
        f"{statistic_name}_critical_difference.pdf",
        bbox_inches="tight",
    )
    pickle.dump(
        cd,
        open(
            f"{save_path}/{statistic_name}/figures/"
            f"{statistic_name}_critical_difference.pickle",
            "wb",
        ),
    )
    plt.close()

    # crashes when scores are the same?

    # box = plot_boxplot_median(scores.transpose(), estimators)
    # box.savefig(
    #     f"{save_path}/{statistic_name}/figures/{statistic_name}_boxplot.pdf",
    #     bbox_inches="tight",
    # )
    # pickle.dump(
    #     box,
    #     open(
    #         f"{save_path}/{statistic_name}/figures/{statistic_name}_boxplot.pickle",
    #         "wb",
    #     ),
    # )
    # plt.close()

    for i, est1 in enumerate(estimators):
        for n, est2 in enumerate(estimators):
            os.makedirs(
                f"{save_path}/{statistic_name}/figures/scatters/{est1}/", exist_ok=True
            )

            scatter = plot_scatter(scores[:, (i, n)], est1, est2)
            scatter.savefig(
                f"{save_path}/{statistic_name}/figures/scatters/{est1}/"
                f"{statistic_name}_scatter_{est1}_{est2}.pdf",
                bbox_inches="tight",
            )
            pickle.dump(
                scatter,
                open(
                    f"{save_path}/{statistic_name}/figures/scatters/{est1}/"
                    f"{statistic_name}_scatter_{est1}_{est2}.pickle",
                    "wb",
                ),
            )
            plt.close()


def _summary_evaluation(stats, estimators, save_path, eval_name):
    with open(f"{save_path}/{eval_name}_summary.csv", "w") as file:
        for stat in stats:
            avg_stat = np.mean(stat[0], axis=0)
            avg_rank = np.mean(stat[1], axis=0)
            sorted_indices = [
                i
                for i in sorted(
                    range(len(avg_rank)),
                    key=lambda x: (
                        avg_rank[x],
                        -avg_stat[x] if stat[3] else avg_stat[x],
                    ),
                )
            ]

            file.write(
                f"{stat[4]}{stat[2]},"
                f"{','.join([estimators[i] for i in sorted_indices])}\n"
            )
            file.write(
                f"{stat[4]}{stat[2]}Mean,"
                f"{','.join([str(n) for n in avg_stat[sorted_indices]])}\n"
            )
            file.write(
                f"{stat[4]}{stat[2]}AvgRank,"
                f"{','.join([str(n) for n in avg_rank[sorted_indices]])}\n\n"
            )
