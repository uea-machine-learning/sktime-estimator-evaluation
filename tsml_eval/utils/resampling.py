"""Utility functions fordata resampling."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = [
    "resample_data",
    "resample_data_indices",
    "stratified_resample_data",
    "stratified_resample_data_indices",
    "make_imbalance",
]


import numpy as np
from sklearn.utils import check_random_state


def resample_data(X_train, y_train, X_test, y_test, random_state=None):
    """Resample data without replacement using a random state.

    Reproducible resampling. Combines train and test, randomly resamples, then returns
    new train and test.

    Parameters
    ----------
    X_train : np.ndarray or list of np.ndarray
        Train data in a 2d or 3d ndarray or list of arrays.
    y_train : np.ndarray
        Train data labels.
    X_test : np.ndarray or list of np.ndarray
        Test data in a 2d or 3d ndarray or list of arrays.
    y_test : np.ndarray
        Test data labels.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Returns
    -------
    train_X : np.ndarray or list of np.ndarray
        New train data.
    train_y : np.ndarray
        New train labels.
    test_X : np.ndarray or list of np.ndarray
        New test data.
    test_y : np.ndarray
        New test labels.
    """
    if isinstance(X_train, np.ndarray):
        is_array = True
    elif isinstance(X_train, list):
        is_array = False
    else:
        raise ValueError(
            "X_train must be a np.ndarray array or list of np.ndarray arrays"
        )

    # add both train and test to a single dataset
    all_labels = np.concatenate((y_train, y_test), axis=None)
    all_data = (
        np.concatenate([X_train, X_test], axis=0) if is_array else X_train + X_test
    )

    # shuffle data indices
    rng = check_random_state(random_state)
    indices = np.arange(len(all_data), dtype=int)
    rng.shuffle(indices)

    train_indices = indices[: len(X_train)]
    test_indices = indices[len(X_train) :]

    # split the shuffled data into train and test
    X_train = (
        all_data[train_indices] if is_array else [all_data[i] for i in train_indices]
    )
    y_train = all_labels[train_indices]
    X_test = all_data[test_indices] if is_array else [all_data[i] for i in test_indices]
    y_test = all_labels[test_indices]

    return X_train, y_train, X_test, y_test


def resample_data_indices(y_train, y_test, random_state=None):
    """Return data resample indices without replacement using a random state.

    Reproducible resampling. Combines train and test, randomly resamples, then returns
    the new position for both the train and test set. Uses indices for a combined train
    and test set, with test indices appearing after train indices.

    Parameters
    ----------
    y_train : np.ndarray
        Train data labels.
    y_test : np.ndarray
        Test data labels.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Returns
    -------
    train_indices : np.ndarray
        The index of cases to use in the train set from the combined train and test
        data.
    test_indices : np.ndarray
        The index of cases to use in the test set from the combined train and test data.
    """
    # shuffle data indices
    rng = check_random_state(random_state)
    indices = np.arange(len(y_train) + len(y_test), dtype=int)
    rng.shuffle(indices)

    train_indices = indices[: len(y_train)]
    test_indices = indices[len(y_train) :]

    return train_indices, test_indices


def stratified_resample_data(X_train, y_train, X_test, y_test, random_state=None):
    """Stratified resample data without replacement using a random state.

    Reproducible resampling. Combines train and test, resamples to get the same class
    distribution, then returns new train and test.

    Parameters
    ----------
    X_train : np.ndarray or list of np.ndarray
        Train data in a 2d or 3d ndarray or list of arrays.
    y_train : np.ndarray
        Train data labels.
    X_test : np.ndarray or list of np.ndarray
        Test data in a 2d or 3d ndarray or list of arrays.
    y_test : np.ndarray
        Test data labels.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Returns
    -------
    train_X : np.ndarray or list of np.ndarray
        New train data.
    train_y : np.ndarray
        New train labels.
    test_X : np.ndarray or list of np.ndarray
        New test data.
    test_y : np.ndarray
        New test labels.
    """
    if isinstance(X_train, np.ndarray):
        is_array = True
    elif isinstance(X_train, list):
        is_array = False
    else:
        raise ValueError(
            "X_train must be a np.ndarray array or list of np.ndarray arrays"
        )

    # add both train and test to a single dataset
    all_labels = np.concatenate((y_train, y_test), axis=None)
    all_data = (
        np.concatenate([X_train, X_test], axis=0) if is_array else X_train + X_test
    )

    # shuffle data indices
    rng = check_random_state(random_state)

    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    # ensure same classes exist in both train and test
    assert list(unique_train) == list(unique_test)

    if is_array:
        shape = list(X_train.shape)
        shape[0] = 0

    X_train = np.zeros(shape) if is_array else []
    y_train = np.zeros(0)
    X_test = np.zeros(shape) if is_array else []
    y_test = np.zeros(0)

    # for each class
    for label_index in range(len(unique_train)):
        # get the indices of all instances with this class label and shuffle them
        label = unique_train[label_index]
        indices = np.where(all_labels == label)[0]
        rng.shuffle(indices)

        train_indices = indices[: counts_train[label_index]]
        test_indices = indices[counts_train[label_index] :]

        # extract data from corresponding indices
        train_cases = (
            all_data[train_indices]
            if is_array
            else [all_data[i] for i in train_indices]
        )
        train_labels = all_labels[train_indices]
        test_cases = (
            all_data[test_indices] if is_array else [all_data[i] for i in test_indices]
        )
        test_labels = all_labels[test_indices]

        # concat onto current data from previous loop iterations
        X_train = (
            np.concatenate([X_train, train_cases], axis=0)
            if is_array
            else X_train + train_cases
        )
        y_train = np.concatenate([y_train, train_labels], axis=None)
        X_test = (
            np.concatenate([X_test, test_cases], axis=0)
            if is_array
            else X_test + test_cases
        )
        y_test = np.concatenate([y_test, test_labels], axis=None)

    return X_train, y_train, X_test, y_test


def stratified_resample_data_indices(y_train, y_test, random_state=None):
    """Return stratified data resample indices without replacement using a random state.

    Reproducible resampling. Combines train and test, resamples to get the same class
    distribution, then returns the new position for both the train and test set.
    Uses indices for a combined train and test set, with test indices appearing after
    train indices.

    Parameters
    ----------
    y_train : np.ndarray
        Train data labels.
    y_test : np.ndarray
        Test data labels.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Returns
    -------
    train_indices : np.ndarray
        The index of cases to use in the train set from the combined train and test
        data.
    test_indices : np.ndarray
        The index of cases to use in the test set from the combined train and test data.
    """
    # add both train and test to a single dataset
    all_labels = np.concatenate((y_train, y_test), axis=None)

    # shuffle data indices
    rng = check_random_state(random_state)

    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    # ensure same classes exist in both train and test
    assert list(unique_train) == list(unique_test)

    train_indices = np.zeros(0, dtype=int)
    test_indices = np.zeros(0, dtype=int)

    # for each class
    for label_index in range(len(unique_train)):
        # get the indices of all instances with this class label and shuffle them
        label = unique_train[label_index]
        indices = np.where(all_labels == label)[0]
        rng.shuffle(indices)

        train_indices = np.concatenate(
            [train_indices, indices[: counts_train[label_index]]], axis=None
        )
        test_indices = np.concatenate(
            [test_indices, indices[counts_train[label_index] :]], axis=None
        )

    return train_indices, test_indices


def make_imbalance(X, y, sampling_ratio=None, random_state=0):
    """Make the data imbalanced."""
    """
    Make the data imbalanced
    :param X: the data
    :param y: the target
    :param sampling_ratio: the sampling ratio
    """

    x_minority = X[y == "1"]  # Minority class
    x_majority = X[y == "0"]  # Majority class
    rng = check_random_state(random_state)
    labels, counts = np.unique(y, return_counts=True)
    imbalance_ratio = counts[0] / counts[1]
    if imbalance_ratio > sampling_ratio:
        indices = np.arange(len(x_majority))  # 创建索引数组
        rng.shuffle(indices)
        x_majority = x_majority[indices][: int(len(x_minority) * sampling_ratio)]

    else:
        indices = np.arange(len(x_minority))
        rng.shuffle(indices)
        minority_num = int(len(x_majority) // sampling_ratio)
        if minority_num <= 1:
            minority_num = 1
        x_minority = x_minority[indices][:minority_num]

    y_majority = np.zeros(len(x_majority))
    y_minority = np.ones(len(x_minority))
    x_imbalanced = np.vstack((x_majority, x_minority))
    y_imbalanced = np.hstack((y_majority, y_minority))

    index_shuffle = np.arange(len(x_imbalanced))
    rng.shuffle(indices)
    x_imbalanced = x_imbalanced[index_shuffle]
    y_imbalanced = y_imbalanced[index_shuffle]

    return x_imbalanced, y_imbalanced
