"""Shift-invariant average."""

import numpy as np

from tsml_eval.estimators.clustering.ksc._shift_invariant_distance import (
    shift_invariant_best_shift,
)


def shift_invariant_average(
    X: np.ndarray, initial_center: np.ndarray, max_shift: int = 2
):
    n_instances, n_dims, n_timepoints = X.shape
    optimal_shifts = np.zeros_like(X)

    for i in range(X.shape[0]):
        if not initial_center.any():
            optimal_shifts[i] = X[i]
        else:
            _, curr_shift = shift_invariant_best_shift(initial_center, X[i], max_shift)
            optimal_shifts[i] = curr_shift

    if optimal_shifts.shape[0] == 0:
        return np.zeros((n_dims, n_timepoints))

    normalized_shifts = optimal_shifts / np.sqrt(
        np.sum(optimal_shifts**2, axis=2)
    ).reshape(n_instances, n_dims, 1)

    M = np.zeros((n_dims, n_timepoints, n_timepoints))
    for d in range(n_dims):
        M[d] = np.dot(
            normalized_shifts[:, d, :].T, normalized_shifts[:, d, :]
        ) - n_instances * np.eye(n_timepoints)

    new_center = np.zeros((n_dims, n_timepoints))
    for d in range(n_dims):
        eigvals, eigvecs = np.linalg.eig(M[d])
        new_center[d] = np.real(eigvecs[:, np.argmax(eigvals)])

        if np.sum(new_center[d]) < 0:
            new_center[d] = -new_center[d]

    return new_center
