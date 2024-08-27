"""Precompute pairwise distances for a given model."""

import os

import numpy as np
from aeon.distances import pairwise_distance
from aeon.transformations.collection import TimeSeriesScaler

from tsml_eval.experiments.set_clusterer import _get_distance_default_params
from tsml_eval.utils.datasets import load_experiment_data

NUM_CHUNKS = 5

PRECOMPUTE_DISTANCE_MODEL_NAME = [
    "euclidean",
    "squared",
    "dtw",
    "ddtw",
    "wdtw",
    "wddtw",
    "lcss",
    "erp",
    "edr",
    "twe",
    "msm",
    "adtw",
    "shape_dtw",
    "soft_dtw",
    "dtw-w:0.2",
    "ddtw-w:0.2",
]


def _extract_parameters(model_name, data_path, dataset_name, row_normalise):
    split_name = model_name.split("-")
    distance = split_name[0]
    parameter_split = split_name[1:]
    data_vars = [
        data_path,
        dataset_name,
        0,
        False,
    ]
    parameters = _get_distance_default_params(distance, data_vars, row_normalise)
    if len(parameter_split) > 0:
        for parameter in parameter_split:
            split_parameter = parameter.split(":")
            if len(split_parameter) == 2:
                parameters[split_parameter[0]] = float(split_parameter[1])
    return distance, parameters


def _precompute_distances(
    model_name, *, data_path, results_path, dataset_name, row_normalise, chunk_idx=None
):
    print(
        f"Computing distances for {model_name} on {dataset_name} of chunk {chunk_idx}"
    )
    distance, distance_parameters = _extract_parameters(
        model_name, data_path, dataset_name, row_normalise
    )

    X_train, y_train, X_test, y_test, resample = load_experiment_data(
        problem_path=data_path,
        dataset=dataset_name,
        resample_id=0,
        predefined_resample=False,
    )

    X_train = (
        np.concatenate([X_train, X_test], axis=0)
        if isinstance(X_train, np.ndarray)
        else X_train + X_test
    )

    if row_normalise:
        scaler = TimeSeriesScaler()
        X_train = scaler.fit_transform(X_train)

    n_samples = X_train.shape[0]

    output_file = f"{results_path}"
    if row_normalise:
        output_file += "/normalised"
    else:
        output_file += "/unnormalised"
    output_file += f"/{distance}"

    # Get distance parameters in alphabetical order by key
    temp_dist_params = sorted(distance_parameters.items(), key=lambda x: x[0])
    for key, value in temp_dist_params:
        output_file += f"/{key}-{value}"
    os.makedirs(output_file, exist_ok=True)

    if chunk_idx is None:
        # Compute the pairwise distances for all data in one go
        if os.path.exists(f"{output_file}/{dataset_name}.npy"):
            raise FileExistsError(
                f"File already exists: {output_file}/{dataset_name}.npy"
            )

        full_distances = pairwise_distance(
            X_train, metric=distance, **distance_parameters
        )
        # Check if the file already exists
        np.save(f"{output_file}/{dataset_name}.npy", full_distances)
        return full_distances

    output_file = f"{output_file}/chunks"

    os.makedirs(output_file, exist_ok=True)

    # Determine the size of each chunk
    chunk_size = n_samples // NUM_CHUNKS

    # Compute the start and end indices for the current chunk
    start_idx = chunk_idx * chunk_size
    end_idx = (chunk_idx + 1) * chunk_size if chunk_idx < NUM_CHUNKS - 1 else n_samples

    if os.path.exists(f"{output_file}/{dataset_name}_{chunk_idx}.npy"):
        raise FileExistsError(f"File already exists: {output_file}/{dataset_name}.npy")
    # Compute pairwise distances for this chunk
    distances_chunk = pairwise_distance(
        X_train[start_idx:end_idx], X_train, metric=distance, **distance_parameters
    )

    # Save the result to a file
    np.save(f"{output_file}/{dataset_name}_{chunk_idx}.npy", distances_chunk)
    return distances_chunk


def _merge_chunks(model_name, *, data_path, results_path, dataset_name, row_normalise):
    # Extract distance and parameters
    distance, distance_parameters = _extract_parameters(
        model_name, data_path, dataset_name, row_normalise
    )

    # Determine the output file path based on the parameters
    output_file = f"{results_path}"
    if row_normalise:
        output_file += "/normalised"
    else:
        output_file += "/unnormalised"
    output_file += f"/{distance}"

    # Get distance parameters in alphabetical order by key
    temp_dist_params = sorted(distance_parameters.items(), key=lambda x: x[0])
    for key, value in temp_dist_params:
        output_file += f"/{key}-{value}"

    # Directory containing chunks
    chunks_dir = f"{output_file}/chunks"

    if not os.path.exists(chunks_dir):
        raise FileNotFoundError(f"Chunks directory does not exist: {chunks_dir}")

    # List all chunk files
    chunk_files = sorted([f for f in os.listdir(chunks_dir) if f.endswith(".npy")])

    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunks_dir}")

    # Load the first chunk to determine the size of the full distance matrix
    first_chunk = np.load(os.path.join(chunks_dir, chunk_files[0]))
    n_samples = first_chunk.shape[1]

    # Create an empty array to store the full distance matrix
    full_distances = np.zeros((n_samples, n_samples))

    # Fill in the full distance matrix with the computed chunks
    for chunk_file in chunk_files:
        chunk_idx = int(
            chunk_file.split("_")[-1].split(".")[0]
        )  # Extract chunk index from filename
        start_idx = chunk_idx * first_chunk.shape[0]
        distances_chunk = np.load(os.path.join(chunks_dir, chunk_file))
        end_idx = start_idx + distances_chunk.shape[0]
        full_distances[start_idx:end_idx, :] = distances_chunk

    # Save the full distance matrix to the parent directory of the chunks folder
    final_output_file = f"{output_file}/{dataset_name}.npy"
    if os.path.exists(final_output_file):
        raise FileExistsError(f"File already exists: {final_output_file}")

    np.save(final_output_file, full_distances)
    print(f"Full distance matrix saved to {final_output_file}")
    return full_distances


import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    print(f"Running precompute_distances.py with args {args}")
    # precompute_distances(sys.argv[1:])
#     model_name = "msm"
# for chunk_idx in range(5):
#     precompute_distances(
#         model_name,
#         data_path="/Users/chris/Documents/Phd-data/Datasets/Univariate_ts",
#         results_path="/Users/chris/Documents/Phd-data/precomputed-distances",
#         dataset_name="GunPoint",
#         row_normalise=True,
#         chunk_idx=chunk_idx,
#     )
#
# merge_chunks(
#     model_name,
#     data_path="/Users/chris/Documents/Phd-data/Datasets/Univariate_ts",
#     results_path="/Users/chris/Documents/Phd-data/precomputed-distances",
#     dataset_name="GunPoint",
#     row_normalise=True,
# )

# full = precompute_distances(
#     model_name,
#     data_path="/Users/chris/Documents/Phd-data/Datasets/Univariate_ts",
#     results_path="/Users/chris/Documents/Phd-data/precomputed-distances/temp",
#     dataset_name="GunPoint",
#     row_normalise=True,
# )
#
# loaded = np.load("/Users/chris/Documents/Phd-data/precomputed-distances/normalised/msm/c-1.0/independent-True/GunPoint.npy")
# shape = loaded.shape
#
# equal = np.array_equal(full, loaded)
# full_shape = full.shape
