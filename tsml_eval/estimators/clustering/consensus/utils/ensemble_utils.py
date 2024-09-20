from typing import List


def extract_distances_from_model_names(model_paths: List[str]):
    distances = []
    for path in model_paths:
        curr = path.split("/Predictions")[0].split("/")[-1].split("-")[0:]
        final_distances = ""
        for i in range(len(curr)):
            final_distances += curr[i]
            if i + 1 != len(curr):
                final_distances += "-"
        distances.append(final_distances)
    return distances


if __name__ == "__main__":
    paths = [
        "/home/chris/Documents/phd-results/31-aug-results/normalised/test-train-split/pam/pam-euclidean/Predictions/NonInvasiveFetalECGThorax2/",
        "/home/chris/Documents/phd-results/31-aug-results/normalised/test-train-split/pam/pam-ddtw/Predictions/NonInvasiveFetalECGThorax2/",
        "/home/chris/Documents/phd-results/31-aug-results/normalised/test-train-split/pam/pam-shape-dtw/Predictions/NonInvasiveFetalECGThorax2/",
        "/home/chris/Documents/phd-results/31-aug-results/normalised/test-train-split/pam/pam-soft-dtw/Predictions/NonInvasiveFetalECGThorax2/",
        "/home/chris/Documents/phd-results/31-aug-results/normalised/test-train-split/pam/pam-twe/Predictions/NonInvasiveFetalECGThorax2/",
        "/home/chris/Documents/phd-results/31-aug-results/normalised/test-train-split/pam/pam-msm/Predictions/NonInvasiveFetalECGThorax2/",
        "/home/chris/Documents/phd-results/31-aug-results/normalised/test-train-split/pam/pam-erp/Predictions/NonInvasiveFetalECGThorax2/",
        "/home/chris/Documents/phd-results/31-aug-results/normalised/test-train-split/pam/pam-edr/Predictions/NonInvasiveFetalECGThorax2/",
    ]
    print(extract_distances_from_model_names(paths))
