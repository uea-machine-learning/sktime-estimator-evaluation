from tsml_eval.evaluation import evaluate_clusterers_by_problem

if __name__ == '__main__':

    classifiers = ["ROCKET", "TSF", "1NN-DTW"]
    datasets = ["Chinatown", "ItalyPowerDemand", "Trace"]

    evaluate_clusterers_by_problem(
        "../tsml_eval/testing/_test_result_files/classification/",
        classifiers,
        datasets,
        "./generated_results/",
        resamples=1,
        eval_name="ExampleEval",
    )