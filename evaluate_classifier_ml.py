"""
Classifier evaluation and retraining.

Usage:
  script.py [--whole] [--male] [--threshold] [--leave-site-out] [--use-phenotypic] [<derivative> ...]

Options:
  --whole                Evaluate models on the whole dataset
  --male                 Evaluate models on male subjects
  --threshold            Evaluate models with thresholded connectivity
  --leave-site-out       Evaluate using leave-site-out cross-validation
  --use-phenotypic       Include phenotypic data during evaluation
  derivative             Derivatives to process (e.g., cc200, aal, ho)
"""

import os
import joblib
import numpy as np
import tabulate
from docopt import docopt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from utils import format_config, hdf5_handler, load_fold, load_phenotypes, reset

# Experiment configuration
experiment_config = {
    "cc200_whole": {"clf_type": "logistic_regression", "best_params": {"C": 1.0}},
    "aal_whole": {"clf_type": "logistic_regression", "best_params": {"C": 1.0}},
    "ez_whole": {"clf_type": "logistic_regression", "best_params": {"C": 1.0}},
    "ho_whole": {"clf_type": "logistic_regression", "best_params": {"C": 1.0}},
    "tt_whole": {"clf_type": "logistic_regression", "best_params": {"C": 1.0}},
    "dosenbach160_whole": {"clf_type": "logistic_regression", "best_params": {"C": 1.0}},
    "cc200_leavesiteout-NYU": {"clf_type": "logistic_regression", "best_params": {"C": 1.0}},
    "aal_leavesiteout-NYU": {"clf_type": "logistic_regression", "best_params": {"C": 1.0}},
    "ez_leavesiteout-NYU": {"clf_type": "logistic_regression", "best_params": {"C": 1.0}},
    "ho_leavesiteout-NYU": {"clf_type": "logistic_regression", "best_params": {"C": 1.0}},
    "tt_leavesiteout-NYU": {"clf_type": "logistic_regression", "best_params": {"C": 1.0}},
    "dosenbach160_leavesiteout-NYU": {"clf_type": "logistic_regression", "best_params": {"C": 1.0}}
}

def get_classifier(clf_type, params):
    classifiers = {
        "random_forest": RandomForestClassifier,
        "logistic_regression": LogisticRegression,
        "svm": SVC
    }
    if clf_type not in classifiers:
        raise ValueError(f"Unknown classifier type: {clf_type}")
    return classifiers[clf_type](**params)

def run(X_train, y_train, X_test, y_test, clf_type="svm", params=None, use_phenotypic=False):
    scaler = None
    if use_phenotypic:
        scaler = StandardScaler()
        X_train[:, -4:] = scaler.fit_transform(X_train[:, -4:])
        X_test[:, -4:] = scaler.transform(X_test[:, -4:])
    else:
        X_train = X_train[:, :-5]
        X_test = X_test[:, :-5]

    clf = get_classifier(clf_type, params)
    clf.fit(X_train, y_train)
    pred_y = clf.predict(X_test)

    # Compute metrics
    cm = confusion_matrix(y_test, pred_y)
    accuracy = accuracy_score(y_test, pred_y)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

    return [accuracy, precision, recall, f1_score, recall, specificity], clf

def retrain_and_evaluate(hdf5_path, pheno_path, experiment, use_phenotypic=False):
    if experiment not in experiment_config:
        print(f"Configuration not found for experiment: {experiment}")
        return None

    config = experiment_config[experiment]
    clf_type = config["clf_type"]
    best_params = config["best_params"]

    hdf5 = hdf5_handler(bytes(hdf5_path, encoding="utf8"), "a")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_fold(
        hdf5["patients"], hdf5["experiments"][experiment], "0"
    )

    X_combined = np.concatenate((X_train, X_valid), axis=0)
    y_combined = np.concatenate((y_train, y_valid), axis=0)

    print(f"Training model for {experiment}...")
    metrics, clf = run(X_combined, y_combined, X_test, y_test, clf_type, best_params, use_phenotypic)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/best_model_{experiment}.joblib"
    joblib.dump(clf, model_path)
    print(f"Model saved at {model_path}")

    return metrics

if __name__ == "__main__":
    reset()
    arguments = docopt(__doc__)

    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)
    hdf5_path = "./data/abide.hdf5"

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [d for d in arguments["<derivative>"] if d in valid_derivatives]
    
    experiments = []

    for derivative in derivatives:
        config = {"derivative": derivative}
        if arguments["--whole"]:
            experiments += [format_config("{derivative}_whole", config)]
        if arguments["--male"]:
            experiments += [format_config("{derivative}_male", config)]
        if arguments["--threshold"]:
            experiments += [format_config("{derivative}_threshold", config)]

        if arguments["--leave-site-out"]:
            for site in pheno["SITE_ID"].unique():
                if site=='NYU':
                    site_config = {"site": site}
                    experiments += [
                        format_config(
                            "{derivative}_leavesiteout-{site}", config, site_config
                        )
                    ]

    experiments = sorted(experiments)
    use_phenotypic = arguments["--use-phenotypic"]

    experiment_results = []
    for experiment in experiments:
        metrics = retrain_and_evaluate(hdf5_path, pheno_path, experiment, use_phenotypic)
        if metrics:
            experiment_results.append([experiment] + metrics)

    print(tabulate.tabulate(
        experiment_results,
        headers=["Experiment", "Accuracy", "Precision", "Recall", "F1-Score", "Sensitivity", "Specificity"],
        tablefmt="grid"
    ))
