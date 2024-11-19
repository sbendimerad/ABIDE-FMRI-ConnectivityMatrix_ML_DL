#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classifier evaluation.

Usage:
  classifier.py [--whole] [--male] [--threshold] [--leave-site-out] [--classifier=<clf>] [--use-phenotypic] [<derivative> ...]
  classifier.py (-h | --help)

Options:
  -h --help              Show this screen
  --whole                Run model for the whole dataset
  --male                 Run model for male subjects
  --threshold            Run model for thresholded subjects
  --leave-site-out       Prepare data using leave-site-out method
  --classifier=<clf>     Classifier to use: svm, logistic, random_forest, knn, xgboost [default: svm]
  --use-phenotypic       Include phenotypic data in the model
  derivative             Derivatives to process
"""

import random
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import tabulate
from docopt import docopt
from sklearn.preprocessing import StandardScaler
import argparse
from utils import (load_phenotypes, format_config, hdf5_handler, load_fold, reset)


def get_classifier(clf_type, params=None):
    """Initialize the classifier based on type and given parameters."""
    if params is None:
        params = {}

    classifiers = {
        "svm": SVC,
        "logistic": lambda **kwargs: LogisticRegression(max_iter=1000, **kwargs),
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier
    }
    
    return classifiers[clf_type](**params)


def get_param_distributions(clf_type):
    """Define parameter distributions for randomized search for each classifier."""
    param_distributions = {
        "svm": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        "logistic": {"C": [0.1, 1, 10]},
        "random_forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
        "knn": {"n_neighbors": [3, 5, 7]},
        "xgboost": {"n_estimators": [50, 100, 200], "max_depth": [3, 6, 10], "learning_rate": [0.01, 0.1, 0.2]}
    }
    
    return param_distributions[clf_type]


def run(X_train, y_train, X_valid, y_valid, clf_type="svm", params=None, use_phenotypic=False):
    """Run a model with given parameters and return evaluation metrics."""
    
    scaler = None

    if use_phenotypic:
        scaler = StandardScaler()
        # Assuming the last four columns are the ones to scale
        X_train[:, -4:] = scaler.fit_transform(X_train[:, -4:])
        X_valid[:, -4:] = scaler.transform(X_valid[:, -4:])
    else:
        # Exclude the last 5 columns
        X_train = X_train[:, :-5]
        X_valid = X_valid[:, :-5]

    clf = get_classifier(clf_type, params)
    clf.fit(X_train, y_train)
    pred_y = clf.predict(X_valid)
    [[TN, FP], [FN, TP]] = confusion_matrix(y_valid, pred_y).astype(float)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    sensitivity = recall = TP / (TP + FN)
    fscore = 2 * TP / (2 * TP + FP + FN)

    return [accuracy, precision, recall, fscore, sensitivity, specificity], scaler


def run_classifier(hdf5, experiment, clf_type="svm", n_iter=5, use_phenotypic=False):
    """Run classifier with random search on hyperparameters for each fold, only using train and validation sets."""
    param_distributions = get_param_distributions(clf_type)
    best_metrics = None
    best_params = None
    final_scaler = None

    # Random search over parameter samples
    for params in ParameterSampler(param_distributions, n_iter=n_iter, random_state=0):
        fold_metrics = []

        for fold in hdf5["experiments"][experiment]:
            X_train, y_train, X_valid, y_valid, X_test, y_test = load_fold(hdf5["patients"], hdf5["experiments"][experiment], fold)

            metrics, scaler = run(X_train, y_train, X_valid, y_valid, clf_type, params, use_phenotypic)
            fold_metrics.append(metrics)

        avg_metrics = np.mean(fold_metrics, axis=0).tolist()

        if best_metrics is None or avg_metrics[0] > best_metrics[0]:  # Assuming accuracy at index 0
            best_metrics = avg_metrics
            best_params = params
            final_scaler = scaler  # Save the scaler used for the best model

    return best_metrics, best_params, final_scaler


# Main function
if __name__ == "__main__":
    reset()
    arguments = docopt(__doc__)

    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)
    hdf5 = hdf5_handler(bytes("./data/abide.hdf5", encoding="utf8"), 'a')

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
                site_config = {"site": site}
                experiments += [format_config("{derivative}_leavesiteout-{site}", config, site_config)]

    experiments = sorted(experiments)
    experiment_results = []
    best_params_results = {}

    use_phenotypic = arguments["--use-phenotypic"]

    for experiment in experiments:
        clf_type = arguments["--classifier"]
        metrics, best_params, scaler = run_classifier(hdf5, experiment, clf_type, use_phenotypic=use_phenotypic)
        experiment_results.append([experiment] + metrics)
        best_params_results[experiment] = best_params  # Store best params for each experiment


    # Display all results in a table
    print(tabulate.tabulate(experiment_results, headers=["exp", "acc", "prec", "recall", "fscore", "sens", "spec"]))

    # Display the best parameters for each experiment
    print("\nBest parameters for each experiment:")
    for experiment, best_params in best_params_results.items():
        print(f"Experiment: {experiment}, Best Parameters: {best_params}")