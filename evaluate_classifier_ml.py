import os
import joblib
import numpy as np
import tabulate
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from utils import (format_config, hdf5_handler, load_fold, load_phenotypes,
                   reset)


def get_classifier(clf_type, params):

    if clf_type == "random_forest":
        return RandomForestClassifier(**params)
    elif clf_type == "logistic_regression":
        return LogisticRegression(**params, max_iter=100)
    elif clf_type == "svm":
        return SVC(**params)
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")


def run(X_train,y_train,X_valid,y_valid,clf_type="svm",params=None,use_phenotypic=False):
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

    return [accuracy, precision, recall, fscore, sensitivity, specificity], scaler, clf


def retrain_and_evaluate(hdf5_path, pheno_path, clf_type_dict, best_params_dict, use_phenotypic=False):
    # Load phenotypic data
    pheno = load_phenotypes(pheno_path)
    hdf5 = hdf5_handler(bytes(hdf5_path, encoding="utf8"), "a")

    # Define the derivatives and experiments
    derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    experiments = []

    for derivative in derivatives:
        config = {"derivative": derivative}
        experiments += [format_config("{derivative}_whole", config)]

    # Create the models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    results = []

    for experiment in experiments:
        print("Training started for:",experiment)


        X_train_all, y_train_all, X_test_all, y_test_all = load_the_data(
            hdf5, experiment
        )

        clf_type = clf_type_dict.get(experiment)
        best_params = best_params_dict.get(experiment, {})
        metrics, scaler, clf = run(
            X_train_all,
            y_train_all,
            X_test_all,
            y_test_all,
            clf_type,
            best_params,
            use_phenotypic,
        )

        # Save the model in the models directory
        model_filename = f"models/best_model_{experiment}.joblib"
        joblib.dump(clf, model_filename)
        print(f"Model saved as {model_filename}")

        # Evaluate the model
        y_pred = clf.predict(X_test_all)
        accuracy = accuracy_score(y_test_all, y_pred)
        report = classification_report(y_test_all, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test_all, y_pred)

        results.append(
            [
                experiment,
                accuracy,
                report["macro avg"]["precision"],
                report["macro avg"]["recall"],
                report["macro avg"]["f1-score"],
                conf_matrix,
            ]
        )

    # Print the results in a tabulated format
    headers = [
        "Experiment",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "Confusion Matrix",
    ]
    table = tabulate.tabulate(results, headers, tablefmt="grid")
    print(table)


def load_the_data(hdf5, experiment):
    # Initialize lists to store the combined data

    # Load the train, validation, and test data for the current fold
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_fold(
        hdf5["patients"], hdf5["experiments"]["cc200_whole"], "0"
    )

    # Combine train, validation, and test data for this fold
    X_combined = np.concatenate((X_train, X_valid), axis=0)
    y_combined = np.concatenate((y_train, y_valid), axis=0)



    # Log the shapes for debugging
    print("Train Shape:")
    print("X_combined shape:", X_combined.shape)
    print("y_combined length:", len(y_combined))
 
    print("-"*40)

    print("Test Shapes:")
    print("X_all_combined shape:", X_test.shape)
    print("y_all_combined length:", len(y_test))

    return X_combined, y_combined, X_test, y_test


if __name__ == "__main__":
    reset()
    hdf5_path = "./data/abide.hdf5"
    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"

    # Define the best parameters for each experiment
    best_params_dict = {
        "cc200_whole": {"C": 0.1},
        "aal_whole": {"C": 0.1},
        "ez_whole": {"C": 0.1},
        "ho_whole": {"C": 0.1},
        "tt_whole": {"C": 0.1},
        "dosenbach160_whole": {"C": 0.1},
    }

    # Define the classifier type for each experiment
    clf_type_dict = {
        "cc200_whole": "logistic_regression",
        "aal_whole": "logistic_regression",
        "ez_whole": "logistic_regression",
        "ho_whole": "logistic_regression",
        "tt_whole": "logistic_regression",
        "dosenbach160_whole": "logistic_regression",
    }

    retrain_and_evaluate(
        hdf5_path, pheno_path, clf_type_dict, best_params_dict, use_phenotypic=True
    )
