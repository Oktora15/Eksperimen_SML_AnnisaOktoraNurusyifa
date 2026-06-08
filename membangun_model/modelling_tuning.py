import argparse
import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def load_data(path, target_column=None):
    df = pd.read_csv(path)
    if target_column and target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns=[target_column])
    else:
        # default: last column is target
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    return X, y


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance(feature_names, importances, out_path, top_n=30):
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    fi = fi.head(top_n)
    plt.figure(figsize=(8, min(0.25 * len(fi) + 2, 12)))
    sns.barplot(x=fi.values, y=fi.index)
    plt.title('Feature importances')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(args):
    # Set MLflow tracking URI if provided (useful for DagsHub or remote tracking)
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    # Create or set experiment
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)

    X, y = load_data(args.data_path, args.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if args.stratify else None
    )

    # Define estimator and param distribution
    estimator = RandomForestClassifier(random_state=args.random_state)
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        cv=args.cv,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=1,
    )

    # Fit search
    search.fit(X_train, y_train)

    best = search.best_estimator_

    # Evaluate on test
    y_pred = best.predict(X_test)
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
    metrics['precision'] = float(precision_score(y_test, y_pred, average='binary' if len(np.unique(y))==2 else 'weighted', zero_division=0))
    metrics['recall'] = float(recall_score(y_test, y_pred, average='binary' if len(np.unique(y))==2 else 'weighted', zero_division=0))
    metrics['f1_score'] = float(f1_score(y_test, y_pred, average='binary' if len(np.unique(y))==2 else 'weighted', zero_division=0))
    # ROC-AUC only for binary
    if len(np.unique(y)) == 2:
        try:
            proba = best.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = float(roc_auc_score(y_test, proba))
        except Exception:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None

    # Start MLflow run and log manually
    with mlflow.start_run(run_name=args.run_name):
        # Log model hyperparameters (best params)
        for k, v in search.best_params_.items():
            mlflow.log_param(k, v)

        # Log search settings
        mlflow.log_param('n_iter', args.n_iter)
        mlflow.log_param('cv', args.cv)

        # Log metrics
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)

        # Log model
        mlflow.sklearn.log_model(best, artifact_path='model')

        # Create temporary folder for artifacts
        with tempfile.TemporaryDirectory() as td:
            # Confusion matrix
            cm_path = os.path.join(td, 'confusion_matrix.png')
            plot_confusion_matrix(y_test, y_pred, cm_path)
            mlflow.log_artifact(cm_path, artifact_path='artifacts')

            # Feature importance (if available)
            try:
                importances = best.feature_importances_
                fi_path = os.path.join(td, 'feature_importance.png')
                plot_feature_importance(X.columns, importances, fi_path)
                mlflow.log_artifact(fi_path, artifact_path='artifacts')
            except Exception:
                pass

            # Sample predictions
            sample_df = X_test.copy()
            sample_df['_y_true'] = y_test.values
            sample_df['_y_pred'] = y_pred
            sample_csv = os.path.join(td, 'sample_predictions.csv')
            sample_df.head(200).to_csv(sample_csv, index=False)
            mlflow.log_artifact(sample_csv, artifact_path='artifacts')

    print('Run completed. Best params:', search.best_params_)
    print('Metrics:', metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='dataset_preprocessing.csv', help='Path to preprocessed CSV')
    parser.add_argument('--target', type=str, default=None, help='Target column name (default: last column)')
    parser.add_argument('--experiment-name', type=str, default='modelling_tuning', help='MLflow experiment name')
    parser.add_argument('--tracking-uri', type=str, default=None, help='MLflow tracking URI (optional)')
    parser.add_argument('--run-name', type=str, default='modelling_tuning_run', help='MLflow run name')
    parser.add_argument('--n-iter', type=int, default=10, help='Number of iterations for RandomizedSearchCV')
    parser.add_argument('--cv', type=int, default=3, help='CV folds')
    parser.add_argument('--n-jobs', type=int, default=-1, help='n_jobs for search')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--stratify', action='store_true', help='Stratify split by target')

    args = parser.parse_args()
    main(args)
