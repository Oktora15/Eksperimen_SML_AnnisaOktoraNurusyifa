import os
import shutil
from pathlib import Path

import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# SETUP DAGSHUB (ONLINE TRACKING - ADVANCE)
# =====================================================
dagshub.init(
    repo_owner="Oktora15",  # GANTI jika perlu
    repo_name="Eksperimen_SML_AnnisaOktoraNurusyifa",
    mlflow=True
)

mlflow.set_experiment("Heart_Disease_Experiment")

# =====================================================
# LOAD DATA
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset_preprocessing.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"File tidak ditemukan: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]

# =====================================================
# SPLIT DATA
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# TRAINING + MANUAL LOGGING (ADVANCE)
# =====================================================
with mlflow.start_run():

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # ===== LOG PARAMETER =====
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # ===== LOG METRIC =====
    mlflow.log_metric("accuracy", acc)

    # ===== ARTIFACT 1: Confusion Matrix =====
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # ===== ARTIFACT 2: Feature Importance =====
    importances = model.feature_importances_
    plt.figure()
    plt.bar(range(len(importances)), importances)
    plt.title("Feature Importance")
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    # ===== SAVE MODEL =====
    local_model_dir = "model"

    mlflow.sklearn.save_model(model, local_model_dir)
    mlflow.log_artifacts(local_model_dir, artifact_path="model")

    shutil.rmtree(local_model_dir)

print("Training selesai dan berhasil dilog ke DagsHub.")