"""
modeling.py
Task 2 - Model Building, Training, and Evaluation
Adey Innovations Inc. - Fraud Detection Project
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score, average_precision_score, confusion_matrix,
    precision_recall_curve, ConfusionMatrixDisplay
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIG_DIR = "outputs/"
MODEL_DIR = "models/"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. MODEL TRAINING
# ─────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, random_state=42):
    """Train a baseline Logistic Regression model."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logger.info("Logistic Regression trained.")
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """Train a Random Forest ensemble model."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight=None  # SMOTE already balances the training data
    )
    model.fit(X_train, y_train)
    logger.info(f"Random Forest trained (n_estimators={n_estimators}, max_depth={max_depth}).")
    return model


# ─────────────────────────────────────────────
# 2. EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a model using AUC-PR, F1-Score, and Confusion Matrix.
    Returns a dict of results for comparison tables.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_pr = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"{model_name} -> AUC-PR: {auc_pr:.4f}, F1: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    return {
        "model_name": model_name,
        "auc_pr": auc_pr,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba
    }


def plot_confusion_matrix(cm, model_name="Model", dataset_name="Dataset"):
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraud"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name} ({dataset_name})")
    plt.tight_layout()
    fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}_{dataset_name.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight", dpi=150)
    print(f"  Saved → {os.path.join(FIG_DIR, fname)}")
    plt.close(fig)


def plot_precision_recall_curve(results_list, dataset_name="Dataset"):
    """Plot precision-recall curves for multiple models on the same axes."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for result in results_list:
        precision, recall, _ = precision_recall_curve(
            result["y_test"], result["y_proba"]
        )
        ax.plot(recall, precision, label=f"{result['model_name']} (AUC-PR={result['auc_pr']:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {dataset_name}")
    ax.legend()
    plt.tight_layout()
    fname = f"precision_recall_{dataset_name.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight", dpi=150)
    print(f"  Saved → {os.path.join(FIG_DIR, fname)}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 3. CROSS-VALIDATION
# ─────────────────────────────────────────────

def cross_validate_model(model_fn, X, y, k=5, random_state=42):
    """
    Run Stratified K-Fold cross-validation, training a fresh model each fold.
    model_fn: a no-arg function that returns a NEW unfit model instance.
    Returns mean/std of AUC-PR and F1 across folds.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    auc_pr_scores, f1_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val_fold = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_train_fold = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        y_val_fold = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]

        model = model_fn()
        model.fit(X_train_fold, y_train_fold)

        y_proba = model.predict_proba(X_val_fold)[:, 1]
        y_pred = model.predict(X_val_fold)

        auc_pr = average_precision_score(y_val_fold, y_proba)
        f1 = f1_score(y_val_fold, y_pred)

        auc_pr_scores.append(auc_pr)
        f1_scores.append(f1)

        logger.info(f"Fold {fold}: AUC-PR={auc_pr:.4f}, F1={f1:.4f}")

    return {
        "auc_pr_mean": np.mean(auc_pr_scores),
        "auc_pr_std": np.std(auc_pr_scores),
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
        "auc_pr_scores": auc_pr_scores,
        "f1_scores": f1_scores
    }


# ─────────────────────────────────────────────
# 4. MODEL COMPARISON
# ─────────────────────────────────────────────

def build_comparison_table(results_list):
    """
    Build a comparison DataFrame from a list of evaluate_model() results
    (each dict should also contain 'cv_auc_pr_mean', 'cv_auc_pr_std',
    'cv_f1_mean', 'cv_f1_std' if cross-validation was run).
    """
    rows = []
    for r in results_list:
        row = {
            "Model": r["model_name"],
            "Test AUC-PR": round(r["auc_pr"], 4),
            "Test F1-Score": round(r["f1_score"], 4),
        }
        if "cv_auc_pr_mean" in r:
            row["CV AUC-PR (mean ± std)"] = f"{r['cv_auc_pr_mean']:.4f} ± {r['cv_auc_pr_std']:.4f}"
            row["CV F1 (mean ± std)"] = f"{r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 5. SAVE / LOAD MODELS
# ─────────────────────────────────────────────

def save_model(model, name, dataset_name="dataset"):
    """Save a trained model to disk."""
    path = os.path.join(MODEL_DIR, f"{name.lower().replace(' ', '_')}_{dataset_name.lower().replace(' ', '_')}.joblib")
    joblib.dump(model, path)
    logger.info(f"Saved model -> {path}")
    return path


def load_model(path):
    return joblib.load(path)
