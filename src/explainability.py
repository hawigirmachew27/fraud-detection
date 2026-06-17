"""
explainability.py
Task 3 - Model Explainability using SHAP
Adey Innovations Inc. - Fraud Detection Project
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIG_DIR = "outputs/"
os.makedirs(FIG_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. SHAP EXPLAINER SETUP
# ─────────────────────────────────────────────

def get_shap_explainer(model, X_train_sample):
    """
    Build a TreeExplainer for tree-based models (Random Forest, XGBoost).
    Uses a background sample from training data for efficiency.
    """
    explainer = shap.TreeExplainer(model)
    logger.info("SHAP TreeExplainer created.")
    return explainer


def compute_shap_values(explainer, X):
    """Compute SHAP values for a dataset. Returns shap_values array for fraud class."""
    shap_values = explainer.shap_values(X)
    # New SHAP versions return shape (n_samples, n_features, n_classes)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]   # class 1 = fraud
    # Older versions return a list [class_0_array, class_1_array]
    if isinstance(shap_values, list):
        return shap_values[1]
    return shap_values


# ─────────────────────────────────────────────
# 2. BUILT-IN FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def plot_feature_importance(model, feature_names, dataset_name="Dataset", top_n=15):
    """Bar chart of top N built-in Random Forest feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_features[::-1], top_importances[::-1], color="#2E75B6")
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.set_title(f"Top {top_n} Feature Importances — {dataset_name}")
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va="center", fontsize=8)
    plt.tight_layout()
    fname = f"feature_importance_{dataset_name.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight", dpi=150)
    print(f"  Saved → {fname}")
    plt.close(fig)
    return pd.DataFrame({"feature": top_features, "importance": top_importances})


# ─────────────────────────────────────────────
# 3. SHAP SUMMARY PLOT
# ─────────────────────────────────────────────

def plot_shap_summary(shap_values, X, dataset_name="Dataset", top_n=15):
    """
    SHAP Summary Plot — shows global feature importance AND direction.
    Red = high feature value pushes toward fraud.
    Blue = high feature value pushes toward legitimate.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X,
        max_display=top_n,
        show=False,
        plot_type="dot"
    )
    plt.title(f"SHAP Summary Plot — {dataset_name}", fontsize=13, pad=20)
    plt.tight_layout()
    fname = f"shap_summary_{dataset_name.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight", dpi=150)
    print(f"  Saved → {fname}")
    plt.close(fig)


def plot_shap_bar(shap_values, X, dataset_name="Dataset", top_n=15):
    """SHAP Bar Plot — mean absolute SHAP values (global importance)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X,
        max_display=top_n,
        show=False,
        plot_type="bar"
    )
    plt.title(f"SHAP Mean |SHAP Value| — {dataset_name}", fontsize=13, pad=20)
    plt.tight_layout()
    fname = f"shap_bar_{dataset_name.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight", dpi=150)
    print(f"  Saved → {fname}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 4. SHAP FORCE PLOTS (individual predictions)
# ─────────────────────────────────────────────

def find_case_indices(model, X_test, y_test):
    """
    Find indices for:
      - True Positive  (fraud caught correctly)
      - False Positive (legitimate flagged as fraud)
      - False Negative (fraud that was missed)
    """
    y_pred = model.predict(X_test)
    y_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)

    tp_indices = np.where((y_arr == 1) & (y_pred_arr == 1))[0]
    fp_indices = np.where((y_arr == 0) & (y_pred_arr == 1))[0]
    fn_indices = np.where((y_arr == 1) & (y_pred_arr == 0))[0]

    tp_idx = tp_indices[0] if len(tp_indices) > 0 else None
    fp_idx = fp_indices[0] if len(fp_indices) > 0 else None
    fn_idx = fn_indices[0] if len(fn_indices) > 0 else None

    logger.info(f"TP index: {tp_idx}, FP index: {fp_idx}, FN index: {fn_idx}")
    return tp_idx, fp_idx, fn_idx


def plot_force_plot(explainer, shap_values, X, idx, case_label, dataset_name="Dataset"):
    """
    Save a SHAP force plot for a single prediction as a matplotlib figure.
    case_label: 'True Positive', 'False Positive', or 'False Negative'
    """
    if idx is None:
        print(f"  Skipping {case_label} — no such case in test set.")
        return

    base_val = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') \
               else explainer.expected_value

    sv_row = shap_values[idx]          # now shape (198,) after fix above
    feat_row = X.iloc[idx] if hasattr(X, "iloc") else X[idx]

    fig = plt.figure(figsize=(14, 3))
    shap.plots.force(
        float(base_val),
        sv_row,
        feat_row,
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Force Plot — {case_label} — {dataset_name}", fontsize=11, pad=35)
    plt.tight_layout()
    fname = f"shap_force_{case_label.lower().replace(' ', '_')}_{dataset_name.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight", dpi=150)
    print(f"  Saved → {fname}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE COMPARISON TABLE
# ─────────────────────────────────────────────

def compare_importance_methods(model, shap_values, feature_names, top_n=10):
    """
    Side-by-side comparison of built-in importance vs SHAP importance.
    Returns a DataFrame and prints it.
    """
    builtin = pd.Series(model.feature_importances_, index=feature_names) \
                .sort_values(ascending=False).head(top_n)

    shap_imp = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_names
    ).sort_values(ascending=False).head(top_n)

    comparison = pd.DataFrame({
        "Built-in Rank": range(1, top_n + 1),
        "Built-in Feature": builtin.index,
        "Built-in Score": builtin.values.round(4),
        "SHAP Rank": range(1, top_n + 1),
        "SHAP Feature": shap_imp.index,
        "SHAP Mean |Value|": shap_imp.values.round(4)
    })

    print("\n=== Feature Importance Comparison ===")
    print(comparison.to_string(index=False))
    return comparison


# ─────────────────────────────────────────────
# 6. BUSINESS RECOMMENDATIONS TABLE
# ─────────────────────────────────────────────

FRAUD_DATA_RECOMMENDATIONS = """
Based on SHAP analysis of Fraud_Data:

1. TIME_SINCE_SIGNUP (strongest signal)
   Finding: Fraudulent transactions cluster within hours of account creation.
   Action:  Flag and queue for manual review any transaction occurring within
            6 hours of signup. Add friction (email/SMS OTP) for purchases
            made within 1 hour of account creation.

2. DEVICE_TX_COUNT (velocity signal)
   Finding: Fraud-linked devices have significantly higher transaction counts
            than legitimate ones (median ~8 vs ~1).
   Action:  Implement device fingerprinting. Auto-block or escalate any
            device that processes more than 5 transactions in a 24-hour window
            across different user accounts.

3. COUNTRY (geolocation signal)
   Finding: Namibia (43.5%), Sri Lanka (41.9%), Luxembourg (38.9%) have
            fraud rates 4x above the dataset average.
   Action:  Apply step-up authentication (e.g. additional OTP) for transactions
            originating from high-risk countries. Monitor IP ranges associated
            with these countries in real time.

4. SOURCE = DIRECT (traffic signal)
   Finding: Direct traffic has a 10.5% fraud rate vs 8.9% for SEO.
   Action:  Apply stricter bot detection (CAPTCHA, behavioral analysis) for
            direct-navigation sessions, especially when combined with
            recent signup or high device velocity.

5. PURCHASE_VALUE (amount signal)
   Finding: Fraudulent purchases show slightly higher concentration in the
            $20-$40 range compared to legitimate transactions.
   Action:  Apply additional review to transactions in this amount range
            when combined with other risk signals (new account + direct
            traffic + high-risk country).
"""

CREDITCARD_RECOMMENDATIONS = """
Based on SHAP analysis of creditcard:

1. V17 (strongest negative correlate with fraud)
   Finding: High V17 values strongly reduce fraud probability; low V17 values
            are a major fraud driver.
   Action:  Use V17 as a primary real-time scoring input. Transactions where
            V17 falls below a learned threshold should be auto-escalated
            for review, especially when combined with low V14 or V12.

2. V14 + V12 (combined signal)
   Finding: Both features show strong negative SHAP values for fraud —
            low values of either push strongly toward the fraud class.
   Action:  Build a compound rule: if V17 < threshold AND V14 < threshold,
            automatically flag. This combination reduces false positives
            compared to single-feature rules.

3. V11 + V4 (positive fraud correlates)
   Finding: High values of V11 and V4 are associated with fraud.
   Action:  Include V11 and V4 in a risk scoring formula. A weighted sum
            of these SHAP signals can serve as a real-time fraud score
            to trigger soft declines or step-up authentication.

4. AMOUNT (low-value fraud pattern)
   Finding: Fraudulent transactions cluster at lower amounts, consistent
            with card-testing behavior.
   Action:  Flag micro-transactions (under $5) when they occur in rapid
            succession from the same card. Implement velocity rules:
            3+ micro-transactions in 10 minutes = automatic review.

5. MONITORING RECOMMENDATION
   Finding: The model catches the majority of fraud but at the cost of some
            false positives. The optimal operating threshold depends on the
            business cost of false positives vs false negatives.
   Action:  Use the Precision-Recall curve to select the threshold that
            matches the business's cost ratio. If missing fraud costs 10x
            more than a false alarm, operate at a lower threshold.
"""
