"""
eda_utils.py
Reusable EDA plotting functions for Task 1
Adey Innovations Inc. - Fraud Detection Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── Consistent style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
COLORS = {"legit": "#4C72B0", "fraud": "#DD8452"}
FIG_DIR = "plots/"
os.makedirs(FIG_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# CLASS IMBALANCE
# ─────────────────────────────────────────────

def plot_class_distribution(y, dataset_name="Dataset", label_col="class"):
    """Bar chart of class counts + percentages."""
    counts = y.value_counts()
    pcts = y.value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Class Distribution — {dataset_name}", fontsize=14, fontweight="bold")

    # Count bar
    axes[0].bar(["Legitimate", "Fraud"], counts.values,
                color=[COLORS["legit"], COLORS["fraud"]], edgecolor="white")
    axes[0].set_title("Transaction Counts")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + counts.max() * 0.01, f"{v:,}", ha="center", fontsize=11)

    # Percentage pie
    axes[1].pie(pcts.values, labels=["Legitimate", "Fraud"],
                colors=[COLORS["legit"], COLORS["fraud"]],
                autopct="%1.2f%%", startangle=90)
    axes[1].set_title("Percentage Split")

    save(fig, f"class_distribution_{dataset_name.lower().replace(' ', '_')}.png")


# ─────────────────────────────────────────────
# NUMERICAL DISTRIBUTIONS
# ─────────────────────────────────────────────

def plot_numeric_distributions(df, cols, target_col, dataset_name="Dataset"):
    """Overlaid histograms for each numeric column split by class."""
    n = len(cols)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4))
    axes = axes.flatten()
    fig.suptitle(f"Numeric Distributions by Class — {dataset_name}", fontsize=14, fontweight="bold")

    for i, col in enumerate(cols):
        for label, color, name in [(0, COLORS["legit"], "Legitimate"), (1, COLORS["fraud"], "Fraud")]:
            subset = df[df[target_col] == label][col].dropna()
            axes[i].hist(subset, bins=40, alpha=0.6, color=color, label=name, density=True)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")
        axes[i].legend()

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save(fig, f"numeric_distributions_{dataset_name.lower().replace(' ', '_')}.png")


# ─────────────────────────────────────────────
# CATEGORICAL DISTRIBUTIONS
# ─────────────────────────────────────────────

def plot_categorical_fraud_rate(df, cols, target_col, dataset_name="Dataset"):
    """Fraud rate per category value for each categorical column."""
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Fraud Rate by Category — {dataset_name}", fontsize=14, fontweight="bold")

    for ax, col in zip(axes, cols):
        rates = df.groupby(col)[target_col].mean().sort_values(ascending=False)
        bars = ax.bar(rates.index, rates.values * 100,
                      color=sns.color_palette("Set2", len(rates)))
        ax.set_title(col)
        ax.set_ylabel("Fraud Rate (%)")
        ax.set_xlabel(col)
        ax.tick_params(axis="x", rotation=30)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%",
                    ha="center", fontsize=9)

    plt.tight_layout()
    save(fig, f"categorical_fraud_rate_{dataset_name.lower().replace(' ', '_')}.png")


# ─────────────────────────────────────────────
# GEOLOCATION
# ─────────────────────────────────────────────

def plot_top_fraud_countries(df, country_col="country", target_col="class", top_n=15):
    """Horizontal bar chart of top N countries by fraud rate."""
    fraud_by_country = (
        df.groupby(country_col)[target_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "fraud_rate", "count": "tx_count"})
        .query("tx_count >= 10")  # exclude tiny samples
        .sort_values("fraud_rate", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(fraud_by_country.index[::-1],
                   fraud_by_country["fraud_rate"][::-1] * 100,
                   color=COLORS["fraud"])
    ax.set_xlabel("Fraud Rate (%)")
    ax.set_title(f"Top {top_n} Countries by Fraud Rate (min 10 transactions)")
    for bar in bars:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    save(fig, "top_fraud_countries.png")


# ─────────────────────────────────────────────
# ENGINEERED FEATURES
# ─────────────────────────────────────────────

def plot_time_since_signup(df, target_col="class"):
    """Boxplot of time_since_signup by fraud class."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=target_col, y="time_since_signup", data=df, ax=ax,
                palette=[COLORS["legit"], COLORS["fraud"]])
    ax.set_xticklabels(["Legitimate", "Fraud"])
    ax.set_ylabel("Hours Since Signup")
    ax.set_title("Time Since Signup vs. Fraud")
    plt.tight_layout()
    save(fig, "time_since_signup_vs_fraud.png")


def plot_hour_of_day(df, target_col="class"):
    """Line chart of fraud rate by hour of day."""
    hourly = df.groupby("hour_of_day")[target_col].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly.index, hourly.values, marker="o", color=COLORS["fraud"])
    ax.fill_between(hourly.index, hourly.values, alpha=0.2, color=COLORS["fraud"])
    ax.set_xlabel("Hour of Day (0 = midnight)")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate by Hour of Day")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    plt.tight_layout()
    save(fig, "fraud_by_hour.png")


def plot_velocity(df, target_col="class"):
    """Boxplots of user_tx_count and device_tx_count vs fraud."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Transaction Velocity vs. Fraud", fontsize=13, fontweight="bold")

    for ax, col in zip(axes, ["user_tx_count", "device_tx_count"]):
        sns.boxplot(x=target_col, y=col, data=df, ax=ax,
                    palette=[COLORS["legit"], COLORS["fraud"]])
        ax.set_xticklabels(["Legitimate", "Fraud"])
        ax.set_title(col.replace("_", " ").title())
        ax.set_ylabel("Transaction Count")

    plt.tight_layout()
    save(fig, "velocity_vs_fraud.png")


# ─────────────────────────────────────────────
# SMOTE BALANCE
# ─────────────────────────────────────────────

def plot_smote_comparison(y_before, y_after, dataset_name="Dataset"):
    """Side-by-side bar chart showing class distribution before/after SMOTE."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Class Balance Before vs After SMOTE — {dataset_name}", fontsize=13, fontweight="bold")

    for ax, y, title in zip(axes, [y_before, pd.Series(y_after)], ["Before SMOTE", "After SMOTE"]):
        counts = pd.Series(y).value_counts()
        ax.bar(["Legitimate", "Fraud"], [counts.get(0, 0), counts.get(1, 0)],
               color=[COLORS["legit"], COLORS["fraud"]], edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("Count")
        for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
            ax.text(i, v + max(counts.values) * 0.01, f"{v:,}", ha="center", fontsize=11)

    plt.tight_layout()
    save(fig, f"smote_comparison_{dataset_name.lower().replace(' ', '_')}.png")


# ─────────────────────────────────────────────
# CORRELATION HEATMAP  (creditcard)
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df, dataset_name="creditcard", top_n=15):
    """Heatmap of top N features most correlated with the target."""
    target = "Class" if "Class" in df.columns else "class"
    corr = df.corr()[target].drop(target).abs().sort_values(ascending=False).head(top_n)
    corr_matrix = df[corr.index.tolist() + [target]].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title(f"Top {top_n} Feature Correlations — {dataset_name}", fontsize=13)
    plt.tight_layout()
    save(fig, f"correlation_heatmap_{dataset_name}.png")
