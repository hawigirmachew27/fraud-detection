"""
preprocess.py
Task 1 - Data Cleaning, Feature Engineering, and Class Imbalance Handling
Adey Innovations Inc. - Fraud Detection Project
"""

import pandas as pd
import numpy as np
import struct
import socket
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. LOADING
# ─────────────────────────────────────────────

def load_fraud_data(path="data/raw/Fraud_Data.csv"):
    df = pd.read_csv(path)
    logger.info(f"Fraud_Data loaded: {df.shape}")
    return df


def load_ip_map(path="data/raw/IpAddress_to_Country.csv"):
    df = pd.read_csv(path)
    logger.info(f"IP map loaded: {df.shape}")
    return df


def load_creditcard(path="data/raw/creditcard.csv"):
    df = pd.read_csv(path)
    logger.info(f"creditcard loaded: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 2. CLEANING
# ─────────────────────────────────────────────

def clean_fraud_data(df):
    """Clean Fraud_Data.csv: fix dtypes, remove duplicates, handle nulls."""
    df = df.copy()

    # Fix datetime columns
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Fraud_Data: dropped {before - len(df)} duplicate rows")

    # Handle nulls
    null_counts = df.isnull().sum()
    logger.info(f"Null counts:\n{null_counts[null_counts > 0]}")
    df = df.dropna()   # Drop rows with any nulls (small fraction expected)

    logger.info(f"Fraud_Data after cleaning: {df.shape}")
    return df


def clean_creditcard(df):
    """Clean creditcard.csv: remove duplicates, handle nulls."""
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"creditcard: dropped {before - len(df)} duplicate rows")

    null_counts = df.isnull().sum()
    logger.info(f"Null counts:\n{null_counts[null_counts > 0]}")
    df = df.dropna()

    logger.info(f"creditcard after cleaning: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 3. GEOLOCATION  (Fraud_Data only)
# ─────────────────────────────────────────────

def ip_to_int(ip):
    """Convert dotted IP string to integer."""
    try:
        return struct.unpack("!I", socket.inet_aton(str(ip)))[0]
    except Exception:
        return np.nan


def merge_geolocation(fraud_df, ip_map_df):
    """
    Map each transaction's IP address to a country using a range lookup.
    Uses merge_asof for efficient range-based matching.
    """
    fraud_df = fraud_df.copy()
    ip_map_df = ip_map_df.copy()

    # Convert IPs to integers (fraud_df only — ip_map already has integers)
    fraud_df["ip_int"] = fraud_df["ip_address"].apply(ip_to_int)
    ip_map_df["lower_int"] = ip_map_df["lower_bound_ip_address"].astype(float)
    ip_map_df["upper_int"] = ip_map_df["upper_bound_ip_address"].astype(float)

    # Sort both for merge_asof
    fraud_df = fraud_df.dropna(subset=["ip_int"])
    ip_map_df = ip_map_df.dropna(subset=["lower_int", "upper_int"])
    fraud_sorted = fraud_df.sort_values("ip_int").reset_index(drop=True)    
    ip_sorted = ip_map_df.sort_values("lower_int").reset_index(drop=True)

    merged = pd.merge_asof(
        fraud_sorted,
        ip_sorted[["lower_int", "upper_int", "country"]],
        left_on="ip_int",
        right_on="lower_int",
        direction="backward"
    )

    # Keep only valid matches (ip must fall within range)
    merged = merged[merged["ip_int"] <= merged["upper_int"]].copy()
    merged["country"] = merged["country"].fillna("Unknown")

    logger.info(f"After geolocation merge: {merged.shape}")
    return merged


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING  (Fraud_Data only)
# ─────────────────────────────────────────────

def engineer_features(df):
    """Add time-based and velocity features to Fraud_Data."""
    df = df.copy()

    # Time between signup and purchase (hours)
    df["time_since_signup"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds() / 3600

    # Time-of-day features
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek   # 0=Mon, 6=Sun

    # Transaction velocity per user and device
    df["user_tx_count"] = df.groupby("user_id")["user_id"].transform("count")
    df["device_tx_count"] = df.groupby("device_id")["device_id"].transform("count")

    logger.info("Feature engineering done: time_since_signup, hour_of_day, day_of_week, user_tx_count, device_tx_count")
    return df


# ─────────────────────────────────────────────
# 5. ENCODING & SCALING
# ─────────────────────────────────────────────

# Columns to drop before modeling
FRAUD_DROP_COLS = ["user_id", "device_id", "ip_address", "signup_time",
                   "purchase_time", "ip_int", "lower_int", "upper_int"]

FRAUD_NUM_COLS = ["purchase_value", "age", "time_since_signup",
                  "user_tx_count", "device_tx_count", "hour_of_day", "day_of_week"]

FRAUD_CAT_COLS = ["source", "browser", "sex", "country"]

CC_NUM_COLS = ["Amount", "Time"]   # V1-V28 already PCA-scaled


def encode_and_scale_fraud(df):
    """One-hot encode categoricals and scale numerics for Fraud_Data."""
    df = df.copy()

    # Drop non-feature columns
    df = df.drop(columns=[c for c in FRAUD_DROP_COLS if c in df.columns])

    # One-hot encode
    df = pd.get_dummies(df, columns=FRAUD_CAT_COLS, drop_first=False)

    # Scale numerics
    scaler = StandardScaler()
    cols_to_scale = [c for c in FRAUD_NUM_COLS if c in df.columns]
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    logger.info(f"Fraud_Data after encoding & scaling: {df.shape}")
    return df, scaler


def encode_and_scale_creditcard(df):
    """Scale Amount and Time for creditcard.csv (V1-V28 already scaled)."""
    df = df.copy()

    scaler = StandardScaler()
    df[CC_NUM_COLS] = scaler.fit_transform(df[CC_NUM_COLS])

    logger.info(f"creditcard after scaling: {df.shape}")
    return df, scaler


# ─────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT + SMOTE
# ─────────────────────────────────────────────

def split_and_resample(df, target_col, test_size=0.2, random_state=42):
    """
    Stratified train/test split then apply SMOTE only on training data.

    Why SMOTE over undersampling:
      Fraud is a rare event (~1-9% of records). Undersampling discards
      most of the legitimate transaction data. SMOTE generates synthetic
      minority-class samples, preserving all real data while balancing
      the classes for training.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Stratified split preserves class ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Class distribution before SMOTE:\n{y_train.value_counts()}")

    # SMOTE only on training data — NEVER on test data
    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    logger.info(f"Class distribution after SMOTE:\n{pd.Series(y_train_res).value_counts()}")

    return X_train_res, X_test, y_train_res, y_test


# ─────────────────────────────────────────────
# 7. FULL PIPELINES
# ─────────────────────────────────────────────

def run_fraud_pipeline(
    fraud_path="data/raw/Fraud_Data.csv",
    ip_path="data/raw/IpAddress_to_Country.csv",
    save_path="data/processed/fraud_processed.csv"
):
    fraud = load_fraud_data(fraud_path)
    ip_map = load_ip_map(ip_path)

    fraud = clean_fraud_data(fraud)
    fraud = merge_geolocation(fraud, ip_map)
    fraud = engineer_features(fraud)
    fraud, scaler = encode_and_scale_fraud(fraud)

    fraud.to_csv(save_path, index=False)
    logger.info(f"Fraud pipeline complete. Saved to {save_path}")
    return fraud, scaler


def run_creditcard_pipeline(
    cc_path="data/raw/creditcard.csv",
    save_path="data/processed/creditcard_processed.csv"
):
    cc = load_creditcard(cc_path)
    cc = clean_creditcard(cc)
    cc, scaler = encode_and_scale_creditcard(cc)

    cc.to_csv(save_path, index=False)
    logger.info(f"Creditcard pipeline complete. Saved to {save_path}")
    return cc, scaler


if __name__ == "__main__":
    # Run both pipelines end-to-end
    fraud_df, _ = run_fraud_pipeline()
    cc_df, _ = run_creditcard_pipeline()

    # Split and resample both
    X_tr_f, X_te_f, y_tr_f, y_te_f = split_and_resample(fraud_df, target_col="class")
    X_tr_c, X_te_c, y_tr_c, y_te_c = split_and_resample(cc_df, target_col="Class")

    print("\n✅ Task 1 complete. Processed files saved to data/processed/")
