"""
src/eda_utils.py
----------------
Reusable EDA helper functions for the Datasets
Fraud Detection project
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

import pandas as pd
import numpy as np


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load and perform initial cleaning of the FNSPID news CSV.

    Parameters
    ----------
    filepath : str
        Path to raw_analyst_ratings.csv

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with typed columns.
    """
    df = pd.read_csv(filepath)

   

    return df