"""
config.py

Central configuration file for the
semantic property matching system.
"""

# Embedding model
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# Data paths
DATA_PATH = "data/raw/Case_Study_2_Data.xlsx"
OUTPUT_DIR = "outputs"

# Matching configuration
TOP_K = 5

# Hybrid scoring weights
SEMANTIC_WEIGHT = 0.7
NUMERICAL_WEIGHT = 0.3

# Numerical feature tolerances / flexibility
BUDGET_TOLERANCE = 0.20
BEDROOM_FLEX = 1
BATHROOM_FLEX = 1
LIVING_AREA_TOLERANCE = 0.15

# Column names
USER_ID_COL = "User ID"
PROPERTY_ID_COL = "Property ID"
