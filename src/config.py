"""
config.py

Central configuration file for the
semantic property matching system.
"""

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Data paths
DATA_PATH = "data/raw/Case_Study_2_Data.xlsx"
OUTPUT_DIR = "outputs"

# Matching configuration
TOP_K = 5

# Column names
USER_ID_COL = "User ID"
PROPERTY_ID_COL = "Property ID"
