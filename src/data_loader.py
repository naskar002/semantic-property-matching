"""
data_loader.py

This module loads the raw Excel dataset and
returns user and property data as pandas DataFrames.
"""

import pandas as pd


def load_data(file_path):
    """
    Load user and property data from Excel file.

    Parameters:
        file_path (str): Path to the Excel file

    Returns:
        users_df (pd.DataFrame): User preferences data
        properties_df (pd.DataFrame): Property characteristics data
    """

    # Read Excel file
    excel_file = pd.ExcelFile(file_path)

    # Load sheets
    users_df = excel_file.parse(excel_file.sheet_names[0])
    properties_df = excel_file.parse(excel_file.sheet_names[1])

    return users_df, properties_df
