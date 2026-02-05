"""
main.py

Entry point for the semantic property matching system.
"""

import os
import pandas as pd

from src.data_loader import load_data
from src.matcher import compute_all_matches
from src.config import DATA_PATH, OUTPUT_DIR, TOP_K


def main():
    # Load data
    print("📦 Loading data...")
    users_df, properties_df = load_data(DATA_PATH)

    print("🧠 Computing matches...")
    results_df = compute_all_matches(
        users_df,
        properties_df,
        top_k=TOP_K
    )

    print("💾 Saving results...")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "top_k_recommendations.csv")
    results_df.to_csv(output_path, index=False)

    print("Matching completed successfully!")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
