"""
matcher.py

This module computes match scores between all
user-property pairs and ranks properties per user.
"""

import pandas as pd

from src.text_builder import user_to_text, property_to_text
from src.embedder import TextEmbedder
from src.similarity import compute_similarity
from src.feature_encoder import compute_numerical_similarity
from src.config import USER_ID_COL, PROPERTY_ID_COL


def compute_all_matches(users_df, properties_df, top_k=5):
    """
    Compute match scores for all user-property pairs.

    Parameters:
        users_df (pd.DataFrame): User preferences dataframe
        properties_df (pd.DataFrame): Property characteristics dataframe
        top_k (int): Number of top properties to keep per user

    Returns:
        pd.DataFrame: Ranked match scores
    """

    embedder = TextEmbedder()

    # ---- Convert users to text ----
    user_texts = users_df.apply(user_to_text, axis=1).tolist()
    user_embeddings = embedder.encode(user_texts)

    # ---- Convert properties to text ----
    property_texts = properties_df.apply(property_to_text, axis=1).tolist()
    property_embeddings = embedder.encode(property_texts)

    results = []

    # ---- Compute similarity for each user-property pair ----
    for user_idx, user_embedding in enumerate(user_embeddings):
        user_row = users_df.iloc[user_idx]
        user_id = user_row[USER_ID_COL]

        for prop_idx, prop_embedding in enumerate(property_embeddings):
            property_row = properties_df.iloc[prop_idx]
            property_id = property_row[PROPERTY_ID_COL]

            numerical_score = compute_numerical_similarity(user_row, property_row)
            score = compute_similarity(
                user_embedding,
                prop_embedding,
                numerical_score=numerical_score
            )

            results.append({
                "user_id": user_id,
                "property_id": property_id,
                "match_score": score
            })

    results_df = pd.DataFrame(results)

    # ---- Rank properties per user ----
    results_df = (
        results_df
        .sort_values(["user_id", "match_score"], ascending=[True, False])
        .groupby("user_id")
        .head(top_k)
        .reset_index(drop=True)
    )

    return results_df
