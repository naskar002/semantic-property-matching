"""
similarity.py

This module computes similarity scores between
user and property embeddings.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(user_embedding, property_embedding):
    """
    Compute cosine similarity between two embeddings
    and return a normalized match score.

    Parameters:
        user_embedding (np.ndarray): User embedding vector
        property_embedding (np.ndarray): Property embedding vector

    Returns:
        float: Match score between 0 and 100
    """

    # Ensure correct shape
    user_embedding = np.array(user_embedding).reshape(1, -1)
    property_embedding = np.array(property_embedding).reshape(1, -1)

    # Cosine similarity
    similarity_score = cosine_similarity(
        user_embedding,
        property_embedding
    )[0][0]

    # Normalize to 0–100
    match_score = round(similarity_score * 100, 2)

    return match_score
