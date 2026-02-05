"""
similarity.py

This module computes similarity scores between
user and property embeddings.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import SEMANTIC_WEIGHT, NUMERICAL_WEIGHT


def compute_similarity(
    user_embedding,
    property_embedding,
    numerical_score=None,
    semantic_weight=SEMANTIC_WEIGHT,
    numerical_weight=NUMERICAL_WEIGHT,
):
    """
    Compute semantic similarity between two embeddings and optionally combine
    with a numerical similarity score using weighted hybrid scoring.

    Parameters:
        user_embedding (np.ndarray): User embedding vector
        property_embedding (np.ndarray): Property embedding vector
        numerical_score (float | None): Numerical similarity score (0-100)
        semantic_weight (float): Weight for semantic similarity
        numerical_weight (float): Weight for numerical similarity

    Returns:
        float: Match score between 0 and 100
    """

    # Ensure correct shape
    user_embedding = np.array(user_embedding).reshape(1, -1)
    property_embedding = np.array(property_embedding).reshape(1, -1)

    # Cosine similarity
    semantic_similarity = cosine_similarity(user_embedding, property_embedding)[0][0]

    # Normalize to 0-100
    semantic_score = round(semantic_similarity * 100, 2)

    if numerical_score is None:
        return semantic_score

    total_weight = semantic_weight + numerical_weight
    if total_weight <= 0:
        return semantic_score

    hybrid_score = (semantic_score * semantic_weight + numerical_score * numerical_weight) / total_weight
    return round(hybrid_score, 2)
