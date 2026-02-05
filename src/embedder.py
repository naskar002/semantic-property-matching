"""
embedder.py

This module handles sentence embedding generation
using a pre-trained SentenceTransformer model.
"""

from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    Wrapper class for sentence embedding model.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Generate embeddings for a list of texts.

        Parameters:
            texts (list[str]): List of text strings

        Returns:
            numpy.ndarray: Array of embeddings
        """
        return self.model.encode(texts, show_progress_bar=True)
