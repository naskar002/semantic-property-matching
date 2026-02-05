"""
feature_encoder.py

Compute numerical similarity scores between user preferences and property features.
"""

import math

from src.config import (
    BUDGET_TOLERANCE,
    BEDROOM_FLEX,
    BATHROOM_FLEX,
    LIVING_AREA_TOLERANCE,
)


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    return val


def _get_value(row, key):
    if row is None:
        return None
    if hasattr(row, "get"):
        value = row.get(key, None)
    else:
        try:
            value = row[key]
        except Exception:
            value = None
    return _safe_float(value)


def _tolerance_score(target, actual, tolerance):
    if target is None or actual is None:
        return None
    if tolerance is None or tolerance <= 0:
        return 1.0 if target == actual else 0.0
    if target == 0:
        return 1.0 if actual == 0 else 0.0

    diff_ratio = abs(actual - target) / target
    if diff_ratio >= tolerance:
        return 0.0
    return max(0.0, 1.0 - (diff_ratio / tolerance))


def _flex_match_score(preferred, actual, flexibility, near_score=0.7):
    if preferred is None or actual is None:
        return None
    diff = abs(actual - preferred)
    if diff == 0:
        return 1.0
    if flexibility is not None and diff <= flexibility:
        return float(near_score)
    return 0.0


def compute_numerical_similarity(
    user_row,
    property_row,
    budget_tolerance=BUDGET_TOLERANCE,
    bedroom_flex=BEDROOM_FLEX,
    bathroom_flex=BATHROOM_FLEX,
    living_area_tolerance=LIVING_AREA_TOLERANCE,
):
    """
    Compute a numerical similarity score (0-100) for structured features.

    Features:
        - Budget vs Price (within tolerance)
        - Bedrooms (exact or flexible)
        - Bathrooms (exact or flexible)
        - Living area (within tolerance, if user preference exists)
    """

    scores = []

    # Budget vs Price
    budget = _get_value(user_row, "Budget")
    price = _get_value(property_row, "Price")
    score = _tolerance_score(budget, price, budget_tolerance)
    if score is not None:
        scores.append(score)

    # Bedrooms
    user_bedrooms = _get_value(user_row, "Bedrooms")
    property_bedrooms = _get_value(property_row, "Bedrooms")
    score = _flex_match_score(user_bedrooms, property_bedrooms, bedroom_flex)
    if score is not None:
        scores.append(score)

    # Bathrooms
    user_bathrooms = _get_value(user_row, "Bathrooms")
    property_bathrooms = _get_value(property_row, "Bathrooms")
    score = _flex_match_score(user_bathrooms, property_bathrooms, bathroom_flex)
    if score is not None:
        scores.append(score)

    # Living area (only if user preference exists)
    user_living_area = _get_value(user_row, "Living Area (sq ft)")
    property_living_area = _get_value(property_row, "Living Area (sq ft)")
    score = _tolerance_score(user_living_area, property_living_area, living_area_tolerance)
    if score is not None:
        scores.append(score)

    if not scores:
        return None

    return round((sum(scores) / len(scores)) * 100, 2)
