"""
text_builder.py

This module converts structured user and property data
into clean natural-language text representations suitable
for sentence embeddings.
"""


def user_to_text(user_row):
    """
    Convert a single user row into text.

    Parameters:
        user_row (pd.Series): One row from user dataframe

    Returns:
        str: Text representation of user preferences
    """

    budget = user_row["Budget"]
    bedrooms = user_row["Bedrooms"]
    bathrooms = user_row["Bathrooms"]
    description = user_row["Qualitative Description"]

    user_text = (
        f"User is looking for a home with a budget of {budget} dollars, "
        f"{bedrooms} bedrooms and {bathrooms} bathrooms. "
        f"Preferences: {description}"
    )

    return user_text


def property_to_text(property_row):
    """
    Convert a single property row into text.

    Parameters:
        property_row (pd.Series): One row from property dataframe

    Returns:
        str: Text representation of property characteristics
    """

    price = property_row["Price"]
    bedrooms = property_row["Bedrooms"]
    bathrooms = property_row["Bathrooms"]
    living_area = property_row["Living Area (sq ft)"]
    description = property_row["Qualitative Description"]

    property_text = (
        f"This property is priced at {price} dollars, "
        f"has {bedrooms} bedrooms and {bathrooms} bathrooms, "
        f"with a living area of {living_area} square feet. "
        f"Property description: {description}"
    )

    return property_text
