"""
streamlit_app.py

Interactive Streamlit UI for semantic property matching.
Users can input preferences and property details to get real-time match scores.
"""

import sys
import os

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
from src.text_builder import user_to_text, property_to_text
from src.embedder import TextEmbedder
from src.similarity import compute_similarity
from src.feature_encoder import compute_numerical_similarity
from src.config import EMBEDDING_MODEL_NAME, SEMANTIC_WEIGHT, NUMERICAL_WEIGHT


# Page configuration
st.set_page_config(
    page_title="Property Matcher",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom styling
st.markdown("""
    <style>
    .match-score-high { color: #28a745; font-size: 48px; font-weight: bold; }
    .match-score-medium { color: #ffc107; font-size: 48px; font-weight: bold; }
    .match-score-low { color: #dc3545; font-size: 48px; font-weight: bold; }
    .metric-card { background-color: #f8f9fa; padding: 20px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for embedder
@st.cache_resource
def load_embedder():
    """Load the embedding model (cached for performance)"""
    return TextEmbedder(model_name=EMBEDDING_MODEL_NAME)


def get_score_color(score):
    """Return color class based on match score"""
    if score >= 75:
        return "match-score-high"
    elif score >= 50:
        return "match-score-medium"
    else:
        return "match-score-low"


def get_score_label(score):
    """Return label and emoji based on match score"""
    if score >= 80:
        return "🟢 Excellent Match", "green"
    elif score >= 70:
        return "🟡 Good Match", "blue"
    elif score >= 50:
        return "🟠 Fair Match", "orange"
    else:
        return "🔴 Poor Match", "red"


# Page title
st.title("🏠 Semantic Property Matcher")
st.write("Find your perfect property match using hybrid semantic + numerical similarity!")

# Create two columns for input sections
col1, col2 = st.columns(2)

# ============== USER PREFERENCES SECTION ==============
with col1:
    st.header("👤 User Preferences")
    st.write("Enter what you're looking for in a property:")
    
    user_budget = st.number_input(
        "Budget ($)",
        min_value=10000,
        max_value=10000000,
        value=500000,
        step=10000,
        help="Maximum budget in dollars"
    )
    
    user_bedrooms = st.slider(
        "Number of Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        help="Desired number of bedrooms"
    )
    
    user_bathrooms = st.slider(
        "Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        help="Desired number of bathrooms"
    )
    
    user_description = st.text_area(
        "Your Preferences (Qualitative)",
        value="Modern kitchen, open floor plan, quiet neighborhood, close to schools",
        height=100,
        help="Describe your preferences in natural language (e.g., 'spacious living room, updated appliances, near parks')"
    )


# ============== PROPERTY CHARACTERISTICS SECTION ==============
with col2:
    st.header("🏢 Property Details")
    st.write("Enter property information:")
    
    property_price = st.number_input(
        "Price ($)",
        min_value=10000,
        max_value=10000000,
        value=475000,
        step=10000,
        help="Property price in dollars"
    )
    
    property_bedrooms = st.slider(
        "Number of Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of bedrooms in property",
        key="prop_beds"
    )
    
    property_bathrooms = st.slider(
        "Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        help="Number of bathrooms in property",
        key="prop_baths"
    )
    
    property_living_area = st.number_input(
        "Living Area (sq ft)",
        min_value=500,
        max_value=50000,
        value=2500,
        step=100,
        help="Living space in square feet"
    )
    
    property_description = st.text_area(
        "Property Description",
        value="Newly renovated home with modern kitchen, open floor plan, in excellent school district, peaceful neighborhood",
        height=100,
        help="Describe the property in natural language"
    )


# ============== COMPUTATION SECTION ==============
st.divider()
st.subheader("🔍 Real-Time Match Computation")

# Create a button to compute match score
if st.button("🎯 Compute Match Score", use_container_width=True, type="primary"):
    
    with st.spinner("⏳ Embedding text and computing similarity..."):
        # Load embedder
        embedder = load_embedder()
        
        # Create text representations
        user_row = {
            "Budget": user_budget,
            "Bedrooms": user_bedrooms,
            "Bathrooms": user_bathrooms,
            "Qualitative Description": user_description
        }
        
        property_row = {
            "Price": property_price,
            "Bedrooms": property_bedrooms,
            "Bathrooms": property_bathrooms,
            "Living Area (sq ft)": property_living_area,
            "Qualitative Description": property_description
        }
        
        # Convert to text
        user_series = pd.Series(user_row)
        property_series = pd.Series(property_row)

        user_text = user_to_text(user_series)
        property_text = property_to_text(property_series)
        
        # Encode to embeddings
        user_embedding = embedder.encode([user_text])[0]
        property_embedding = embedder.encode([property_text])[0]
        
        # Compute numerical + hybrid similarity
        numerical_score = compute_numerical_similarity(user_series, property_series)
        match_score = float(compute_similarity(user_embedding, property_embedding, numerical_score=numerical_score))
        
        # Store in session state
        st.session_state.match_score = match_score
        st.session_state.user_text = user_text
        st.session_state.property_text = property_text
        st.session_state.numerical_score = numerical_score


# ============== RESULTS DISPLAY SECTION ==============
if "match_score" in st.session_state:
    match_score = st.session_state.match_score
    
    st.divider()
    st.subheader("✨ Match Results")
    
    # Display match score prominently
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        score_label, score_color = get_score_label(match_score)
        st.markdown(f"<div style='text-align: center;'><h1>{score_label}</h1></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 56px; color: {score_color};'>{match_score:.2f}/100</div>", unsafe_allow_html=True)
    
    # Display score gauge
    st.progress(match_score / 100, text=f"Match Score: {match_score:.2f}%")
    
    # Detailed metrics
    st.subheader("📊 Detailed Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Budget Match", f"${user_budget:,.0f} vs ${property_price:,.0f}", delta="Check Details")
    
    with col2:
        st.metric("Bedrooms", f"{user_bedrooms} needed vs {property_bedrooms} available", delta="Match" if user_bedrooms == property_bedrooms else "Mismatch")
    
    with col3:
        st.metric("Bathrooms", f"{user_bathrooms} needed vs {property_bathrooms} available", delta="Match" if user_bathrooms == property_bathrooms else "Mismatch")
    
    with col4:
        living_area_diff = property_living_area - (user_budget / 500)  # Rough estimate
        st.metric("Living Area", f"{property_living_area:,} sq ft", delta="Spacious" if property_living_area > 2000 else "Compact")
    
    # Show text representations used for matching
    with st.expander("📄 View Text Representations (Used for Matching)"):
        st.subheader("User Preferences (as text):")
        st.info(st.session_state.user_text)
        
        st.subheader("Property Description (as text):")
        st.info(st.session_state.property_text)
    
    # Score interpretation
    with st.expander("📚 Score Interpretation"):
        st.write(f"""
        **Match Score Ranges:**
        - **80-100**: 🟢 **Excellent Match** - Property strongly aligns with preferences
        - **70-79**: 🟡 **Good Match** - Property generally meets requirements
        - **50-69**: 🟠 **Fair Match** - Some aspects align, others may not
        - **0-49**: 🔴 **Poor Match** - Property does not meet preferences

        **What the Score Measures:**
        The match score blends semantic similarity with numerical feature alignment. Semantic similarity is computed from text embeddings, and numerical similarity considers budget, bedrooms, bathrooms, and living area when a user preference is provided. Final score = {SEMANTIC_WEIGHT:.0%} semantic + {NUMERICAL_WEIGHT:.0%} numerical.
        """)


# ============== SIDEBAR INFO ==============
with st.sidebar:
    st.header("ℹ️ About This App")
    st.write(f"""
    **Semantic Property Matcher** uses a hybrid of semantic embeddings and numerical matching to recommend properties.
    
    **How it works:**
    1. Your preferences and property details are converted to natural language
    2. A pre-trained AI model ({EMBEDDING_MODEL_NAME}) converts text to semantic embeddings
    3. Semantic similarity is computed using cosine similarity (0-100 scale)
    4. Numerical similarity is computed from budget, bedrooms, bathrooms, and living area when provided
    5. Final score = {SEMANTIC_WEIGHT:.0%} semantic + {NUMERICAL_WEIGHT:.0%} numerical
    """)
    
    st.divider()
    
    st.subheader("🎯 Tips for Better Matches")
    st.write("""
    - Be specific in your qualitative descriptions
    - Include neighborhood preferences
    - Mention lifestyle priorities
    - Describe must-have vs nice-to-have features
    """)
    
    st.divider()
    
    st.subheader("📚 Technologies")
    st.write("""
    - **Streamlit**: Interactive web UI
    - **Sentence Transformers**: Semantic embeddings
    - **Scikit-learn**: Cosine similarity computation
    """)


# Footer
st.divider()
st.caption("🏠 Semantic Property Matching System | Built with Streamlit & AI | February 2026")
