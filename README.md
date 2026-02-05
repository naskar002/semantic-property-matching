# Semantic Property Matching System

Semantic Property Matching is a hybrid recommendation engine that matches users to properties using both semantic text similarity and structured numeric features.

## Overview
The system converts user preferences and property descriptions into text, embeds them with a transformer model, and combines that semantic score with a numerical feature score (budget, bedrooms, bathrooms, and optional living area). The output is a ranked Top-K list of property recommendations per user.

## Key Features
- Hybrid scoring: semantic similarity + numerical alignment
- Pre-trained sentence embeddings with `all-mpnet-base-v2`
- Flexible numeric matching with tolerances
- Batch recommendation generation to CSV
- Interactive Streamlit app for real-time matching
- Visualization utilities for score analysis

## How It Works
1. Load user and property data from Excel.
2. Convert structured fields into natural language text.
3. Generate embeddings using Sentence Transformers.
4. Compute semantic similarity with cosine similarity.
5. Compute numerical similarity from budget, bedrooms, bathrooms, and living area (if provided by user).
6. Combine scores using weighted hybrid scoring.
7. Rank Top-K properties for each user and export results.

## Data Schema
User Data sheet
- `User ID` (int)
- `Budget` (float)
- `Bedrooms` (int)
- `Bathrooms` (int)
- `Qualitative Description` (str)

Property Data sheet
- `Property ID` (int)
- `Price` (float)
- `Bedrooms` (int)
- `Bathrooms` (int)
- `Living Area (sq ft)` (float)
- `Qualitative Description` (str)

## Hybrid Scoring
Semantic score is computed from text embeddings. Numerical score is computed from structured features:
- Budget vs Price within tolerance
- Bedrooms exact or flexible match
- Bathrooms exact or flexible match
- Living area within tolerance (only if user preference exists)

Final score:
```
final_score = semantic_score * SEMANTIC_WEIGHT + numerical_score * NUMERICAL_WEIGHT
```

Default weights (in `src/config.py`):
- `SEMANTIC_WEIGHT = 0.7`
- `NUMERICAL_WEIGHT = 0.3`

## Project Structure
```
semantic-property-matching/
  app/
    streamlit_app.py
  data/
    raw/Case_Study_2_Data.xlsx
  outputs/
    top_k_recommendations.csv
    figures/
  src/
    config.py
    data_loader.py
    embedder.py
    feature_encoder.py
    matcher.py
    similarity.py
    text_builder.py
    visualize.py
  main.py
  requirements.txt
  README.md
```

## Setup
Prerequisites
- Python 3.10 or 3.11

Install dependencies
```
pip install -r requirements.txt
```

## Run the Batch Pipeline
```
python main.py
```
Output file:
- `outputs/top_k_recommendations.csv`

## Run the Streamlit App
```
streamlit run app/streamlit_app.py
```

## Visualizations
Generate plots from the output CSV:
```
python src/visualize.py
```
Saved figures:
- `outputs/figures/user_property_heatmap.png`
- `outputs/figures/score_distribution.png`
- `outputs/figures/user_average_scores.png`
- `outputs/figures/property_average_scores.png`

## Configuration
Edit `src/config.py` to change:
- `EMBEDDING_MODEL_NAME`
- `TOP_K`
- `SEMANTIC_WEIGHT`, `NUMERICAL_WEIGHT`
- `BUDGET_TOLERANCE`, `BEDROOM_FLEX`, `BATHROOM_FLEX`, `LIVING_AREA_TOLERANCE`

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Create a new app in Streamlit Cloud.
3. Set app file to `app/streamlit_app.py`.
4. Choose Python 3.10 or 3.11.
5. Deploy.

## Troubleshooting
Streamlit Cloud error about openpyxl
- If you see a dependency error related to `openpyxl>=3.6.0`, pin it to a real version in `requirements.txt`:
```
openpyxl==3.1.5
```

## License
See `LICENSE`.
