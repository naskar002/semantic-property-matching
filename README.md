# Semantic Property Matching System

## Project Title
**Semantic Property Matching: AI-Powered Real Estate Recommendation Engine**

---

## Overview
The Semantic Property Matching System is an intelligent recommendation engine that matches users with suitable properties based on their preferences. Using advanced Natural Language Processing (NLP) and semantic embeddings, the system understands both user requirements and property characteristics in context, providing personalized property recommendations ranked by match quality.

---

## Problem Statement
Traditional property matching systems rely on simple keyword matching or basic filtering rules, which often fail to capture the nuanced preferences and contextual requirements of users. The challenge is to:

1. **Understand Complex User Preferences**: Users express their needs qualitatively (e.g., "quiet neighborhood," "close to schools"), not just numerically.
2. **Capture Property Characteristics**: Properties have both numerical features (price, bedrooms) and qualitative descriptions that matter to users.
3. **Find Semantic Alignment**: Match users with properties that align semantically with their preferences, not just numerical criteria.
4. **Rank Recommendations**: Provide top-K relevant properties for each user, ranked by similarity to their preferences.

This system solves these challenges using semantic embeddings to understand meaning and context.

---

## Dataset Description
The project uses an Excel dataset (`data/raw/Case_Study_2_Data.xlsx`) containing two sheets:

### Sheet 1: User Data
| Column | Type | Description |
|--------|------|-------------|
| User ID | Integer | Unique identifier for each user |
| Budget | Float | Maximum budget in dollars |
| Bedrooms | Integer | Number of bedrooms desired |
| Bathrooms | Integer | Number of bathrooms desired |
| Qualitative Description | String | User preferences in natural language |

### Sheet 2: Property Data
| Column | Type | Description |
|--------|------|-------------|
| Property ID | Integer | Unique identifier for each property |
| Price | Float | Property price in dollars |
| Bedrooms | Integer | Number of bedrooms |
| Bathrooms | Integer | Number of bathrooms |
| Living Area (sq ft) | Float | Living space in square feet |
| Qualitative Description | String | Property description in natural language |

---

## Approach / Methodology

### High-Level Pipeline
```
Raw Data (Excel)
    ↓
Text Representation (Combine numerical + qualitative features)
    ↓
Semantic Embeddings (Convert text to dense vectors)
    ↓
Similarity Computation (Compare user ↔ property embeddings)
    ↓
Top-K Ranking (Select best matches per user)
    ↓
Output (CSV with recommendations)
```

### Key Steps

1. **Data Loading**: Load user and property data from Excel using Pandas.
2. **Text Representation**: Convert structured data into coherent natural language sentences.
3. **Encoding**: Generate semantic embeddings from text using a pre-trained transformer model.
4. **Similarity Scoring**: Compute cosine similarity between user and property embeddings.
5. **Ranking & Filtering**: Rank properties for each user and keep top-K matches.
6. **Export Results**: Save matched pairs with scores to CSV.

---

## Text Representation

### User Text Format
Each user is converted to a sentence combining their preferences:
```
"User is looking for a home with a budget of $500,000 dollars, 3 bedrooms and 2 bathrooms. 
Preferences: Modern kitchen, open floor plan, near good schools, quiet neighborhood"
```

**Function**: `user_to_text()` in `src/text_builder.py`

### Property Text Format
Each property is converted to a descriptive sentence:
```
"This property is priced at $475,000 dollars, has 3 bedrooms and 2 bathrooms, 
with a living area of 2,500 square feet. Property description: Newly renovated home 
with modern kitchen, open floor plan, in excellent school district"
```

**Function**: `property_to_text()` in `src/text_builder.py`

### Rationale
Converting structured + qualitative data into natural language allows the semantic embedding model to understand context, relationships, and implicit meaning that numerical features alone cannot capture.

---

## Embedding Model Used

### Model: `all-MiniLM-L6-v2`
- **Architecture**: Lightweight SentenceTransformer based on MiniLM
- **Dimensions**: 384-dimensional embeddings
- **Training Data**: Trained on diverse English text for semantic similarity
- **Advantages**:
  - Fast inference with minimal computational overhead
  - High-quality semantic representations
  - Pre-trained on general English, suitable for property/preference text
  - Optimized for sentence-level similarity tasks

### Library: `sentence-transformers`
- PyTorch-based library for state-of-the-art sentence embeddings
- Provides `SentenceTransformer` class that handles tokenization and encoding
- Returns NumPy arrays of embeddings ready for similarity computation

**Class**: `TextEmbedder` in `src/embedder.py`
- `__init__(model_name)`: Loads the pre-trained model
- `encode(texts)`: Converts list of text strings to embeddings (batch processing)

---

## Similarity Metric

### Cosine Similarity
The system computes **cosine similarity** between user and property embeddings:

$$\text{similarity} = \frac{\mathbf{u} \cdot \mathbf{p}}{|\mathbf{u}| \cdot |\mathbf{p}|}$$

Where:
- $\mathbf{u}$ = user embedding vector
- $\mathbf{p}$ = property embedding vector

### Score Normalization
- Raw cosine similarity ranges: **[−1, 1]** (but typically [0, 1] for normalized embeddings)
- Normalized match score: **[0, 100]** (multiply by 100 and round to 2 decimals)

### Why Cosine Similarity?
- **Semantically meaningful**: Measures angle between vectors, not distance
- **Scale-invariant**: Works with any embedding magnitude
- **Interpretable**: Higher score = more similar meaning
- **Efficient**: Fast computation with scikit-learn

**Function**: `compute_similarity()` in `src/similarity.py`

---

## Recommendation Logic (Top-K)

### Algorithm
1. Compute match scores for **all user–property pairs** (O(users × properties))
2. For each user, **sort properties** by match score in descending order
3. Keep top-K properties per user (K=5 by default, configurable)
4. Return ranked results as a DataFrame

### DataFrame Columns
| Column | Type | Description |
|--------|------|-------------|
| user_id | Integer | User ID |
| property_id | Integer | Property ID |
| match_score | Float | Similarity score [0–100] |

### Configuration
- **TOP_K**: Default = 5 (change in `src/config.py`)
- **User ID Column**: "User ID"
- **Property ID Column**: "Property ID"

**Function**: `compute_all_matches()` in `src/matcher.py`

---

## Project Folder Structure

```
semantic-property-matching/
├── main.py                      # Entry point for the pipeline
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── LICENSE                      # Project license
│
├── src/                         # Core source code
│   ├── config.py               # Central configuration (paths, models, TOP_K)
│   ├── data_loader.py          # Load data from Excel
│   ├── text_builder.py         # Convert data to natural language
│   ├── embedder.py             # Sentence embedding generation
│   ├── similarity.py           # Cosine similarity computation
│   └── matcher.py              # Main matching logic & top-K ranking
│
├── data/                        # Data directory
│   ├── raw/                    # Raw Excel files
│   │   └── Case_Study_2_Data.xlsx
│   └── processed/              # (For future processed data)
│
├── notebooks/                   # Jupyter notebooks for exploration
│
├── outputs/                     # Generated output files
│   ├── figures/                # Visualization outputs
│   └── top_k_recommendations.csv # Final recommendations
│
└── app/                         # (For future web app integration)
```

---

## How to Run the Project

### Prerequisites
- **Python 3.8+**
- **Virtual Environment** (recommended)

### Step 1: Clone/Setup Project
```bash
cd D:\Property_similarity\semantic-property-matching
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install pandas sentence-transformers scikit-learn openpyxl
```

Or from `requirements.txt` (when populated):
```powershell
pip install -r requirements.txt
```

### Step 4: Ensure Data File Exists
Place `Case_Study_2_Data.xlsx` in `data/raw/`:
```powershell
Test-Path .\data\raw\Case_Study_2_Data.xlsx
```

### Step 5: Run the Pipeline
```powershell
python main.py
```

### Expected Output
```
📦 Loading data...
🧠 Computing matches...
💾 Saving results...
Matching completed successfully!
Results saved to: outputs/top_k_recommendations.csv
```

### Step 6: Check Results
Open `outputs/top_k_recommendations.csv` to view the recommendations:
```powershell
# View in PowerShell
Import-Csv .\outputs\top_k_recommendations.csv | Format-Table
```

---

## Output Description

### File: `outputs/top_k_recommendations.csv`

**Format**: CSV with the following columns:

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| user_id | Integer | 1 | User identifier |
| property_id | Integer | 15 | Property identifier |
| match_score | Float | 87.45 | Similarity score (0–100) |

### Interpretation
- **user_id=1, property_id=15, match_score=87.45** means:
  - User 1 is matched with Property 15
  - The match quality is 87.45/100
  - Results are sorted by user_id, then by match_score (descending)
  - Top 5 properties per user are included (K=5)

### Example Output
```
user_id,property_id,match_score
1,15,87.45
1,23,84.12
1,8,79.63
1,42,76.88
1,19,73.21
2,12,92.14
2,30,88.76
...
```

---

## Technologies Used

### Core Libraries
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Programming language |
| **Pandas** | Latest | Data manipulation & CSV I/O |
| **sentence-transformers** | Latest | Semantic embeddings (SentenceTransformer) |
| **scikit-learn** | Latest | Cosine similarity computation |
| **NumPy** | Latest | Numerical operations for embeddings |
| **openpyxl** | Latest | Excel file parsing (via Pandas) |

### Model
- **Pre-trained Embedding Model**: `all-MiniLM-L6-v2` (384-dim, lightweight)
- **Framework**: Hugging Face Transformers (used by sentence-transformers)

### Development Tools (Optional)
- **Jupyter Notebook**: For exploratory analysis
- **Git**: Version control
- **VS Code / IDE**: Code editing

---

## Summary / Conclusion

### What the System Achieves
The Semantic Property Matching System demonstrates how **semantic understanding** transforms real estate recommendations:

1. **Beyond Numeric Filters**: Instead of simple budget/bedroom filters, the system understands qualitative preferences like "quiet neighborhood" or "modern kitchen."

2. **Contextual Matching**: Embeddings capture semantic relationships—a property described as "newly renovated with modern amenities" will match users seeking "updated homes with contemporary design."

3. **Personalized Recommendations**: Each user receives a ranked list of top-5 properties tailored to their unique profile and preferences.

4. **Scalable & Efficient**: The pipeline processes users and properties in batches, making it suitable for real estate platforms with thousands of users and listings.

### Key Insights
- **Text Representation**: Converting mixed numeric and qualitative data into natural language is crucial for semantic models.
- **Pre-trained Models**: Using a general-purpose embedding model (`all-MiniLM-L6-v2`) is effective and cost-efficient.
- **Cosine Similarity**: A simple, interpretable metric that works well for semantic matching.
- **Top-K Filtering**: Returning top-5 matches provides a manageable, actionable recommendation list.

### Future Enhancements
- **User Feedback Loop**: Incorporate implicit signals (clicks, time spent) to refine recommendations.
- **Fine-tuning**: Fine-tune the embedding model on real estate data for domain-specific performance.
- **Web Dashboard**: Build a REST API and web UI for real-time recommendations.
- **Advanced Features**: Add filters (price range, location), neighborhood analytics, or mortgage calculators.
- **A/B Testing**: Compare semantic matching with other approaches (collaborative filtering, hybrid methods).
- **Visualization**: Create heatmaps, similarity distributions, and recommendation explanations.

---

## License
See [LICENSE](LICENSE) file for details.

---

## Questions?
For issues or questions, please refer to the code comments in `src/` modules or review the inline documentation in each file.

**Happy matching! 🏠✨**
