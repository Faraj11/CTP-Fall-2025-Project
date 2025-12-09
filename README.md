# BetterBites 🍽️

A Flask web application for discovering restaurants in New York City with comprehensive analytics and AI-powered search.

[Visit the deployed app!](https://huggingface.co/spaces/kobrakai11/BetterBites)

## Features

### **Smart Restaurant Search**

**Image Search:**
- Upload food images or take photos with a camera
- AI-powered captioning using BLIP model
- Caption interpretation for cuisine + food type matching
- Weighted matching: cuisine (60%), food items (25%), caption-cuisine (10%), name/ratings (5%)

**Text Search:**
- Intelligent matching with weighted algorithm (cuisine 35%, location 30%, name 25%, rating/reviews 10%)
- Supports over 20 cuisine types, including halal, kosher, and vegetarian
- Borough and neighborhood matching
- Match score system with detailed restaurant profiles
  
### **Dashboards**

**NYC Restaurants Dashboard:**
- Analytics for 1,585 NYC restaurants
- Geographic heatmap with lat/lon coordinates
- Cuisine distribution graph
- Rating analysis with custom intervals
- Price category distribution
- Ratings comparison by food, service, and ambiance
  
**USA Yelp Analysis Dashboard:**
- Sentiment distribution visualizations
- Word frequency analysis charts
- Word clouds for positive + negative sentiment
- Theme analysis from reviews

## Installation

### Prerequisites
- Python 3.7+

### Setup

1. **Clone and navigate:**
   ```bash
   git clone <repository-url>
   cd CTP-Fall-2025-Project-main
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Access:** Open `http://127.0.0.1:5000`

### Data
- Processed NYC restaurant data (`nyc_restaurants_merged.csv`) is included
- Yelp Academic Dataset JSON files are optional - the app works without them
- For USA Dashboard functionality, download Yelp Academic Dataset from https://www.yelp.com/dataset

## Project Structure

```
.
├── app.py                      # Flask application with API endpoints
├── merge_restaurants.py        # Data merging and normalization
├── yelp_chart_generator.py    # Yelp analysis chart generator
├── sentiment_analyzer.py       # Sentiment analysis module
├── word_analyzer.py            # Word frequency analysis
├── theme_extractor.py          # Theme extraction
├── fast_cache.py              # Cache generation utility
├── image_captioner.py         # BLIP image captioning model
├── templates/
│   ├── dashboard.html         # Dashboard with analytics charts
│   ├── index.html             # Text search interface
│   └── image_search.html      # Image search with AI captioning + camera
├── cache/                     # Generated cache files
├── nyc_restaurants_merged.csv  # Restaurant dataset
├── yelp_academic_dataset_*.json # Yelp datasets
├── yelp_analysis_*.ipynb      # Reference notebooks
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration for deployment
└── README.md                  # This file
```

## Deployment

### Hugging Face Spaces

1. Create Space: https://huggingface.co/spaces → New Space (Docker SDK)
2. Set root directory: `CTP-Fall-2025-Project-main`
3. Build: First build takes ~10-15 minutes (downloads BLIP model ~990MB)
4. Access: `https://yourusername-betterbites.hf.space`

**Settings:**
- SDK: **Docker** (not Gradio)
- Root directory: `CTP-Fall-2025-Project-main`

**Optional - Enable USA Dashboard:**
- Upload Yelp JSON files to a Hugging Face Dataset
- Set environment variable: `YELP_DATASET_NAME=yourusername/yelp-academic-dataset`
- Restart Space (files download automatically)

The app works without Yelp files - only the USA Dashboard tab requires them.

## API Endpoints

**Dashboard:**
- `GET /` - Dashboard home page
- `GET /api/dashboard/stats` - Restaurant statistics
- `GET /api/dashboard/geographic` - Geographic data for heatmap
- `GET /api/dashboard/ratings-comparison` - Rating analysis

**Search:**
- `GET /search` - Text search page
- `GET /api/search?query=<query>` - Restaurant text search
- `GET /image-search` - Image search page
- `POST /api/image-search` - Restaurant image search

**Response Example:**
```json
{
  "best_match": {
    "name": "Restaurant Name",
    "overall_rating": 4.5,
    "reviews": 123,
    "price_category": "$$",
    "borough": "Manhattan",
    "neighborhood": "Chelsea",
    "cuisine": "Italian",
    "match_score_percentile": 95.5
  },
  "all_matches": [...]
}
```

## Technical Stack

- **Backend:** Flask 2.3+ (Python 3.7+)
- **Frontend:** HTML5, CSS3, JavaScript (ES6+)
- **Visualization:** Plotly.js
- **Data Processing:** Pandas
- **AI/ML:** Transformers (BLIP), PyTorch
- **NLP:** NLTK, TextBlob, VADER Sentiment

## Requirements

```
flask>=2.3.0
pandas>=2.0.0
plotly>=5.17.0
nltk>=3.8
textblob>=0.17.1
vaderSentiment>=3.3.2
wordcloud>=1.9.2
numpy>=1.24.0
transformers>=4.30.0
torch>=2.0.0
Pillow>=10.0.0
datasets>=2.14.0
huggingface-hub>=0.16.0
```

**Note:** BLIP model (~990MB) downloads automatically on the first image search use and is cached locally.
