# BetterBites 🍽️

A modern Flask web application for discovering restaurants in New York City. BetterBites provides comprehensive restaurant analytics through an interactive dashboard and intelligent search functionality.

## Features

### 📊 **Interactive Dashboard**
- **Restaurant Analytics**: Comprehensive statistics for 1,585+ NYC restaurants
- **Geographic Heatmap**: Restaurant density visualization across NYC with lat/lon coordinates
- **Cuisine Distribution**: Top 15 cuisines with consolidated categories (bar chart)
- **Rating Analysis**: Rating distribution with custom intervals and detailed breakdowns
- **Price Category Analysis**: Price distribution with detailed ranges (pie chart)
- **Ratings Comparison**: Average Food, Service, and Ambiance ratings by rating interval

### 🔍 **Smart Restaurant Search**
- **Text Search**: Intelligent matching algorithm with 25% weight on name matching
- **Image Search**: Upload food images or take photos with your camera to find matching restaurants using AI-powered image captioning
- **Camera Integration**: Real-time camera capture for instant food photo analysis
- **Comprehensive Cuisine Support**: Includes halal, kosher, vegetarian, and 20+ cuisine types
- **Location-Based Search**: Borough and neighborhood matching
- **Match Score System**: Displays match percentages for search relevance
- **Detailed Restaurant Profiles**: 9 key data points per restaurant

### 🎨 **Modern UI/UX**
- **Dark Theme**: Beautiful gradient dark theme with smooth animations
- **Responsive Design**: Works seamlessly across desktop and mobile
- **Interactive Charts**: Powered by Plotly.js for rich data visualization
- **Intuitive Navigation**: Clean interface with Home (Dashboard), Image Search, and Text Search tabs

### 📈 **USA Yelp Analysis Dashboard**
- **Sentiment Distribution**: Visual analysis of review sentiment (positive, negative, neutral)
- **Word Frequency Analysis**: Top 10 words from positive and negative reviews
- **Word Clouds**: 50 unique words per sentiment with color intensity indicating sentiment strength
- **Theme Analysis**: Restaurant themes mentioned in reviews (service, food quality, ambiance, etc.)

## Installation

### Prerequisites
- Python 3.7 or higher

### Setup

1. **Clone the repository:**
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

4. **Access the app:**
   
   Open your browser and navigate to: `http://127.0.0.1:5000`

### Data Included
This repository includes the processed NYC restaurant data (`nyc_restaurants_merged.csv`). 

**Note**: The original Yelp Academic Dataset JSON files (`yelp_academic_dataset_review.json` ~5.3GB and `yelp_academic_dataset_business.json` ~118MB) are excluded from this repository due to GitHub's 100MB file size limit. These files are required for the USA Yelp Analysis Dashboard. You can obtain them from the [Yelp Open Dataset](https://www.yelp.com/dataset) if you want to use the USA analysis features. Place them in the project root directory.

## Usage

### Dashboard (Home Page)
- View comprehensive restaurant analytics
- Explore geographic distribution of restaurants
- Analyze cuisine and price trends
- Compare ratings across different categories

### Search Functionality

#### Text Search
1. Navigate to the Text Search tab
2. Enter queries like:
   - "halal food" 
   - "pizza in manhattan"
   - "sushi restaurants"
   - Restaurant names (e.g., "Joe's Pizza")
3. View match scores and detailed restaurant information
4. Click "View All Matches" to see complete results

#### Image Search
1. Navigate to the Image Search tab
2. Choose between "Upload Image" or "Take Photo" mode
3. **Upload Mode**: Upload an image of food, dishes, or restaurant scenes
4. **Camera Mode**: Take a photo directly with your device camera (camera starts automatically)
5. The AI will generate a detailed caption describing the image
6. The system interprets the caption to determine the best cuisine/food match
7. Restaurants matching the interpreted cuisine will be displayed
8. Click on any restaurant from "View All Matches" to see details

## Project Structure

```
.
├── app.py                      # Flask application with API endpoints
├── start_flask.py              # Flask startup script with error handling
├── merge_restaurants.py        # Data merging and normalization script
├── yelp_chart_generator.py    # Yelp analysis chart generator
├── sentiment_analyzer.py       # Sentiment analysis module
├── word_analyzer.py            # Word frequency analysis module
├── theme_extractor.py          # Theme extraction module
├── fast_cache.py              # Cache generation utility
├── templates/
│   ├── dashboard.html         # Dashboard with analytics charts
│   ├── index.html             # Text search interface
│   └── image_search.html      # Image search with AI captioning + camera
├── image_captioner.py         # BLIP image captioning model
├── cache/                     # Cache directory (generated files excluded)
│   └── .gitkeep              # Keeps cache directory in git
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── nyc_restaurants_merged.csv  # Restaurant dataset (included)
├── yelp_analysis_streamlined.ipynb # Reference notebook for Yelp analysis
└── yelp_academic_dataset_*.json # Yelp datasets (if included)
```

## API Endpoints

### Dashboard APIs
- `GET /` - Dashboard home page
- `GET /api/dashboard/stats` - Restaurant statistics and distributions
- `GET /api/dashboard/geographic` - Geographic data for heatmap
- `GET /api/dashboard/ratings-comparison` - Detailed rating analysis

### Search APIs
- `GET /search` - Text search page
- `GET /api/search?query=<search_query>` - Restaurant text search with match scoring
- `GET /image-search` - Image search page
- `POST /api/image-search` - Restaurant image search with AI captioning

**Search Response Example:**
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
    "dining_style": "Casual Dining",
    "food": 4.5,
    "service": 4.3,
    "ambiance": 4.2,
    "dietary_accommodations": "Halal",
    "match_score_percentile": 95.5
  },
  "all_matches": [...]
}
```

## Matching Algorithm

### Text Search Algorithm
The text search uses an optimized matching algorithm:

- **Cuisine Matching (35% weight)**: Prioritizes cuisine matches with extensive keyword support
- **Location Matching (30% weight)**: Borough and neighborhood matching
- **Name Matching (25% weight)**: Restaurant name matching with word boundary detection
- **Rating Boost (7% weight)**: Quality indicator
- **Review Count Boost (3% weight)**: Popularity indicator

### Image Search Algorithm
The image search uses a sophisticated caption interpretation system:

1. **Caption Generation**: BLIP model generates detailed, accurate captions from food images
2. **Caption Interpretation**: Full caption analyzed to determine primary cuisine/food type with confidence scoring
3. **Restaurant Matching**:
   - **Primary Cuisine Match (60% weight)**: Prioritizes restaurants matching interpreted cuisine
   - **Food Item Match (25% weight)**: Matches specific food items found in caption
   - **Caption-Cuisine Match (10% weight)**: General caption-cuisine matching
   - **Restaurant Name (3% weight)**: Name matching
   - **Rating/Reviews (2% weight)**: Quality indicators

## Data Features

### Comprehensive Coverage
- **1,585+ restaurants** across all NYC boroughs
- **92.4% price data coverage** with intelligent normalization
- **Geographic coordinates** for mapping and density analysis
- **Multi-source data integration** with address lookup

### Data Normalization
- **Price Categories**: Consistent ranges (Under $15, $15-$30, $30-$50, Over $50)
- **Cuisine Classification**: 20+ cuisine types with inference from restaurant names
- **Borough Extraction**: From locality, address, and ZIP codes
- **Rating Intervals**: Custom intervals for meaningful insights

### Cuisine Support
Supports comprehensive cuisine matching including:
- Traditional cuisines (Italian, Chinese, Japanese, etc.)
- Dietary restrictions (Halal, Kosher, Vegetarian, Vegan)
- Food types (Pizza, Bakery, Coffee, Deli, Steakhouse)
- Regional specialties (Caribbean, African, Fusion)

## Technical Stack

- **Backend**: Flask 2.3+ with Python 3.7+
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Data Visualization**: Plotly.js for interactive charts
- **Data Processing**: Pandas for data manipulation and analysis
- **Styling**: Modern CSS with dark theme and responsive design

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
```

**Note**: The image search feature uses Hugging Face's BLIP (Bootstrapping Language-Image Pre-training) model for accurate image captioning. On first use, the model will be downloaded automatically (approximately 990MB). Subsequent uses will be faster as the model is cached locally.

## Performance Features

- **Efficient Data Loading**: CSV data loaded once at startup
- **Optimized Search**: Intelligent filtering reduces search space
- **Responsive Charts**: Interactive visualizations with proper scaling
- **Background Processing**: Non-blocking data operations

## Notes

- This is a complete, self-contained project with all data included
- The app automatically handles missing data with intelligent defaults  
- Search algorithm optimized for discovery rather than exact matching
- All charts are responsive and work across different screen sizes
- Original Yelp dataset files are excluded due to size, but processed data is included

## Recent Updates

### Image Search Enhancements
- **Camera Integration**: Added ability to take photos directly from device camera
- **Auto-start Camera**: Camera automatically starts when camera mode is selected
- **Improved Caption Generation**: Upgraded from ViT-GPT2 to BLIP model for more accurate, detailed captions
- **Caption Interpretation**: New system that analyzes full captions to determine primary cuisine/food type
- **Better Matching**: Image search now prioritizes restaurants based on interpreted cuisine (60% weight)
- **Typo Fixes**: Added post-processing to fix common typos and incomplete words in captions

### Code Improvements
- Removed unused imports and improved code cleanliness
- Enhanced error handling in image captioning
- Better camera stream cleanup
- Enhanced food keyword extraction (100+ food items)

## License

This project is provided as-is for educational purposes.

---

**BetterBites** - Discover your perfect restaurant match in New York City! 🗽🍽️