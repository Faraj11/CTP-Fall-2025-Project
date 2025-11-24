# BetterBites 🍽️

A modern Flask web application for discovering restaurants in New York City. BetterBites helps users find restaurants based on cuisine, location, and other preferences with an intelligent matching algorithm optimized for discovery.

## Features

- **Smart Restaurant Discovery**: Search by cuisine and location (e.g., "asian food in queens")
- **Comprehensive Restaurant Data**: View 9 key details for each restaurant:
  - Borough & Neighborhood
  - Price Category (normalized to $, $$, $$$, $$$$)
  - Cuisine (normalized and cleaned)
  - Dietary Accommodations
  - Dining Style
  - Food, Service, and Ambiance Ratings
  - Match Score Percentage
- **Match Score System**: Each restaurant displays a match score percentage in green
- **Modern Dark UI**: Beautiful gradient dark theme with smooth animations
- **Normalized Data**: All fields are normalized for consistency (cuisine, dining style, price categories)
- **Address Lookup**: Automatically fills missing addresses when possible

## Installation

### Prerequisites

- Python 3.7 or higher
- CSV files: `nyc_restaurants.csv` and `kayak_data.csv` (or the merged `nyc_restaurants_merged.csv`)

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

3. **Prepare the data:**
   
   If you have separate CSV files, merge them first:
   ```bash
   python merge_restaurants.py
   ```
   
   This will create `nyc_restaurants_merged.csv` in the same directory as the script.
   
   **Note:** Update the CSV file paths in `merge_restaurants.py` to point to your data files.

4. **Update CSV path in app.py:**
   
   Edit `app.py` and update the `CSV_PATH` variable to point to your merged CSV file:
   ```python
   CSV_PATH = Path("path/to/nyc_restaurants_merged.csv")
   ```

5. **Run the application:**
   ```bash
   python app.py
   ```

6. **Access the app:**
   
   Open your browser and navigate to: `http://127.0.0.1:5000`

## Usage

1. Enter a search query in the search box (e.g., "italian food in manhattan", "asian food in queens")
2. Click "Search" to find matching restaurants
3. View the best match with all 9 tiles displayed
4. Click "View All Matches" to see all matching restaurants
5. Click on any match in the grid to view its details

## Project Structure

```
.
├── app.py                      # Flask web application
├── merge_restaurants.py        # Script to merge restaurant CSV files
├── templates/
│   └── index.html              # Frontend template
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## API Endpoints

### `GET /`
Serves the main search page.

### `GET /api/search?query=<search_query>`
Searches for restaurants matching the query.

**Response:**
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
    "dietary_accommodations": "None",
    "match_score_percentile": 95.5
  },
  "all_matches": [...]
}
```

## Matching Algorithm

The app uses an intelligent matching algorithm optimized for discovery queries:

- **Cuisine Matching (40% weight)**: Prioritizes cuisine matches for queries like "asian food"
- **Location Matching (35% weight)**: Matches borough and neighborhood for location-based queries
- **Name Matching (10% weight)**: Lower priority for general discovery
- **Rating Boost (10% weight)**: Considers restaurant quality
- **Review Count Boost (5% weight)**: Considers popularity

## Data Normalization

All data fields are normalized for consistency:

- **Price Category**: Normalized to $, $$, $$$, $$$$ format
- **Cuisine**: Special characters removed (Café → Cafe, Thaï → Thai), proper capitalization
- **Dining Style**: Standardized capitalization
- **Borough**: Title case normalization
- **Neighborhood**: Preserves proper capitalization for names like "SoHo", "NoHo"

## Requirements

- Python 3.7+
- Flask >= 2.3.0
- Pandas >= 2.0.0
- See `requirements.txt` for full list

## Notes

- The app requires a merged CSV file with restaurant data
- Large CSV files should be excluded from git (see `.gitignore`)
- The matching algorithm is optimized for discovery queries rather than exact name searches
- Address lookup attempts to fill missing addresses by matching restaurant names

## License

This project is provided as-is for educational purposes.
