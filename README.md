# California Restaurant Finder

A Flask web application for searching and viewing California restaurant reviews from the Yelp Academic Dataset. The app displays restaurant information along with the top positive and negative reviews.

## Features

- **Restaurant Search**: Search for California restaurants by name
- **Review Display**: View top 5 positive and negative reviews for each restaurant
- **Dark Mode UI**: Modern dark-themed interface
- **Fast Search**: SQLite database for quick queries
- **Secure**: XSS protection and input validation

## Installation

### Prerequisites

- Python 3.7 or higher
- Yelp Academic Dataset files (business and review JSON files)

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

3. **Set up the database:**
   
   Download the Yelp Academic Dataset from [Yelp's website](https://www.yelp.com/dataset) and extract the JSON files.
   
   Then run:
   ```bash
   python setup_db.py --business-json path/to/yelp_academic_dataset_business.json --review-json path/to/yelp_academic_dataset_review.json
   ```
   
   For testing with limited data:
   ```bash
   python setup_db.py --business-json path/to/business.json --review-json path/to/review.json --business-limit 100 --review-limit 1000
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```
   
   Or using Flask CLI:
   ```bash
   flask --app app.py --debug run
   ```

5. **Access the app:**
   
   Open your browser and navigate to: `http://127.0.0.1:5000`

## Usage

1. Enter a restaurant name in the search box
2. Click "Search" to find matching restaurants
3. View restaurant details, ratings, and top reviews

## Project Structure

```
.
├── app.py                  # Flask web application
├── setup_db.py             # Database setup script
├── templates/
│   └── index.html          # Frontend template
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## API Endpoints

### `GET /`
Serves the main search page.

### `GET /api/search?query=<restaurant_name>`
Searches for restaurants matching the query.

**Response:**
```json
{
  "business": {
    "business_id": "...",
    "name": "Restaurant Name",
    "address": "...",
    "city": "...",
    "state": "CA",
    "stars": 4.5,
    "review_count": 123,
    "categories": ["Restaurants", "Italian"]
  },
  "top_positive_reviews": [...],
  "top_negative_reviews": [...]
}
```

## Database Schema

### businesses
- `business_id` (TEXT PRIMARY KEY)
- `name`, `address`, `city`, `state`, `postal_code`
- `latitude`, `longitude`
- `stars`, `review_count`, `is_open`
- `categories`

### reviews
- `review_id` (TEXT PRIMARY KEY)
- `business_id` (FOREIGN KEY)
- `stars`, `text`, `date`
- `useful`, `funny`, `cool`

## Requirements

- Python 3.7+
- Flask >= 2.3.0
- See `requirements.txt` for full list

## Notes

- The database contains only California restaurants
- Large dataset files are excluded from git (see `.gitignore`)
- The app uses SQLite for simplicity and portability
- For production, consider using PostgreSQL or MySQL

## License

This project is provided as-is for educational purposes.
