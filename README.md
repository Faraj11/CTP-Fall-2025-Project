# Yelp Review Analysis MVP

A comprehensive analysis tool for extracting insights from Yelp restaurant reviews. This MVP analyzes sentiment, identifies common words, extracts themes (Ambiance, Service, Crowdedness, etc.), and provides actionable insights.

## Features

- **Sentiment Analysis**: Classifies reviews as positive, negative, or neutral using VADER sentiment analyzer
- **Word Frequency Analysis**: Identifies most common words in positive vs negative reviews
- **Theme Extraction**: Analyzes mentions of key themes:
  - Ambiance/Atmosphere
  - Service quality
  - Crowdedness
  - Food quality
  - Price/Value
  - Location
  - Wait times
- **Visualizations**: Generates charts and word clouds for easy interpretation
- **Sentiment-Specific Words**: Finds words that are more common in positive vs negative reviews

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. The script will automatically download required NLTK data on first run.

## Usage

### Option 1: Jupyter Notebook (Recommended for Interactive Analysis)

1. **Install Jupyter** (if not already installed):
   ```bash
   pip install jupyter notebook
   # or
   pip install jupyterlab
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

3. **Open the notebook**: Navigate to `yelp_analysis.ipynb` and open it

4. **Run the cells**: 
   - Update the `DATA_FILE` path in the data loading cell
   - Run all cells (Cell → Run All) or run them individually (Shift + Enter)
   - All visualizations will display inline in the notebook

The notebook provides an interactive environment with:
- Step-by-step analysis
- Inline visualizations
- Easy parameter adjustment
- Exportable results

### Option 2: Command Line

```bash
python main_analysis.py --data path/to/your/reviews.json --format json
```

### Command Line Options

- `--data`: Path to your Yelp review data file (required)
- `--format`: Data format - `json` or `csv` (default: `json`)
- `--sample`: Optional sample size for faster testing (e.g., `--sample 1000`)
- `--text-column`: Name of the column containing review text (default: `text`)
- `--output`: Output directory for visualizations (default: `output`)

### Example Commands

```bash
# Analyze full dataset (JSON format)
python main_analysis.py --data yelp_reviews.json --format json

# Analyze sample of 5000 reviews
python main_analysis.py --data yelp_reviews.json --format json --sample 5000

# Use CSV format with custom text column
python main_analysis.py --data reviews.csv --format csv --text-column review_text

# Custom output directory
python main_analysis.py --data yelp_reviews.json --output results
```

### Using as a Python Module

```python
from main_analysis import YelpReviewAnalyzer

# Initialize analyzer
analyzer = YelpReviewAnalyzer()

# Load data
analyzer.load_data('yelp_reviews.json', format='json')

# Run full analysis
results = analyzer.run_full_analysis(text_column='text', sample_size=5000)

# Print summary
analyzer.print_summary()

# Generate visualizations
analyzer.create_visualizations(output_dir='output')
```

## Data Format

### JSON Format (Yelp Dataset Standard)
Each line should be a JSON object with at least a text field:
```json
{"text": "Great food and service!", "stars": 5, "business_id": "abc123"}
{"text": "Terrible experience.", "stars": 1, "business_id": "xyz789"}
```

### CSV Format
CSV file with at least one column containing review text:
```csv
text,stars,business_id
"Great food and service!",5,abc123
"Terrible experience.",1,xyz789
```

## Output

The analysis generates:

1. **Console Output**: Summary statistics and insights
2. **Visualizations** (in `output/` directory):
   - `sentiment_distribution.png`: Distribution of positive/negative/neutral reviews
   - `top_words_comparison.png`: Side-by-side comparison of top words
   - `theme_analysis.png`: Theme mentions by sentiment
   - `wordcloud_positive.png`: Word cloud for positive reviews
   - `wordcloud_negative.png`: Word cloud for negative reviews
3. **Results JSON** (`output/analysis_results.json`): Complete analysis results in JSON format

## Project Structure

```
.
├── main_analysis.py          # Main analysis script (command line)
├── yelp_analysis.ipynb       # Jupyter notebook for interactive analysis
├── sentiment_analyzer.py     # Sentiment analysis module
├── word_analyzer.py          # Word frequency analysis module
├── theme_extractor.py        # Theme extraction module
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                 # This file
```

## Key Insights Provided

1. **Sentiment Distribution**: Overall positive vs negative review ratio
2. **Most Common Words**: Top words in positive and negative reviews separately
3. **Sentiment-Specific Words**: Words that appear more frequently in one sentiment category
4. **Theme Analysis**: 
   - Which themes are mentioned more in positive vs negative reviews
   - Percentage of reviews mentioning each theme
   - Average mentions per review for each theme

## Example Insights

The analysis can reveal patterns like:
- "Service" is mentioned 45% more in negative reviews than positive
- "Ambiance" appears 30% more in positive reviews
- Top positive words: "great", "delicious", "friendly", "amazing"
- Top negative words: "slow", "rude", "overpriced", "disappointed"

## Customization

### Adding Custom Themes

Edit `theme_extractor.py` and add keywords to the `theme_keywords` dictionary:

```python
self.theme_keywords = {
    'your_theme': ['keyword1', 'keyword2', 'keyword3'],
    # ... existing themes
}
```

### Adjusting Sentiment Thresholds

Modify the `classify_sentiment` method in `sentiment_analyzer.py`:

```python
if compound >= 0.05:  # Adjust threshold
    return 'positive'
```


## Setup for database creation
``` ar -xf yelp_dataset.tar yelp_academic_dataset_business.json yelp_academic_dataset_review.json
```
```python -m venv .venv```
then run  
```.\.venv\Scripts\Activate.ps1 ``` and 
```pip install flask.```
```python setup_db.py``` (use --business-limit / --review-limit for dry runs)
```flask --app app.py --debug run```

## Requirements

- Python 3.7+
- See `requirements.txt` for full list of dependencies



## Notes

- For large datasets, use the `--sample` option to test with a subset first
- Processing time depends on dataset size (approximately 1000 reviews/second)
- The analysis automatically handles missing or malformed data

## License

This project is provided as-is for MVP purposes.


