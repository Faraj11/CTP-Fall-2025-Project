# Project Structure

```
yelp-review-analysis/
├── main_analysis.py          # Command-line analysis script
├── yelp_analysis.ipynb       # Jupyter notebook for interactive analysis
├── sentiment_analyzer.py     # Sentiment analysis module
├── word_analyzer.py          # Word frequency analysis module
├── theme_extractor.py        # Theme extraction module
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore rules
├── run_analysis.bat         # Windows batch script (optional)
└── run_analysis.ps1          # PowerShell script (optional)
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with Jupyter Notebook (Recommended):**
   ```bash
   jupyter notebook
   # Open yelp_analysis.ipynb
   ```

3. **Or run from command line:**
   ```bash
   python main_analysis.py --data your_reviews.json --format json --sample 5000
   ```

## Features

- ✅ Sentiment Analysis (Positive/Negative/Neutral)
- ✅ Word Frequency Analysis
- ✅ Theme Extraction (Ambiance, Service, Crowdedness, etc.)
- ✅ Interactive Visualizations
- ✅ Cross-platform compatible

## Notes

- Update file paths in the notebook or command line arguments
- The notebook will automatically download NLTK data on first run
- For large datasets, use the `--sample` parameter for testing

