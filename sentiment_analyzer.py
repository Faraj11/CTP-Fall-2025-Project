"""
Sentiment Analysis Module for Yelp Reviews
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Download required NLTK data
        # Try punkt_tab first (newer NLTK versions), fallback to punkt
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                try:
                    nltk.download('punkt_tab', quiet=True)
                except:
                    nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def clean_text(self, text):
        """Clean and preprocess text."""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_vader_sentiment(self, text):
        """Get sentiment using VADER."""
        if not text or pd.isna(text):
            return 'neutral', 0.0
        
        scores = self.vader_analyzer.polarity_scores(str(text))
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'positive', compound
        elif compound <= -0.05:
            return 'negative', compound
        else:
            return 'neutral', compound
    
    def get_textblob_sentiment(self, text):
        """Get sentiment using TextBlob."""
        if not text or pd.isna(text):
            return 'neutral', 0.0
        
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive', polarity
        elif polarity < -0.1:
            return 'negative', polarity
        else:
            return 'neutral', polarity
    
    def analyze_reviews(self, df, text_column='text', method='vader'):
        """
        Analyze sentiment of reviews in a DataFrame.
        
        Args:
            df: DataFrame with review text
            text_column: Name of column containing review text
            method: 'vader' or 'textblob'
        
        Returns:
            DataFrame with added sentiment columns
        """
        print("Analyzing sentiment...")
        
        # Make a copy to avoid modifying original
        result_df = df.copy()
        
        # Clean text
        result_df['cleaned_text'] = result_df[text_column].apply(self.clean_text)
        
        # Analyze sentiment
        if method == 'vader':
            sentiment_results = result_df[text_column].apply(self.get_vader_sentiment)
        else:
            sentiment_results = result_df[text_column].apply(self.get_textblob_sentiment)
        
        # Extract sentiment and scores
        result_df['sentiment'] = sentiment_results.apply(lambda x: x[0])
        result_df['sentiment_score'] = sentiment_results.apply(lambda x: x[1])
        
        # Print summary
        sentiment_counts = result_df['sentiment'].value_counts()
        total = len(result_df)
        
        print("\nSentiment Distribution:")
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in sentiment_counts:
                count = sentiment_counts[sentiment]
                pct = (count / total) * 100
                print(f"  {sentiment.title()}: {count} ({pct:.1f}%)")
        
        print("\n[OK] Sentiment analysis complete")
        
        return result_df
