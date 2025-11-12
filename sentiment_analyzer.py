"""
Sentiment Analysis Module for Yelp Reviews
Classifies reviews as positive or negative using multiple methods
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re


class SentimentAnalyzer:
    """Analyzes sentiment of restaurant reviews"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def analyze_with_vader(self, text):
        """Analyze sentiment using VADER (good for social media/text)"""
        scores = self.vader_analyzer.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def analyze_with_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        return {
            'polarity': polarity,
            'subjectivity': subjectivity
        }
    
    def classify_sentiment(self, text, method='vader'):
        """
        Classify sentiment as positive or negative
        
        Args:
            text: Review text
            method: 'vader' or 'textblob'
        
        Returns:
            'positive', 'negative', or 'neutral'
        """
        if method == 'vader':
            scores = self.analyze_with_vader(text)
            compound = scores['compound']
            if compound >= 0.05:
                return 'positive'
            elif compound <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        else:  # textblob
            scores = self.analyze_with_textblob(text)
            polarity = scores['polarity']
            if polarity > 0:
                return 'positive'
            elif polarity < 0:
                return 'negative'
            else:
                return 'neutral'
    
    def analyze_reviews(self, df, text_column='text', method='vader'):
        """
        Analyze sentiment for all reviews in dataframe
        
        Args:
            df: DataFrame with reviews
            text_column: Name of column containing review text
            method: 'vader' or 'textblob'
        
        Returns:
            DataFrame with sentiment columns added
        """
        df = df.copy()
        
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Analyze sentiment
        if method == 'vader':
            sentiment_scores = df['cleaned_text'].apply(self.analyze_with_vader)
            df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
            df['sentiment_positive'] = sentiment_scores.apply(lambda x: x['positive'])
            df['sentiment_negative'] = sentiment_scores.apply(lambda x: x['negative'])
        else:
            sentiment_scores = df['cleaned_text'].apply(self.analyze_with_textblob)
            df['sentiment_polarity'] = sentiment_scores.apply(lambda x: x['polarity'])
            df['sentiment_subjectivity'] = sentiment_scores.apply(lambda x: x['subjectivity'])
        
        # Classify sentiment
        df['sentiment'] = df['cleaned_text'].apply(
            lambda x: self.classify_sentiment(x, method=method)
        )
        
        return df

