"""
Theme Extraction Module
Identifies common themes like Ambiance, Service, Crowdedness, etc.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import re


class ThemeExtractor:
    """Extracts themes from restaurant reviews"""
    
    def __init__(self):
        # Define theme keywords
        self.theme_keywords = {
            'ambiance': [
                'ambiance', 'atmosphere', 'ambience', 'decor', 'decorated', 'decoration',
                'cozy', 'romantic', 'intimate', 'lively', 'vibrant', 'quiet', 'loud',
                'music', 'lighting', 'dim', 'bright', 'elegant', 'casual', 'upscale',
                'rustic', 'modern', 'traditional', 'chic', 'stylish', 'welcoming'
            ],
            'service': [
                'service', 'server', 'waiter', 'waitress', 'staff', 'host', 'hostess',
                'attentive', 'friendly', 'helpful', 'professional', 'rude', 'slow',
                'fast', 'efficient', 'inefficient', 'manager', 'owner', 'tip',
                'service', 'served', 'serving', 'order', 'ordered', 'check', 'bill'
            ],
            'crowdedness': [
                'crowded', 'busy', 'packed', 'full', 'empty', 'quiet', 'loud',
                'wait', 'waiting', 'reservation', 'walk-in', 'line', 'queue',
                'seating', 'table', 'available', 'full house', 'rush', 'peak',
                'off-peak', 'weekend', 'weekday', 'lunch', 'dinner', 'brunch'
            ],
            'food_quality': [
                'delicious', 'tasty', 'flavorful', 'bland', 'fresh', 'stale',
                'cooked', 'undercooked', 'overcooked', 'burnt', 'raw', 'quality',
                'ingredients', 'presentation', 'plating', 'portion', 'portions',
                'appetizer', 'entree', 'main', 'dessert', 'menu', 'cuisine'
            ],
            'price_value': [
                'price', 'prices', 'expensive', 'cheap', 'affordable', 'value',
                'worth', 'overpriced', 'reasonable', 'budget', 'cost', 'bill',
                'check', 'tip', 'dollar', 'dollars', 'money', 'pay', 'paid'
            ],
            'location': [
                'location', 'located', 'parking', 'street', 'corner', 'downtown',
                'neighborhood', 'area', 'accessible', 'convenient', 'nearby',
                'walking distance', 'drive', 'metro', 'subway', 'bus'
            ],
            'wait_time': [
                'wait', 'waited', 'waiting', 'time', 'minutes', 'hour', 'hours',
                'quick', 'fast', 'slow', 'immediate', 'prompt', 'delayed',
                'reservation', 'walk-in', 'seated', 'seating'
            ]
        }
        
        # Create regex patterns for each theme
        self.theme_patterns = {}
        for theme, keywords in self.theme_keywords.items():
            pattern = r'\b(?:' + '|'.join(keywords) + r')\b'
            self.theme_patterns[theme] = re.compile(pattern, re.IGNORECASE)
    
    def extract_themes_from_text(self, text):
        """
        Extract themes mentioned in a single review text
        
        Args:
            text: Review text string
        
        Returns:
            Dictionary with theme names as keys and counts as values
        """
        if pd.isna(text) or text == "":
            return {}
        
        text = str(text).lower()
        theme_counts = {}
        
        for theme, pattern in self.theme_patterns.items():
            matches = pattern.findall(text)
            if matches:
                theme_counts[theme] = len(matches)
        
        return theme_counts
    
    def extract_themes_from_reviews(self, df, text_column='cleaned_text'):
        """
        Extract themes from all reviews
        
        Args:
            df: DataFrame with reviews
            text_column: Column name with review text
        
        Returns:
            DataFrame with theme columns added
        """
        df = df.copy()
        
        # Extract themes for each review
        theme_data = df[text_column].apply(self.extract_themes_from_text)
        
        # Create columns for each theme
        for theme in self.theme_keywords.keys():
            df[f'theme_{theme}'] = theme_data.apply(lambda x: x.get(theme, 0))
            df[f'mentions_{theme}'] = df[f'theme_{theme}'] > 0
        
        return df
    
    def analyze_themes_by_sentiment(self, df, sentiment_column='sentiment'):
        """
        Analyze theme mentions by sentiment
        
        Args:
            df: DataFrame with themes and sentiment
            sentiment_column: Column name with sentiment labels
        
        Returns:
            Dictionary with theme statistics by sentiment
        """
        theme_stats = {}
        
        for theme in self.theme_keywords.keys():
            theme_col = f'mentions_{theme}'
            if theme_col not in df.columns:
                continue
            
            stats = {
                'positive': {
                    'count': len(df[(df[sentiment_column] == 'positive') & (df[theme_col] == True)]),
                    'percentage': len(df[(df[sentiment_column] == 'positive') & (df[theme_col] == True)]) / 
                                 len(df[df[sentiment_column] == 'positive']) * 100 
                                 if len(df[df[sentiment_column] == 'positive']) > 0 else 0,
                    'avg_mentions': df[df[sentiment_column] == 'positive'][f'theme_{theme}'].mean()
                },
                'negative': {
                    'count': len(df[(df[sentiment_column] == 'negative') & (df[theme_col] == True)]),
                    'percentage': len(df[(df[sentiment_column] == 'negative') & (df[theme_col] == True)]) / 
                                 len(df[df[sentiment_column] == 'negative']) * 100 
                                 if len(df[df[sentiment_column] == 'negative']) > 0 else 0,
                    'avg_mentions': df[df[sentiment_column] == 'negative'][f'theme_{theme}'].mean()
                }
            }
            
            theme_stats[theme] = stats
        
        return theme_stats
    
    def get_theme_insights(self, df, sentiment_column='sentiment'):
        """
        Generate insights about themes in positive vs negative reviews
        
        Args:
            df: DataFrame with themes and sentiment
            sentiment_column: Column name with sentiment labels
        
        Returns:
            Dictionary with insights for each theme
        """
        theme_stats = self.analyze_themes_by_sentiment(df, sentiment_column)
        insights = {}
        
        for theme, stats in theme_stats.items():
            pos_pct = stats['positive']['percentage']
            neg_pct = stats['negative']['percentage']
            
            insight = {
                'theme': theme,
                'positive_mention_rate': pos_pct,
                'negative_mention_rate': neg_pct,
                'difference': pos_pct - neg_pct,
                'more_in_positive': pos_pct > neg_pct,
                'avg_mentions_positive': stats['positive']['avg_mentions'],
                'avg_mentions_negative': stats['negative']['avg_mentions']
            }
            
            insights[theme] = insight
        
        return insights

