"""
Theme Extraction Module for Restaurant Reviews
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict

class ThemeExtractor:
    def __init__(self):
        """Initialize the theme extractor with predefined themes and keywords."""
        
        # Define themes and their associated keywords
        self.themes = {
            'service': [
                'service', 'server', 'waiter', 'waitress', 'staff', 'employee', 
                'friendly', 'rude', 'helpful', 'attentive', 'slow', 'fast',
                'professional', 'unprofessional', 'polite', 'courteous'
            ],
            'food_quality': [
                'delicious', 'tasty', 'flavor', 'fresh', 'stale', 'bland', 
                'seasoned', 'cooked', 'overcooked', 'undercooked', 'quality',
                'amazing', 'terrible', 'disgusting', 'wonderful', 'awful'
            ],
            'ambiance': [
                'atmosphere', 'ambiance', 'ambience', 'decor', 'music', 'lighting',
                'cozy', 'comfortable', 'noisy', 'loud', 'quiet', 'romantic',
                'casual', 'upscale', 'clean', 'dirty', 'beautiful', 'ugly'
            ],
            'price_value': [
                'price', 'expensive', 'cheap', 'affordable', 'value', 'worth',
                'overpriced', 'reasonable', 'cost', 'money', 'budget', 'deal'
            ],
            'wait_time': [
                'wait', 'waiting', 'quick', 'slow', 'fast', 'time', 'minutes',
                'hour', 'long', 'short', 'immediately', 'forever', 'prompt'
            ],
            'location': [
                'location', 'parking', 'convenient', 'accessible', 'downtown',
                'neighborhood', 'area', 'street', 'building', 'address'
            ],
            'crowdedness': [
                'busy', 'crowded', 'packed', 'empty', 'full', 'space', 'room',
                'table', 'seat', 'reservation', 'line', 'queue'
            ]
        }
    
    def extract_themes_from_text(self, text):
        """
        Extract themes mentioned in a single text.
        
        Args:
            text: Review text string
        
        Returns:
            Dictionary with theme names as keys and boolean values
        """
        if pd.isna(text) or not text:
            return {theme: False for theme in self.themes.keys()}
        
        text_lower = str(text).lower()
        
        theme_mentions = {}
        
        for theme_name, keywords in self.themes.items():
            # Check if any keyword for this theme appears in the text
            mentioned = any(keyword in text_lower for keyword in keywords)
            theme_mentions[theme_name] = mentioned
        
        return theme_mentions
    
    def extract_themes_from_reviews(self, df, text_column='text'):
        """
        Extract themes from all reviews in a DataFrame.
        
        Args:
            df: DataFrame with review data
            text_column: Name of column containing review text
        
        Returns:
            DataFrame with added theme columns
        """
        print("Extracting themes...")
        
        # Make a copy to avoid modifying original
        result_df = df.copy()
        
        # Extract themes for each review
        theme_data = result_df[text_column].apply(self.extract_themes_from_text)
        
        # Convert to separate columns
        for theme in self.themes.keys():
            result_df[f'theme_{theme}'] = theme_data.apply(lambda x: x[theme])
        
        print("\n✓ Theme extraction complete")
        
        return result_df
    
    def get_theme_insights(self, df):
        """
        Analyze theme mentions by sentiment.
        
        Args:
            df: DataFrame with sentiment and theme columns
        
        Returns:
            Dictionary with theme insights
        """
        insights = {}
        
        print("🎯 THEME INSIGHTS:\n")
        
        for theme in self.themes.keys():
            theme_col = f'theme_{theme}'
            
            if theme_col not in df.columns:
                continue
            
            # Calculate mention rates by sentiment
            positive_reviews = df[df['sentiment'] == 'positive']
            negative_reviews = df[df['sentiment'] == 'negative']
            
            pos_mentions = positive_reviews[theme_col].sum()
            pos_total = len(positive_reviews)
            pos_rate = (pos_mentions / pos_total * 100) if pos_total > 0 else 0
            
            neg_mentions = negative_reviews[theme_col].sum()
            neg_total = len(negative_reviews)
            neg_rate = (neg_mentions / neg_total * 100) if neg_total > 0 else 0
            
            # Calculate difference
            difference = pos_rate - neg_rate
            more_in_positive = difference > 0
            
            insights[theme] = {
                'positive_mentions': pos_mentions,
                'positive_total': pos_total,
                'positive_mention_rate': pos_rate,
                'negative_mentions': neg_mentions,
                'negative_total': neg_total,
                'negative_mention_rate': neg_rate,
                'difference': difference,
                'more_in_positive': more_in_positive
            }
            
            # Print insight
            print(f"{theme.upper().replace('_', ' ')}:")
            print(f"  Positive reviews: {pos_rate:.1f}% mention this theme")
            print(f"  Negative reviews: {neg_rate:.1f}% mention this theme")
            direction = "positive" if more_in_positive else "negative"
            print(f"  Difference: {difference:+.1f}% (more in {direction})")
            print()
        
        return insights
