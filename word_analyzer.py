"""
Word Frequency Analysis Module
Extracts and analyzes most common words from reviews
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data if not already present"""
    import os
    import platform
    
    # Set NLTK data path - cross-platform
    if platform.system() == 'Windows':
        nltk_data_dir = os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'nltk_data')
    else:
        # Linux/Mac
        nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add to NLTK path
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    
    # Download punkt and punkt_tab (both may be needed)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=False)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=False)
        except:
            pass  # punkt_tab might not be available in all versions

    # Download stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=False)

    # Download wordnet
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet', quiet=False)
    
    # Download omw-1.4 for wordnet
    try:
        nltk.download('omw-1.4', quiet=False)
    except:
        pass  # May already be included

# Download on import
download_nltk_data()


class WordAnalyzer:
    """Analyzes word frequency and patterns in reviews"""
    
    def __init__(self):
        # Ensure NLTK data is downloaded
        download_nltk_data()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Downloading stopwords...")
            nltk.download('stopwords', quiet=False)
            self.stop_words = set(stopwords.words('english'))
        
        # Add common restaurant review stopwords
        self.stop_words.update(['yelp', 'restaurant', 'place', 'food', 'would', 'get', 'got'])
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text):
        """Tokenize, remove stopwords, and lemmatize"""
        if pd.isna(text) or text == "":
            return []
        
        text_str = str(text).lower()
        
        # Try to use NLTK word_tokenize, with fallback to simple regex tokenization
        try:
            # Tokenize using NLTK
            tokens = word_tokenize(text_str)
        except (LookupError, OSError) as e:
            # If punkt tokenizer is missing, try to download it
            try:
                import os
                import platform
                # Cross-platform NLTK data path
                if platform.system() == 'Windows':
                    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'nltk_data')
                else:
                    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
                if nltk_data_dir not in nltk.data.path:
                    nltk.data.path.append(nltk_data_dir)
                nltk.download('punkt', quiet=False)
                # Also try punkt_tab (needed for newer NLTK versions)
                try:
                    nltk.download('punkt_tab', quiet=False)
                except:
                    pass
                tokens = word_tokenize(text_str)
            except:
                # Fallback to simple regex-based tokenization
                import re
                tokens = re.findall(r'\b[a-z]+\b', text_str)
        
        # Remove stopwords and non-alphabetic tokens
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token.isalpha() and token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def get_word_frequencies(self, texts, min_freq=2):
        """
        Get word frequency counts from a list of texts
        
        Args:
            texts: List of text strings or Series
            min_freq: Minimum frequency to include word
        
        Returns:
            Counter object with word frequencies
        """
        all_tokens = []
        for text in texts:
            tokens = self.preprocess_text(text)
            all_tokens.extend(tokens)
        
        word_freq = Counter(all_tokens)
        
        # Filter by minimum frequency
        if min_freq > 1:
            word_freq = Counter({word: count for word, count in word_freq.items() if count >= min_freq})
        
        return word_freq
    
    def get_top_words(self, texts, top_n=50, min_freq=2):
        """
        Get top N most common words
        
        Args:
            texts: List of text strings or Series
            top_n: Number of top words to return
            min_freq: Minimum frequency to include word
        
        Returns:
            List of tuples (word, frequency) sorted by frequency
        """
        word_freq = self.get_word_frequencies(texts, min_freq=min_freq)
        return word_freq.most_common(top_n)
    
    def analyze_by_sentiment(self, df, text_column='cleaned_text', sentiment_column='sentiment'):
        """
        Analyze word frequencies separately for positive and negative reviews
        
        Args:
            df: DataFrame with reviews and sentiment
            text_column: Column name with review text
            sentiment_column: Column name with sentiment labels
        
        Returns:
            Dictionary with 'positive' and 'negative' word frequencies
        """
        positive_texts = df[df[sentiment_column] == 'positive'][text_column]
        negative_texts = df[df[sentiment_column] == 'negative'][text_column]
        
        positive_words = self.get_word_frequencies(positive_texts)
        negative_words = self.get_word_frequencies(negative_texts)
        
        return {
            'positive': positive_words,
            'negative': negative_words
        }
    
    def get_sentiment_specific_words(self, positive_words, negative_words, top_n=30):
        """
        Find words that are more common in positive vs negative reviews
        
        Args:
            positive_words: Counter of positive review words
            negative_words: Counter of negative review words
            top_n: Number of top words to return for each
        
        Returns:
            Dictionary with 'positive_specific' and 'negative_specific' word lists
        """
        # Calculate relative frequencies
        total_positive = sum(positive_words.values())
        total_negative = sum(negative_words.values())
        
        # Words more common in positive reviews
        positive_specific = {}
        for word in set(positive_words.keys()) | set(negative_words.keys()):
            pos_freq = positive_words.get(word, 0) / total_positive if total_positive > 0 else 0
            neg_freq = negative_words.get(word, 0) / total_negative if total_negative > 0 else 0
            
            if pos_freq > neg_freq and pos_freq > 0:
                positive_specific[word] = {
                    'positive_freq': pos_freq,
                    'negative_freq': neg_freq,
                    'ratio': pos_freq / neg_freq if neg_freq > 0 else pos_freq * 100
                }
        
        # Words more common in negative reviews
        negative_specific = {}
        for word in set(positive_words.keys()) | set(negative_words.keys()):
            pos_freq = positive_words.get(word, 0) / total_positive if total_positive > 0 else 0
            neg_freq = negative_words.get(word, 0) / total_negative if total_negative > 0 else 0
            
            if neg_freq > pos_freq and neg_freq > 0:
                negative_specific[word] = {
                    'positive_freq': pos_freq,
                    'negative_freq': neg_freq,
                    'ratio': neg_freq / pos_freq if pos_freq > 0 else neg_freq * 100
                }
        
        # Sort and get top N
        top_positive = sorted(
            positive_specific.items(), 
            key=lambda x: x[1]['ratio'], 
            reverse=True
        )[:top_n]
        
        top_negative = sorted(
            negative_specific.items(), 
            key=lambda x: x[1]['ratio'], 
            reverse=True
        )[:top_n]
        
        return {
            'positive_specific': top_positive,
            'negative_specific': top_negative
        }

