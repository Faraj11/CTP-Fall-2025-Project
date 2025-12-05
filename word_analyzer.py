"""
Word Frequency Analysis Module for Yelp Reviews
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class WordAnalyzer:
    def __init__(self):
        """Initialize the word analyzer."""
        # Download required NLTK data - ensure punkt_tab is available
        self._ensure_nltk_data()
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def _ensure_nltk_data(self):
        """Ensure all required NLTK data is downloaded."""
        # Download punkt_tab (required for newer NLTK)
        try:
            nltk.data.find('tokenizers/punkt_tab/english')
        except LookupError:
            print("[*] Downloading NLTK punkt_tab...")
            try:
                nltk.download('punkt_tab', quiet=False)
                print("[OK] punkt_tab downloaded")
            except Exception as e:
                print(f"[WARNING] punkt_tab download failed: {e}, trying punkt...")
                try:
                    nltk.download('punkt', quiet=False)
                    print("[OK] punkt downloaded as fallback")
                except Exception as e2:
                    print(f"[ERROR] Both punkt_tab and punkt download failed: {e2}")
        
        # Download stopwords
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("[*] Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
            
        # Download wordnet
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("[*] Downloading NLTK wordnet...")
            nltk.download('wordnet', quiet=True)
            
        # Download omw-1.4
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            print("[*] Downloading NLTK omw-1.4...")
            try:
                nltk.download('omw-1.4', quiet=True)
            except:
                pass  # omw-1.4 is optional
        
        # Add custom stop words for restaurant reviews
        custom_stops = {
            'restaurant', 'place', 'food', 'eat', 'go', 'get', 'got', 'come', 
            'came', 'went', 'would', 'could', 'should', 'one', 'two', 'three',
            'also', 'really', 'very', 'much', 'well', 'way', 'time', 'times',
            'first', 'last', 'next', 'day', 'night', 'week', 'year', 'years'
        }
        self.stop_words.update(custom_stops)
    
    def preprocess_text(self, text):
        """Preprocess text for word analysis."""
        if pd.isna(text) or not text:
            return []
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters, keep only letters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize - ensure NLTK data is available
        try:
            tokens = word_tokenize(text)
        except LookupError as e:
            # If punkt_tab is missing, try to download it now
            if 'punkt_tab' in str(e) or 'punkt' in str(e):
                print("[*] punkt_tab missing during tokenization, downloading now...")
                try:
                    nltk.download('punkt_tab', quiet=False)
                except:
                    nltk.download('punkt', quiet=False)
                tokens = word_tokenize(text)
            else:
                raise
        
        # Remove stop words and short words
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def get_top_words(self, texts, top_n=20):
        """
        Get top N most frequent words from a collection of texts.
        
        Args:
            texts: Series or list of text strings
            top_n: Number of top words to return
        
        Returns:
            List of tuples (word, frequency)
        """
        print(f"Downloading NLTK wordnet...")
        
        all_words = []
        
        print(f"  Analyzing {len(texts)} reviews...")
        
        for text in texts:
            words = self.preprocess_text(text)
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        return word_counts.most_common(top_n)
    
    def get_word_frequencies(self, texts):
        """
        Get word frequencies from a collection of texts.
        
        Args:
            texts: Series or list of text strings
        
        Returns:
            Counter object with word frequencies
        """
        from collections import Counter
        
        all_words = []
        for text in texts:
            words = self.preprocess_text(text)
            all_words.extend(words)
        
        return Counter(all_words)
    
    def compare_word_frequencies(self, positive_texts, negative_texts, top_n=15):
        """
        Compare word frequencies between positive and negative reviews.
        
        Args:
            positive_texts: Series of positive review texts
            negative_texts: Series of negative review texts
            top_n: Number of top words to analyze
        
        Returns:
            Dictionary with comparison results
        """
        print("Comparing word frequencies between positive and negative reviews...")
        
        # Get word frequencies for each sentiment
        pos_words = dict(self.get_top_words(positive_texts, top_n * 3))  # Get more to ensure good comparison
        neg_words = dict(self.get_top_words(negative_texts, top_n * 3))
        
        # Calculate ratios and differences
        all_words = set(pos_words.keys()) | set(neg_words.keys())
        
        word_ratios = []
        
        for word in all_words:
            pos_count = pos_words.get(word, 0)
            neg_count = neg_words.get(word, 0)
            
            # Calculate normalized frequencies (per 1000 reviews)
            pos_freq = (pos_count / len(positive_texts)) * 1000 if len(positive_texts) > 0 else 0
            neg_freq = (neg_count / len(negative_texts)) * 1000 if len(negative_texts) > 0 else 0
            
            # Calculate ratio (avoid division by zero and infinity)
            if neg_freq > 0:
                ratio = pos_freq / neg_freq
            elif pos_freq > 0:
                ratio = 999.0  # Use large number instead of infinity
            else:
                ratio = 1.0  # Neither
            
            word_ratios.append({
                'word': word,
                'pos_count': pos_count,
                'neg_count': neg_count,
                'pos_freq': pos_freq,
                'neg_freq': neg_freq,
                'ratio': ratio
            })
        
        # Sort by ratio
        word_ratios.sort(key=lambda x: x['ratio'], reverse=True)
        
        # Get words more common in positive reviews
        positive_words = [w for w in word_ratios if w['ratio'] > 1.5][:top_n]
        
        # Get words more common in negative reviews  
        negative_words = [w for w in word_ratios if w['ratio'] < 0.67][:top_n]
        negative_words.sort(key=lambda x: x['ratio'])  # Sort ascending for negative
        
        return {
            'positive_words': positive_words,
            'negative_words': negative_words,
            'all_comparisons': word_ratios
        }
