"""
Main Analysis Script for Yelp Review Analysis MVP
Combines sentiment analysis, word frequency, and theme extraction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import json
import os

from sentiment_analyzer import SentimentAnalyzer
from word_analyzer import WordAnalyzer
from theme_extractor import ThemeExtractor


class YelpReviewAnalyzer:
    """Main class for analyzing Yelp reviews"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.word_analyzer = WordAnalyzer()
        self.theme_extractor = ThemeExtractor()
        self.df = None
    
    def load_data(self, filepath, format='json'):
        """
        Load Yelp review data
        
        Args:
            filepath: Path to data file
            format: 'json' or 'csv'
        """
        print(f"Loading data from {filepath}...")
        
        if format == 'json':
            # For JSON files (common Yelp dataset format)
            reviews = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        reviews.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            self.df = pd.DataFrame(reviews)
        else:
            self.df = pd.read_csv(filepath)
        
        print(f"Loaded {len(self.df)} reviews")
        
        # Try to identify text column
        possible_text_cols = ['text', 'review_text', 'content', 'review', 'comment']
        text_col = None
        for col in possible_text_cols:
            if col in self.df.columns:
                text_col = col
                break
        
        if text_col is None and len(self.df.columns) > 0:
            text_col = self.df.columns[0]
            print(f"Using '{text_col}' as text column")
        
        return text_col
    
    def run_full_analysis(self, text_column='text', sample_size=None):
        """
        Run complete analysis pipeline
        
        Args:
            text_column: Name of column containing review text
            sample_size: If specified, analyze only a sample of reviews
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("\n" + "="*60)
        print("YELP REVIEW ANALYSIS - FULL PIPELINE")
        print("="*60)
        
        # Sample data if specified
        if sample_size and sample_size < len(self.df):
            print(f"\nSampling {sample_size} reviews for analysis...")
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Step 1: Sentiment Analysis
        print("\n[1/4] Analyzing sentiment...")
        self.df = self.sentiment_analyzer.analyze_reviews(
            self.df, 
            text_column=text_column, 
            method='vader'
        )
        
        sentiment_counts = self.df['sentiment'].value_counts()
        print(f"Sentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            pct = count / len(self.df) * 100
            print(f"  {sentiment.capitalize()}: {count} ({pct:.1f}%)")
        
        # Step 2: Word Frequency Analysis
        print("\n[2/4] Analyzing word frequencies...")
        positive_texts = self.df[self.df['sentiment'] == 'positive']['cleaned_text']
        negative_texts = self.df[self.df['sentiment'] == 'negative']['cleaned_text']
        
        print(f"  Analyzing {len(positive_texts)} positive reviews...")
        top_positive_words = self.word_analyzer.get_top_words(positive_texts, top_n=30)
        
        print(f"  Analyzing {len(negative_texts)} negative reviews...")
        top_negative_words = self.word_analyzer.get_top_words(negative_texts, top_n=30)
        
        # Get sentiment-specific words
        positive_word_freq = self.word_analyzer.get_word_frequencies(positive_texts)
        negative_word_freq = self.word_analyzer.get_word_frequencies(negative_texts)
        sentiment_specific = self.word_analyzer.get_sentiment_specific_words(
            positive_word_freq, 
            negative_word_freq, 
            top_n=20
        )
        
        # Step 3: Theme Extraction
        print("\n[3/4] Extracting themes...")
        self.df = self.theme_extractor.extract_themes_from_reviews(self.df)
        theme_insights = self.theme_extractor.get_theme_insights(self.df)
        
        # Step 4: Generate Reports
        print("\n[4/4] Generating insights and visualizations...")
        
        # Store results
        self.results = {
            'sentiment_distribution': sentiment_counts.to_dict(),
            'top_positive_words': top_positive_words,
            'top_negative_words': top_negative_words,
            'positive_specific_words': sentiment_specific['positive_specific'],
            'negative_specific_words': sentiment_specific['negative_specific'],
            'theme_insights': theme_insights
        }
        
        return self.results
    
    def print_summary(self):
        """Print summary of analysis results"""
        if not hasattr(self, 'results'):
            print("No analysis results available. Run run_full_analysis() first.")
            return
        
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        # Sentiment Summary
        print("\n📊 SENTIMENT DISTRIBUTION:")
        for sentiment, count in self.results['sentiment_distribution'].items():
            print(f"   {sentiment.capitalize()}: {count}")
        
        # Top Words
        print("\n📝 TOP WORDS IN POSITIVE REVIEWS:")
        for i, (word, freq) in enumerate(self.results['top_positive_words'][:15], 1):
            print(f"   {i:2d}. {word:15s} ({freq:4d} mentions)")
        
        print("\n📝 TOP WORDS IN NEGATIVE REVIEWS:")
        for i, (word, freq) in enumerate(self.results['top_negative_words'][:15], 1):
            print(f"   {i:2d}. {word:15s} ({freq:4d} mentions)")
        
        # Sentiment-Specific Words
        print("\n✅ WORDS MORE COMMON IN POSITIVE REVIEWS:")
        for i, (word, data) in enumerate(self.results['positive_specific_words'][:10], 1):
            ratio = data['ratio']
            print(f"   {i:2d}. {word:15s} (ratio: {ratio:.2f}x)")
        
        print("\n❌ WORDS MORE COMMON IN NEGATIVE REVIEWS:")
        for i, (word, data) in enumerate(self.results['negative_specific_words'][:10], 1):
            ratio = data['ratio']
            print(f"   {i:2d}. {word:15s} (ratio: {ratio:.2f}x)")
        
        # Theme Insights
        print("\n🎯 THEME INSIGHTS:")
        for theme, insight in self.results['theme_insights'].items():
            pos_pct = insight['positive_mention_rate']
            neg_pct = insight['negative_mention_rate']
            diff = insight['difference']
            direction = "more in positive" if insight['more_in_positive'] else "more in negative"
            
            print(f"\n   {theme.upper().replace('_', ' ')}:")
            print(f"      Positive reviews: {pos_pct:.1f}% mention this theme")
            print(f"      Negative reviews: {neg_pct:.1f}% mention this theme")
            print(f"      Difference: {diff:+.1f}% ({direction})")
    
    def create_visualizations(self, output_dir='output'):
        """Create visualization plots"""
        if not hasattr(self, 'results'):
            print("No analysis results available. Run run_full_analysis() first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Sentiment Distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        sentiment_counts = pd.Series(self.results['sentiment_distribution'])
        sentiment_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c', '#95a5a6'])
        ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Number of Reviews', fontsize=12)
        ax.tick_params(axis='x', rotation=0)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved sentiment distribution plot")
        
        # 2. Top Words Comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Positive words
        pos_words = dict(self.results['top_positive_words'][:15])
        ax1.barh(range(len(pos_words)), list(pos_words.values()), color='#2ecc71')
        ax1.set_yticks(range(len(pos_words)))
        ax1.set_yticklabels(list(pos_words.keys()))
        ax1.set_xlabel('Frequency', fontsize=12)
        ax1.set_title('Top 15 Words - Positive Reviews', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # Negative words
        neg_words = dict(self.results['top_negative_words'][:15])
        ax2.barh(range(len(neg_words)), list(neg_words.values()), color='#e74c3c')
        ax2.set_yticks(range(len(neg_words)))
        ax2.set_yticklabels(list(neg_words.keys()))
        ax2.set_xlabel('Frequency', fontsize=12)
        ax2.set_title('Top 15 Words - Negative Reviews', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_words_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved top words comparison plot")
        
        # 3. Theme Analysis
        themes = list(self.results['theme_insights'].keys())
        pos_rates = [self.results['theme_insights'][t]['positive_mention_rate'] for t in themes]
        neg_rates = [self.results['theme_insights'][t]['negative_mention_rate'] for t in themes]
        
        x = np.arange(len(themes))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, pos_rates, width, label='Positive Reviews', color='#2ecc71')
        ax.bar(x + width/2, neg_rates, width, label='Negative Reviews', color='#e74c3c')
        
        ax.set_xlabel('Theme', fontsize=12)
        ax.set_ylabel('Mention Rate (%)', fontsize=12)
        ax.set_title('Theme Mentions by Sentiment', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in themes], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/theme_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved theme analysis plot")
        
        # 4. Word Clouds
        if len(self.df) > 0:
            # Positive word cloud
            positive_text = ' '.join(
                self.df[self.df['sentiment'] == 'positive']['cleaned_text'].astype(str)
            )
            if positive_text.strip():
                wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
                plt.figure(figsize=(12, 6))
                plt.imshow(wordcloud_pos, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud - Positive Reviews', fontsize=16, fontweight='bold', pad=20)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/wordcloud_positive.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved positive word cloud")
            
            # Negative word cloud
            negative_text = ' '.join(
                self.df[self.df['sentiment'] == 'negative']['cleaned_text'].astype(str)
            )
            if negative_text.strip():
                wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
                plt.figure(figsize=(12, 6))
                plt.imshow(wordcloud_neg, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud - Negative Reviews', fontsize=16, fontweight='bold', pad=20)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/wordcloud_negative.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved negative word cloud")
        
        print(f"\n✓ All visualizations saved to '{output_dir}/' directory")


def main():
    """Main function to run the analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Yelp reviews')
    parser.add_argument('--data', type=str, required=True, help='Path to Yelp review data file')
    parser.add_argument('--format', type=str, default='json', choices=['json', 'csv'], 
                       help='Data file format (json or csv)')
    parser.add_argument('--sample', type=int, default=None, 
                       help='Sample size (optional, for faster testing)')
    parser.add_argument('--text-column', type=str, default='text', 
                       help='Name of column containing review text')
    parser.add_argument('--output', type=str, default='output', 
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = YelpReviewAnalyzer()
    
    # Load data
    text_col = analyzer.load_data(args.data, format=args.format)
    if args.text_column != 'text':
        text_col = args.text_column
    
    # Run analysis
    results = analyzer.run_full_analysis(text_column=text_col, sample_size=args.sample)
    
    # Print summary
    analyzer.print_summary()
    
    # Create visualizations
    analyzer.create_visualizations(output_dir=args.output)
    
    # Save results to JSON
    results_file = f'{args.output}/analysis_results.json'
    # Convert results to JSON-serializable format
    json_results = {
        'sentiment_distribution': analyzer.results['sentiment_distribution'],
        'top_positive_words': analyzer.results['top_positive_words'],
        'top_negative_words': analyzer.results['top_negative_words'],
        'positive_specific_words': {
            word: {
                'positive_freq': data['positive_freq'],
                'negative_freq': data['negative_freq'],
                'ratio': data['ratio']
            }
            for word, data in analyzer.results['positive_specific_words']
        },
        'negative_specific_words': {
            word: {
                'positive_freq': data['positive_freq'],
                'negative_freq': data['negative_freq'],
                'ratio': data['ratio']
            }
            for word, data in analyzer.results['negative_specific_words']
        },
        'theme_insights': {
            theme: {
                'positive_mention_rate': insight['positive_mention_rate'],
                'negative_mention_rate': insight['negative_mention_rate'],
                'difference': insight['difference'],
                'more_in_positive': insight['more_in_positive'],
                'avg_mentions_positive': float(insight['avg_mentions_positive']),
                'avg_mentions_negative': float(insight['avg_mentions_negative'])
            }
            for theme, insight in analyzer.results['theme_insights'].items()
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to '{results_file}'")
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

