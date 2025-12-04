#!/usr/bin/env python3
"""
Simple caching system for Yelp chart data
Creates a preprocessed cache file for instant loading
"""

import pickle
import json
import pandas as pd
from pathlib import Path
from yelp_chart_generator import YelpChartGenerator

def create_fast_cache(sample_size=1000):
    """Create a cache file with preprocessed data for fast loading."""
    print(f"🚀 Creating fast cache with {sample_size} reviews...")
    
    # Create cache directory
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Generate charts with specified sample size
    generator = YelpChartGenerator(sample_size=sample_size)
    charts = generator.generate_all_charts()
    
    if charts:
        # Save to cache file
        cache_file = cache_dir / f"fast_cache_{sample_size}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(charts, f)
        
        print(f"[OK] Cache created: {cache_file}")
        print(f"[*] Cached {charts['stats']['total_reviews']} reviews")
        print(f"[*] Cache file size: {cache_file.stat().st_size / 1024:.1f} KB")
        return True
    else:
        print("❌ Failed to generate charts for cache")
        return False

def load_fast_cache(sample_size=1000):
    """Load charts from cache if available."""
    cache_file = Path("cache") / f"fast_cache_{sample_size}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                charts = pickle.load(f)
            print(f"[*] Loaded from cache: {charts['stats']['total_reviews']} reviews")
            return charts
        except Exception as e:
            print(f"❌ Cache loading failed: {e}")
    
    return None

if __name__ == "__main__":
    # Create cache with different sample sizes
    sizes = [500, 1000, 2000]
    
    for size in sizes:
        print(f"\n{'='*50}")
        create_fast_cache(size)
    
    print(f"\n{'='*50}")
    print("🎉 Fast cache creation complete!")
    print("[TIP] Your Flask app will now load much faster!")
