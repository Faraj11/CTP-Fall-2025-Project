from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template, request
from PIL import Image
import io

# Import Yelp chart generator
try:
    from yelp_chart_generator import YelpChartGenerator
    YELP_CHARTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Yelp chart generator not available: {e}")
    YELP_CHARTS_AVAILABLE = False

# Import image captioner
try:
    from image_captioner import ImageCaptioner
    IMAGE_CAPTIONER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Image captioner not available: {e}")
    IMAGE_CAPTIONER_AVAILABLE = False

# CSV file path - update this to point to your merged restaurants CSV file
# The CSV should be created by running merge_restaurants.py
CSV_PATH = Path(__file__).parent / "nyc_restaurants_merged.csv"

app = Flask(__name__)

# Load restaurants data once at startup
restaurants_df = None
address_lookup = None

# Global cache for USA Yelp charts (loaded at startup for fast access)
usa_charts_cache = None
usa_charts_json = None  # Pre-serialized JSON string for ultra-fast responses

def load_usa_charts_cache():
    """Preload USA Yelp charts at startup for fast access."""
    global usa_charts_cache, usa_charts_json
    if not YELP_CHARTS_AVAILABLE:
        return
    
    try:
        import pickle
        import json
        import time
        from flask import Response
        start_time = time.time()
        
        # Try to load directly from cache file first (fastest)
        cache_file = Path(__file__).parent / "cache" / "yelp_charts_cache.pkl"
        if cache_file.exists():
            print("[*] Loading USA charts from cache file...")
            with open(cache_file, 'rb') as f:
                usa_charts_cache = pickle.load(f)
            
            # Pre-serialize Plotly charts to JSON strings for faster API responses
            if usa_charts_cache and isinstance(usa_charts_cache, dict):
                for key in ['sentiment_chart', 'word_frequency_chart', 'theme_chart']:
                    if key in usa_charts_cache:
                        # Already a JSON string, skip
                        if isinstance(usa_charts_cache[key], str):
                            continue
                        # Convert Plotly figure to JSON string if it's a figure object
                        if hasattr(usa_charts_cache[key], 'to_json'):
                            usa_charts_cache[key] = usa_charts_cache[key].to_json()
            
            # Pre-serialize entire response to JSON string for ultra-fast API responses
            serialize_start = time.time()
            usa_charts_json = json.dumps(usa_charts_cache)
            serialize_time = time.time() - serialize_start
            
            elapsed = time.time() - start_time
            print(f"[OK] USA charts cache loaded and pre-serialized in {elapsed:.2f}s (JSON: {serialize_time:.2f}s)")
            return
        
        # Fallback: try fast_cache file
        fast_cache = Path(__file__).parent / "cache" / "fast_cache_5000.pkl"
        if fast_cache.exists():
            print("[*] Loading USA charts from fast cache file...")
            with open(fast_cache, 'rb') as f:
                usa_charts_cache = pickle.load(f)
            
            # Pre-serialize Plotly charts to JSON strings for faster API responses
            if usa_charts_cache and isinstance(usa_charts_cache, dict):
                for key in ['sentiment_chart', 'word_frequency_chart', 'theme_chart']:
                    if key in usa_charts_cache:
                        # Already a JSON string, skip
                        if isinstance(usa_charts_cache[key], str):
                            continue
                        # Convert Plotly figure to JSON string if it's a figure object
                        if hasattr(usa_charts_cache[key], 'to_json'):
                            usa_charts_cache[key] = usa_charts_cache[key].to_json()
            
            # Pre-serialize entire response to JSON string
            serialize_start = time.time()
            usa_charts_json = json.dumps(usa_charts_cache)
            serialize_time = time.time() - serialize_start
            
            elapsed = time.time() - start_time
            print(f"[OK] USA charts cache loaded and pre-serialized in {elapsed:.2f}s (JSON: {serialize_time:.2f}s)")
            return
        
        # Last resort: generate charts (slow)
        print("[*] No cache found, generating charts (this may take a while)...")
        generator = YelpChartGenerator()
        usa_charts_cache = generator.generate_all_charts()
        if usa_charts_cache:
            # Pre-serialize Plotly charts to JSON strings
            if isinstance(usa_charts_cache, dict):
                for key in ['sentiment_chart', 'word_frequency_chart', 'theme_chart']:
                    if key in usa_charts_cache:
                        # Already a JSON string, skip
                        if isinstance(usa_charts_cache[key], str):
                            continue
                        # Convert Plotly figure to JSON string if it's a figure object
                        if hasattr(usa_charts_cache[key], 'to_json'):
                            usa_charts_cache[key] = usa_charts_cache[key].to_json()
            
            # Pre-serialize entire response to JSON string
            serialize_start = time.time()
            usa_charts_json = json.dumps(usa_charts_cache)
            serialize_time = time.time() - serialize_start
            
            elapsed = time.time() - start_time
            print(f"[OK] USA charts generated and pre-serialized in {elapsed:.2f}s (JSON: {serialize_time:.2f}s)")
        else:
            print("[WARNING] Failed to generate USA charts cache")
    except Exception as e:
        import traceback
        print(f"[WARNING] Error loading USA charts cache: {e}")
        traceback.print_exc()
        usa_charts_cache = None
        usa_charts_json = None


def normalize_name_for_matching(name: str) -> str:
    """Normalize restaurant name for matching."""
    if pd.isna(name) or name == "":
        return ""
    return str(name).lower().strip().replace("'", "").replace("&", "and").replace("-", " ")


def infer_cuisine_from_name(name: str) -> str:
    """Infer cuisine type from restaurant name using keyword matching."""
    if pd.isna(name) or name == "":
        return None
        
    name_lower = str(name).lower()
    
    # Refined cuisine keywords with better specificity
    cuisine_keywords = {
        'Pizza': ['pizzeria', 'pizza'],
        'Steakhouse': ['steakhouse', 'steak house', 'prime rib', 'chophouse'],
        'Bakery': ['bakery', 'bread', 'bagel', 'donut', 'pastry', 'cake shop'],
        'Coffee': ['coffee shop', 'cafe', 'espresso bar', 'roastery'],
        'Deli': ['deli', 'delicatessen', 'sandwich shop'],
        'Burger': ['burger', 'hamburger', 'burger joint'],
        'BBQ': ['bbq', 'barbecue', 'smokehouse', 'ribs'],
        'Seafood': ['seafood', 'lobster', 'crab', 'oyster', 'clam', 'shrimp', 'fish market'],
        'Caribbean': ['caribbean', 'jamaican', 'jerk', 'roti', 'curry goat', 'plantain'],
        'Japanese': ['japanese', 'sushi', 'ramen', 'tokyo', 'sakura', 'hibachi', 'teriyaki', 'yakitori'],
        'Korean': ['korean', 'korea', 'kimchi', 'seoul', 'kbbq'],
        'Thai': ['thai', 'thailand', 'pad thai', 'bangkok', 'tom yum'],
        'Indian': ['indian', 'curry', 'tandoor', 'masala', 'delhi', 'mumbai', 'bengal', 'biryani'],
        'Greek': ['greek', 'gyro', 'athens', 'santorini', 'mediterranean'],
        'French': ['french', 'bistro', 'brasserie', 'patisserie', 'crepe', 'paris', 'provence'],
        'Mexican': ['mexican', 'taco', 'burrito', 'cantina', 'salsa', 'tortilla', 'quesadilla', 'tex-mex'],
        'Chinese': ['chinese', 'china town', 'wok', 'dim sum', 'szechuan', 'hunan', 'cantonese'],
        'Italian': ['trattoria', 'ristorante', 'pasta', 'gelato', 'espresso cafe', 'italian']
    }
    
    # Priority order for overlapping keywords (more specific first)
    priority_order = ['Pizza', 'Steakhouse', 'Bakery', 'Coffee', 'Deli', 'Burger', 'BBQ', 'Seafood', 
                     'Caribbean', 'Japanese', 'Korean', 'Thai', 'Indian', 'Greek', 'French', 
                     'Mexican', 'Chinese', 'Italian']
    
    # Check cuisines in priority order
    for cuisine in priority_order:
        if cuisine in cuisine_keywords:
            for keyword in cuisine_keywords[cuisine]:
                if keyword in name_lower:
                    return cuisine
    
    return None


def load_restaurants() -> pd.DataFrame:
    """Load restaurants from merged CSV file."""
    global restaurants_df, address_lookup
    if restaurants_df is None:
        if not CSV_PATH.exists():
            raise RuntimeError(f"CSV file {CSV_PATH} not found. Run merge_restaurants.py first.")
        restaurants_df = pd.read_csv(CSV_PATH)
        # Clean up data: handle NaN values
        restaurants_df = restaurants_df.fillna("")
        # Convert numeric columns
        numeric_cols = ["overall_rating", "reviews", "food", "service", "ambiance"]
        for col in numeric_cols:
            if col in restaurants_df.columns:
                restaurants_df[col] = pd.to_numeric(restaurants_df[col], errors="coerce").fillna(0)
        
        # Create address lookup from original nyc_restaurants.csv to fill missing addresses
        if address_lookup is None:
            try:
                # Try to find nyc_restaurants.csv in the same directory as the merged CSV
                nyc_path = CSV_PATH.parent / "nyc_restaurants.csv"
                if not nyc_path.exists():
                    # Fallback: try current directory
                    nyc_path = Path(__file__).parent / "nyc_restaurants.csv"
                if nyc_path.exists():
                    nyc_df = pd.read_csv(nyc_path)
                    address_lookup = {}
                    for _, row in nyc_df.iterrows():
                        name = row.get("Name", "")
                        address = row.get("Address", "")
                        if name and address and pd.notna(address) and str(address).strip() != "":
                            normalized = normalize_name_for_matching(name)
                            if normalized:
                                address_lookup[normalized] = str(address).strip()
            except Exception as e:
                print(f"Warning: Could not load address lookup: {e}")
                address_lookup = {}
    return restaurants_df


# Cuisine keyword mappings for better matching
CUISINE_KEYWORDS = {
    "asian": ["asian", "chinese", "japanese", "korean", "thai", "vietnamese", "indian", "sushi", "ramen", "pho", "curry"],
    "italian": ["italian", "pasta", "pizza", "trattoria", "ristorante", "gelato"],
    "mexican": ["mexican", "taco", "burrito", "quesadilla", "enchilada", "margarita"],
    "american": ["american", "burger", "steak", "bbq", "barbecue", "grill"],
    "french": ["french", "bistro", "brasserie", "croissant", "crepe"],
    "mediterranean": ["mediterranean", "greek", "turkish", "lebanese", "middle eastern"],
    "seafood": ["seafood", "fish", "oyster", "lobster", "sushi"],
    "japanese": ["japanese", "sushi", "sashimi", "ramen", "izakaya", "yakitori"],
    "indian": ["indian", "curry", "tandoori", "naan", "biryani"],
    "thai": ["thai", "pad thai", "tom yum", "green curry"],
    "korean": ["korean", "kbbq", "kimchi", "bulgogi"],
    "chinese": ["chinese", "dim sum", "szechuan", "cantonese", "dumpling"],
    "spanish": ["spanish", "tapas", "paella", "sangria"],
    "latin": ["latin", "cuban", "puerto rican", "dominican", "brazilian"],
    "halal": ["halal", "halaal", "muslim", "islamic"],
    "kosher": ["kosher", "jewish", "orthodox"],
    "caribbean": ["caribbean", "jamaican", "jerk", "roti", "plantain", "curry goat"],
    "african": ["african", "ethiopian", "moroccan", "nigerian", "senegalese"],
    "pizza": ["pizza", "pizzeria", "pie"],
    "bakery": ["bakery", "bread", "bagel", "donut", "pastry", "cake"],
    "coffee": ["coffee", "cafe", "espresso", "latte", "cappuccino"],
    "deli": ["deli", "delicatessen", "sandwich"],
    "steakhouse": ["steakhouse", "steak house", "prime rib", "chophouse"],
    "vegetarian": ["vegetarian", "vegan", "plant-based", "veggie"],
    "fusion": ["fusion", "contemporary", "modern", "eclectic"],
}

# NYC borough names (exact matches only)
BOROUGH_NAMES = ["manhattan", "brooklyn", "queens", "bronx", "staten island"]

# NYC location keywords - maps neighborhoods to their boroughs
# Note: The first item in each list is the borough name itself
LOCATION_KEYWORDS = {
    "manhattan": ["midtown", "downtown", "uptown", "upper east side", "upper west side", "east village", "west village", "soho", "noho", "tribeca", "chelsea", "gramercy", "flatiron", "hell's kitchen", "financial district", "battery park", "greenwich village", "lower east side", "upper manhattan"],
    "queens": ["astoria", "flushing", "long island city", "lic", "jamaica", "forest hills", "elmhurst"],
    "brooklyn": ["williamsburg", "dumbo", "park slope", "red hook", "brooklyn heights", "downtown brooklyn", "prospect heights", "crown heights"],
    "bronx": ["south bronx", "yankee stadium", "fordham"],
    "staten island": ["st. george", "stapleton"],
}

def normalize_price_category(price_cat_val, price_per_person_val):
    """Normalize price to dollar sign format: $, $$, $$$, $$$$"""
    # Try price_category first (numeric: 1, 2, 3, 4)
    if price_cat_val and pd.notna(price_cat_val):
        try:
            price_num = float(price_cat_val)
            if 1 <= price_num <= 4:
                return "$" * int(price_num)
        except (ValueError, TypeError):
            pass
    
    # Try price_per_person (text format)
    if price_per_person_val and pd.notna(price_per_person_val):
        price_str = str(price_per_person_val).strip().lower()
        if "$30 and under" in price_str or "under $30" in price_str or "under 30" in price_str:
            return "$$"
        elif "$31 to $50" in price_str or "$31-$50" in price_str or "31 to 50" in price_str:
            return "$$$"
        elif "$50 and over" in price_str or "over $50" in price_str or "over 50" in price_str:
            return "$$$$"
        # Try to extract numeric value from price_per_person
        try:
            # Look for dollar amounts
            amounts = re.findall(r'\$?(\d+)', price_str)
            if amounts:
                max_amount = max(int(a) for a in amounts)
                if max_amount <= 30:
                    return "$$"
                elif max_amount <= 50:
                    return "$$$"
                else:
                    return "$$$$"
        except:
            pass
    
    return None

def extract_query_keywords(query: str) -> Tuple[List[str], List[str]]:
    """Extract cuisine and location keywords from query."""
    query_lower = query.lower()
    words = re.findall(r'\b\w+\b', query_lower)
    
    cuisine_keywords = []
    location_keywords = []
    
    # Remove common stop words that don't help with matching
    stop_words = {"food", "restaurant", "restaurants", "in", "near", "at", "the", "a", "an", "and", "or", "for", "with"}
    meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Check for cuisine matches (check all keywords, not just first match)
    for cuisine_type, keywords in CUISINE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                if cuisine_type not in cuisine_keywords:
                    cuisine_keywords.append(cuisine_type)
                # Also add the specific keyword for better matching
                if keyword not in cuisine_keywords:
                    cuisine_keywords.append(keyword)
    
    # Check for location matches (check all keywords)
    for location, keywords in LOCATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                if location not in location_keywords:
                    location_keywords.append(location)
                # Also add the specific keyword
                if keyword not in location_keywords:
                    location_keywords.append(keyword)
    
    # If no specific keywords found, use meaningful words as fallback
    if not cuisine_keywords:
        cuisine_keywords = meaningful_words
    if not location_keywords:
        location_keywords = meaningful_words
    
    return cuisine_keywords, location_keywords


def calculate_match_score(restaurant: Dict, query: str) -> float:
    """Calculate a match score optimized for general discovery queries (cuisine + location)."""
    name = str(restaurant.get("name", "")).lower()
    address = str(restaurant.get("address", "")).lower()
    cuisine = str(restaurant.get("cuisine", "")).lower()
    locality = str(restaurant.get("locality", "")).lower()
    borough = str(restaurant.get("borough", "")).lower()
    neighborhood = str(restaurant.get("neighborhood", "")).lower()
    query_lower = query.lower()
    
    # Extract keywords from query
    cuisine_keywords, location_keywords = extract_query_keywords(query)
    
    # CUISINE MATCHING (35% weight) - Most important for discovery
    cuisine_score = 0.0
    if cuisine:
        # Exact cuisine match
        for keyword in cuisine_keywords:
            if keyword in cuisine:
                cuisine_score = max(cuisine_score, 1.0)
                break
        
        # Partial match or similarity
        if cuisine_score == 0:
            for keyword in cuisine_keywords:
                if keyword in cuisine or cuisine in keyword:
                    cuisine_score = max(cuisine_score, 0.7)
                else:
                    # Use similarity for close matches
                    similarity = difflib.SequenceMatcher(None, cuisine, keyword).ratio()
                    if similarity > 0.6:
                        cuisine_score = max(cuisine_score, similarity * 0.8)
        
        # Also check if query words appear in cuisine
        query_words = re.findall(r'\b\w+\b', query_lower)
        for word in query_words:
            if len(word) > 3 and word in cuisine:  # Only meaningful words
                cuisine_score = max(cuisine_score, 0.5)
    
    # LOCATION MATCHING (30% weight) - Second most important
    location_score = 0.0
    
    # Check borough first (highest priority)
    if borough:
        for keyword in location_keywords:
            if keyword in borough or borough in keyword:
                location_score = max(location_score, 1.0)
                break
    
    # Check neighborhood/locality
    location_text = neighborhood if neighborhood else locality
    if location_text and location_score < 1.0:
        for keyword in location_keywords:
            if keyword in location_text or location_text in keyword:
                location_score = max(location_score, 0.9)
                break
        
        # Partial match
        if location_score < 0.9:
            for keyword in location_keywords:
                similarity = difflib.SequenceMatcher(None, location_text, keyword).ratio()
                if similarity > 0.5:
                    location_score = max(location_score, similarity * 0.8)
    
    # Check address as fallback
    if location_score < 0.5 and address:
        for keyword in location_keywords:
            if keyword in address:
                location_score = max(location_score, 0.6)
                break
    
    # NAME MATCHING (25% weight) - Increased importance for direct name searches
    name_score = 0.0
    if query_lower in name:
        name_score = 1.0
    else:
        # Check if any query words are in name (more flexible)
        query_words = re.findall(r'\b\w+\b', query_lower)
        for word in query_words:
            if len(word) > 2 and word in name:  # Reduced minimum length from 3 to 2
                name_score = max(name_score, 0.8)
        
        # Check partial matches (word boundaries)
        for word in query_words:
            if len(word) > 2:
                # Check if word appears at word boundaries in name
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, name, re.IGNORECASE):
                    name_score = max(name_score, 0.9)
        
        # Similarity as fallback (increased weight)
        if name_score == 0:
            similarity = difflib.SequenceMatcher(None, name, query_lower).ratio()
            if similarity > 0.3:  # Only use if reasonably similar
                name_score = similarity * 0.6
    
    # RATING BOOST (10% weight) - Quality indicator
    rating = float(restaurant.get("overall_rating", 0) or 0)
    rating_boost = (rating / 5.0) if rating > 0 else 0
    
    # REVIEW COUNT BOOST (5% weight) - Popularity indicator
    review_count = float(restaurant.get("reviews", 0) or 0)
    review_boost = min(review_count / 50000.0, 1.0) if review_count > 0 else 0
    
    # Combined score optimized for discovery queries
    # Cuisine (40%), Location (35%), Name (10%), Rating (10%), Reviews (5%)
    score = (
        cuisine_score * 0.35 +
        location_score * 0.30 +
        name_score * 0.25 +
        rating_boost * 0.07 +
        review_boost * 0.03
    )
    
    return score


def interpret_caption_for_cuisine(caption: str) -> Dict[str, any]:
    """
    Interpret the full caption to determine the primary cuisine/food type.
    Returns a dictionary with the primary cuisine, food items, and confidence.
    
    This function analyzes the caption as a whole to make a holistic determination
    of what cuisine/food type the image represents.
    """
    if not caption or caption.strip() == "":
        return {"primary_cuisine": None, "food_items": [], "confidence": 0.0}
    
    caption_lower = caption.lower()
    
    # Comprehensive food-to-cuisine mapping with priority scoring
    # Higher priority items are more specific indicators
    food_cuisine_map = {
        # Italian (high priority - specific dishes)
        'spaghetti': {'cuisine': 'italian', 'priority': 10, 'food': 'spaghetti'},
        'meatballs': {'cuisine': 'italian', 'priority': 9, 'food': 'meatballs'},
        'spaghetti and meatballs': {'cuisine': 'italian', 'priority': 15, 'food': 'spaghetti and meatballs'},
        'pizza': {'cuisine': 'italian', 'priority': 10, 'food': 'pizza'},
        'pasta': {'cuisine': 'italian', 'priority': 8, 'food': 'pasta'},
        'lasagna': {'cuisine': 'italian', 'priority': 9, 'food': 'lasagna'},
        'ravioli': {'cuisine': 'italian', 'priority': 9, 'food': 'ravioli'},
        'fettuccine': {'cuisine': 'italian', 'priority': 9, 'food': 'fettuccine'},
        'penne': {'cuisine': 'italian', 'priority': 8, 'food': 'penne'},
        'marinara': {'cuisine': 'italian', 'priority': 7, 'food': 'marinara sauce'},
        'parmesan': {'cuisine': 'italian', 'priority': 6, 'food': 'parmesan'},
        'mozzarella': {'cuisine': 'italian', 'priority': 6, 'food': 'mozzarella'},
        'risotto': {'cuisine': 'italian', 'priority': 9, 'food': 'risotto'},
        'pizzeria': {'cuisine': 'italian', 'priority': 8, 'food': 'pizza'},
        
        # American
        'burger': {'cuisine': 'american', 'priority': 10, 'food': 'burger'},
        'hamburger': {'cuisine': 'american', 'priority': 10, 'food': 'burger'},
        'beef burger': {'cuisine': 'american', 'priority': 11, 'food': 'burger'},
        'french fries': {'cuisine': 'american', 'priority': 7, 'food': 'french fries'},
        'fries': {'cuisine': 'american', 'priority': 6, 'food': 'french fries'},
        'steak': {'cuisine': 'steakhouse', 'priority': 10, 'food': 'steak'},
        'bbq': {'cuisine': 'american', 'priority': 9, 'food': 'bbq'},
        'barbecue': {'cuisine': 'american', 'priority': 9, 'food': 'barbecue'},
        'ribs': {'cuisine': 'american', 'priority': 8, 'food': 'ribs'},
        'fried chicken': {'cuisine': 'american', 'priority': 9, 'food': 'fried chicken'},
        'wings': {'cuisine': 'american', 'priority': 7, 'food': 'wings'},
        
        # Japanese
        'sushi': {'cuisine': 'japanese', 'priority': 10, 'food': 'sushi'},
        'sashimi': {'cuisine': 'japanese', 'priority': 10, 'food': 'sashimi'},
        'ramen': {'cuisine': 'japanese', 'priority': 10, 'food': 'ramen'},
        'tempura': {'cuisine': 'japanese', 'priority': 9, 'food': 'tempura'},
        'teriyaki': {'cuisine': 'japanese', 'priority': 9, 'food': 'teriyaki'},
        'miso': {'cuisine': 'japanese', 'priority': 7, 'food': 'miso'},
        'udon': {'cuisine': 'japanese', 'priority': 9, 'food': 'udon'},
        'yakitori': {'cuisine': 'japanese', 'priority': 9, 'food': 'yakitori'},
        
        # Mexican
        'taco': {'cuisine': 'mexican', 'priority': 10, 'food': 'taco'},
        'burrito': {'cuisine': 'mexican', 'priority': 10, 'food': 'burrito'},
        'quesadilla': {'cuisine': 'mexican', 'priority': 9, 'food': 'quesadilla'},
        'enchilada': {'cuisine': 'mexican', 'priority': 9, 'food': 'enchilada'},
        'salsa': {'cuisine': 'mexican', 'priority': 6, 'food': 'salsa'},
        'guacamole': {'cuisine': 'mexican', 'priority': 8, 'food': 'guacamole'},
        'nachos': {'cuisine': 'mexican', 'priority': 8, 'food': 'nachos'},
        
        # Indian
        'curry': {'cuisine': 'indian', 'priority': 9, 'food': 'curry'},
        'tandoori': {'cuisine': 'indian', 'priority': 9, 'food': 'tandoori'},
        'naan': {'cuisine': 'indian', 'priority': 8, 'food': 'naan'},
        'biryani': {'cuisine': 'indian', 'priority': 10, 'food': 'biryani'},
        'masala': {'cuisine': 'indian', 'priority': 8, 'food': 'masala'},
        'samosa': {'cuisine': 'indian', 'priority': 9, 'food': 'samosa'},
        'dal': {'cuisine': 'indian', 'priority': 7, 'food': 'dal'},
        
        # Thai
        'pad thai': {'cuisine': 'thai', 'priority': 12, 'food': 'pad thai'},
        'tom yum': {'cuisine': 'thai', 'priority': 10, 'food': 'tom yum'},
        'green curry': {'cuisine': 'thai', 'priority': 9, 'food': 'green curry'},
        'red curry': {'cuisine': 'thai', 'priority': 9, 'food': 'red curry'},
        'thai curry': {'cuisine': 'thai', 'priority': 10, 'food': 'thai curry'},
        
        # Chinese
        'dumpling': {'cuisine': 'chinese', 'priority': 9, 'food': 'dumpling'},
        'dim sum': {'cuisine': 'chinese', 'priority': 10, 'food': 'dim sum'},
        'kung pao': {'cuisine': 'chinese', 'priority': 9, 'food': 'kung pao'},
        'szechuan': {'cuisine': 'chinese', 'priority': 8, 'food': 'szechuan'},
        'lo mein': {'cuisine': 'chinese', 'priority': 9, 'food': 'lo mein'},
        'fried rice': {'cuisine': 'chinese', 'priority': 8, 'food': 'fried rice'},
        'wonton': {'cuisine': 'chinese', 'priority': 8, 'food': 'wonton'},
        
        # Vietnamese
        'pho': {'cuisine': 'vietnamese', 'priority': 12, 'food': 'pho'},
        'banh mi': {'cuisine': 'vietnamese', 'priority': 11, 'food': 'banh mi'},
        'spring roll': {'cuisine': 'vietnamese', 'priority': 8, 'food': 'spring roll'},
        
        # Spanish
        'paella': {'cuisine': 'spanish', 'priority': 12, 'food': 'paella'},
        'tapas': {'cuisine': 'spanish', 'priority': 10, 'food': 'tapas'},
        'sangria': {'cuisine': 'spanish', 'priority': 6, 'food': 'sangria'},
        
        # Greek/Mediterranean
        'gyro': {'cuisine': 'greek', 'priority': 11, 'food': 'gyro'},
        'kebab': {'cuisine': 'mediterranean', 'priority': 9, 'food': 'kebab'},
        'hummus': {'cuisine': 'mediterranean', 'priority': 8, 'food': 'hummus'},
        'falafel': {'cuisine': 'mediterranean', 'priority': 9, 'food': 'falafel'},
        
        # Deli
        'bagel': {'cuisine': 'deli', 'priority': 9, 'food': 'bagel'},
        'sandwich': {'cuisine': 'deli', 'priority': 8, 'food': 'sandwich'},
        'pastrami': {'cuisine': 'deli', 'priority': 8, 'food': 'pastrami'},
        'reuben': {'cuisine': 'deli', 'priority': 9, 'food': 'reuben'},
        
        # Coffee
        'coffee': {'cuisine': 'coffee', 'priority': 7, 'food': 'coffee'},
        'latte': {'cuisine': 'coffee', 'priority': 8, 'food': 'latte'},
        'espresso': {'cuisine': 'coffee', 'priority': 8, 'food': 'espresso'},
        'cappuccino': {'cuisine': 'coffee', 'priority': 8, 'food': 'cappuccino'},
        
        # Bakery
        'pancake': {'cuisine': 'bakery', 'priority': 9, 'food': 'pancake'},
        'pancakes': {'cuisine': 'bakery', 'priority': 9, 'food': 'pancakes'},
        'blueberry pancakes': {'cuisine': 'bakery', 'priority': 11, 'food': 'blueberry pancakes'},
        'maple syrup': {'cuisine': 'bakery', 'priority': 6, 'food': 'maple syrup'},
        'cake': {'cuisine': 'bakery', 'priority': 8, 'food': 'cake'},
        'pastry': {'cuisine': 'bakery', 'priority': 7, 'food': 'pastry'},
        'bread': {'cuisine': 'bakery', 'priority': 6, 'food': 'bread'},
        'croissant': {'cuisine': 'bakery', 'priority': 9, 'food': 'croissant'},
        'donut': {'cuisine': 'bakery', 'priority': 8, 'food': 'donut'},
        'muffin': {'cuisine': 'bakery', 'priority': 7, 'food': 'muffin'},
        
        # Seafood
        'seafood': {'cuisine': 'seafood', 'priority': 8, 'food': 'seafood'},
        'fish': {'cuisine': 'seafood', 'priority': 7, 'food': 'fish'},
        'lobster': {'cuisine': 'seafood', 'priority': 10, 'food': 'lobster'},
        'oyster': {'cuisine': 'seafood', 'priority': 9, 'food': 'oyster'},
        'crab': {'cuisine': 'seafood', 'priority': 9, 'food': 'crab'},
        'shrimp': {'cuisine': 'seafood', 'priority': 8, 'food': 'shrimp'},
        'salmon': {'cuisine': 'seafood', 'priority': 9, 'food': 'salmon'},
        'tuna': {'cuisine': 'seafood', 'priority': 8, 'food': 'tuna'},
    }
    
    # Score each cuisine based on matches in caption
    cuisine_scores = {}
    food_items_found = []
    
    # Check for multi-word food items first (higher specificity)
    for food_item, info in sorted(food_cuisine_map.items(), key=lambda x: -len(x[0]) if ' ' in x[0] else 0):
        if food_item in caption_lower:
            cuisine = info['cuisine']
            priority = info['priority']
            food = info['food']
            
            if cuisine not in cuisine_scores:
                cuisine_scores[cuisine] = 0
            cuisine_scores[cuisine] += priority
            food_items_found.append(food)
    
    # Check for single-word items (lower priority if multi-word already found)
    if not food_items_found:
        for food_item, info in food_cuisine_map.items():
            if ' ' not in food_item:  # Single word items
                if food_item in caption_lower:
                    cuisine = info['cuisine']
                    priority = info['priority']
                    food = info['food']
                    
                    if cuisine not in cuisine_scores:
                        cuisine_scores[cuisine] = 0
                    cuisine_scores[cuisine] += priority
                    food_items_found.append(food)
    
    # Also check for direct cuisine mentions
    for cuisine_type in CUISINE_KEYWORDS.keys():
        if cuisine_type.lower() in caption_lower:
            if cuisine_type.lower() not in cuisine_scores:
                cuisine_scores[cuisine_type.lower()] = 0
            cuisine_scores[cuisine_type.lower()] += 5  # Moderate boost for direct mention
    
    # Determine primary cuisine
    primary_cuisine = None
    confidence = 0.0
    
    if cuisine_scores:
        # Get cuisine with highest score
        primary_cuisine = max(cuisine_scores.items(), key=lambda x: x[1])[0]
        max_score = cuisine_scores[primary_cuisine]
        total_score = sum(cuisine_scores.values())
        
        # Confidence based on score dominance
        if total_score > 0:
            confidence = min(max_score / total_score, 1.0) if total_score > 0 else 0.0
            # Boost confidence if we have specific food items
            if food_items_found:
                confidence = min(confidence + 0.2, 1.0)
    
    return {
        "primary_cuisine": primary_cuisine,
        "food_items": list(set(food_items_found)),  # Remove duplicates
        "confidence": confidence,
        "cuisine_scores": cuisine_scores
    }


def extract_food_keywords_from_caption(caption: str) -> List[str]:
    """Extract food and restaurant-related keywords from image caption.
    Enhanced to handle more detailed captions from improved BLIP model."""
    caption_lower = caption.lower()
    words = re.findall(r'\b\w+\b', caption_lower)
    
    # Food-related keywords that should be emphasized
    food_keywords = []
    
    # Expanded food items that map to cuisines (handles plurals and variations)
    food_mappings = {
        # Italian
        'pizza': 'italian', 'pizzas': 'italian', 'pizzeria': 'italian',
        'pasta': 'italian', 'pastas': 'italian', 'spaghetti': 'italian', 
        'spagetti': 'italian', 'meatball': 'italian', 'meatballs': 'italian',
        'lasagna': 'italian', 'lasagne': 'italian', 'ravioli': 'italian',
        'fettuccine': 'italian', 'penne': 'italian', 'marinara': 'italian',
        'parmesan': 'italian', 'mozzarella': 'italian', 'risotto': 'italian',
        # American
        'burger': 'american', 'burgers': 'american', 'hamburger': 'american',
        'hamburgers': 'american', 'steak': 'steakhouse', 'steaks': 'steakhouse',
        'bbq': 'american', 'barbecue': 'american', 'ribs': 'american',
        'fried chicken': 'american', 'wings': 'american',
        # Japanese
        'sushi': 'japanese', 'sashimi': 'japanese', 'ramen': 'japanese',
        'tempura': 'japanese', 'teriyaki': 'japanese', 'miso': 'japanese',
        'udon': 'japanese', 'yakitori': 'japanese',
        # Mexican
        'taco': 'mexican', 'tacos': 'mexican', 'burrito': 'mexican',
        'burritos': 'mexican', 'quesadilla': 'mexican', 'quesadillas': 'mexican',
        'enchilada': 'mexican', 'enchiladas': 'mexican', 'salsa': 'mexican',
        'guacamole': 'mexican', 'nachos': 'mexican',
        # Indian
        'curry': 'indian', 'curries': 'indian', 'tandoori': 'indian',
        'naan': 'indian', 'biryani': 'indian', 'masala': 'indian',
        'samosa': 'indian', 'dal': 'indian',
        # Thai
        'pad thai': 'thai', 'tom yum': 'thai', 'green curry': 'thai',
        'red curry': 'thai', 'thai curry': 'thai',
        # Chinese
        'dumpling': 'chinese', 'dumplings': 'chinese', 'dim sum': 'chinese',
        'kung pao': 'chinese', 'szechuan': 'chinese', 'lo mein': 'chinese',
        'fried rice': 'chinese', 'wonton': 'chinese',
        # Vietnamese
        'pho': 'vietnamese', 'banh mi': 'vietnamese', 'spring roll': 'vietnamese',
        # Spanish
        'paella': 'spanish', 'tapas': 'spanish', 'sangria': 'spanish',
        # Greek/Mediterranean
        'gyro': 'greek', 'gyros': 'greek', 'kebab': 'mediterranean',
        'kebabs': 'mediterranean', 'hummus': 'mediterranean', 'falafel': 'mediterranean',
        # Deli
        'bagel': 'deli', 'bagels': 'deli', 'sandwich': 'deli',
        'sandwiches': 'deli', 'pastrami': 'deli', 'reuben': 'deli',
        # Coffee
        'coffee': 'coffee', 'latte': 'coffee', 'espresso': 'coffee',
        'cappuccino': 'coffee', 'americano': 'coffee',
        # Bakery
        'cake': 'bakery', 'cakes': 'bakery', 'pastry': 'bakery',
        'pastries': 'bakery', 'bread': 'bakery', 'croissant': 'bakery',
        'donut': 'bakery', 'donuts': 'bakery', 'muffin': 'bakery',
        # Seafood
        'seafood': 'seafood', 'fish': 'seafood', 'lobster': 'seafood',
        'lobsters': 'seafood', 'oyster': 'seafood', 'oysters': 'seafood',
        'crab': 'seafood', 'crabs': 'seafood', 'shrimp': 'seafood',
        'salmon': 'seafood', 'tuna': 'seafood', 'sashimi': 'seafood',
    }
    
    # Extract food keywords from individual words
    for word in words:
        if len(word) > 2:  # Only meaningful words
            # Check direct mappings (handles plurals via word stem)
            word_singular = word.rstrip('s') if word.endswith('s') and len(word) > 3 else word
            if word in food_mappings:
                food_keywords.append(food_mappings[word])
                food_keywords.append(word)
            elif word_singular in food_mappings:
                food_keywords.append(food_mappings[word_singular])
                food_keywords.append(word_singular)
    
    # Check for multi-word food items (more important for detailed captions)
    for food_item, cuisine in food_mappings.items():
        if ' ' in food_item and food_item in caption_lower:
            food_keywords.append(cuisine)
            food_keywords.append(food_item)
        # Also check for partial matches of multi-word items
        elif ' ' in food_item:
            # Check if all words in the food item appear in caption
            food_words = food_item.split()
            if all(fw in caption_lower for fw in food_words):
                food_keywords.append(cuisine)
                food_keywords.append(food_item)
    
    # Extract cuisine mentions directly from caption
    cuisine_keywords_lower = {k.lower(): k for k in CUISINE_KEYWORDS.keys()}
    for word in words:
        if len(word) > 3:
            for cuisine_lower, cuisine in cuisine_keywords_lower.items():
                if cuisine_lower in word or word in cuisine_lower:
                    food_keywords.append(cuisine)
    
    return list(set(food_keywords))  # Remove duplicates


def calculate_image_match_score_with_interpretation(restaurant: Dict, caption: str, interpretation: Dict) -> float:
    """
    Calculate match score for image search using the caption interpretation.
    Prioritizes restaurants that match the interpreted primary cuisine/food type.
    """
    name = str(restaurant.get("name", "")).lower()
    cuisine = str(restaurant.get("cuisine", "")).lower()
    caption_lower = caption.lower()
    
    primary_cuisine = interpretation.get("primary_cuisine")
    food_items = interpretation.get("food_items", [])
    confidence = interpretation.get("confidence", 0.0)
    cuisine_scores = interpretation.get("cuisine_scores", {})
    
    # PRIMARY CUISINE MATCH (60% weight) - Most important
    # This is the key improvement: prioritize based on interpreted cuisine
    primary_cuisine_score = 0.0
    if primary_cuisine and cuisine:
        # Direct cuisine match
        if primary_cuisine in cuisine or cuisine in primary_cuisine:
            primary_cuisine_score = 1.0 * confidence  # Scale by confidence
        # Check cuisine keywords
        elif primary_cuisine in CUISINE_KEYWORDS:
            for keyword in CUISINE_KEYWORDS[primary_cuisine]:
                if keyword in cuisine:
                    primary_cuisine_score = 0.9 * confidence
                    break
        
        # Also check if restaurant cuisine matches any scored cuisines
        for scored_cuisine, score in cuisine_scores.items():
            if scored_cuisine in cuisine or cuisine in scored_cuisine:
                # Weight by the score relative to primary
                if scored_cuisine == primary_cuisine:
                    primary_cuisine_score = max(primary_cuisine_score, 1.0 * confidence)
                else:
                    # Secondary cuisine match (lower weight)
                    primary_cuisine_score = max(primary_cuisine_score, 0.7 * (score / max(cuisine_scores.values()) if cuisine_scores else 1.0))
    
    # FOOD ITEM MATCH (25% weight) - Match specific food items found
    food_item_score = 0.0
    if food_items:
        for food_item in food_items:
            food_lower = food_item.lower()
            # Check if food item appears in cuisine or name
            if food_lower in cuisine or food_lower in name:
                food_item_score = max(food_item_score, 1.0)
            # Check for partial matches
            elif any(word in cuisine or word in name for word in food_lower.split() if len(word) > 3):
                food_item_score = max(food_item_score, 0.8)
    
    # CAPTION-CUISINE DIRECT MATCH (10% weight) - Fallback matching
    caption_cuisine_score = 0.0
    if cuisine:
        # Check if caption contains cuisine name or related terms
        if cuisine in caption_lower:
            caption_cuisine_score = 0.8
        # Check cuisine keywords
        for cuisine_type, keywords in CUISINE_KEYWORDS.items():
            if cuisine_type in cuisine:
                for keyword in keywords:
                    if keyword in caption_lower:
                        caption_cuisine_score = max(caption_cuisine_score, 0.9)
                        break
    
    # RESTAURANT NAME MATCH (3% weight)
    name_score = 0.0
    caption_words = re.findall(r'\b\w+\b', caption_lower)
    for word in caption_words:
        if len(word) > 3 and word in name:
            name_score = max(name_score, 0.7)
    
    # RATING BOOST (1% weight)
    rating = float(restaurant.get("overall_rating", 0) or 0)
    rating_boost = (rating / 5.0) if rating > 0 else 0
    
    # REVIEW COUNT BOOST (1% weight)
    review_count = float(restaurant.get("reviews", 0) or 0)
    review_boost = min(review_count / 50000.0, 1.0) if review_count > 0 else 0
    
    # Combined score prioritizing interpreted cuisine
    score = (
        primary_cuisine_score * 0.60 +      # Primary cuisine match (most important)
        food_item_score * 0.25 +            # Specific food items
        caption_cuisine_score * 0.10 +      # General caption-cuisine match
        name_score * 0.03 +                  # Restaurant name match
        rating_boost * 0.01 +                # Rating boost
        review_boost * 0.01                  # Review count boost
    )
    
    return score


def calculate_image_match_score(restaurant: Dict, caption: str, food_keywords: List[str]) -> float:
    """Calculate match score for image search, emphasizing food and restaurant aspects."""
    name = str(restaurant.get("name", "")).lower()
    cuisine = str(restaurant.get("cuisine", "")).lower()
    caption_lower = caption.lower()
    
    # FOOD/CUISINE MATCHING (50% weight) - Most important for image search
    food_score = 0.0
    
    # Check cuisine match against food keywords
    if cuisine:
        for keyword in food_keywords:
            if keyword in cuisine or cuisine in keyword:
                food_score = max(food_score, 1.0)
                break
        
        # Check if caption words match cuisine
        caption_words = re.findall(r'\b\w+\b', caption_lower)
        for word in caption_words:
            if len(word) > 3 and word in cuisine:
                food_score = max(food_score, 0.8)
        
        # Check cuisine keywords mapping
        for cuisine_type, keywords in CUISINE_KEYWORDS.items():
            if cuisine_type in cuisine or any(kw in cuisine for kw in keywords):
                # Check if any food keyword matches this cuisine
                for keyword in food_keywords:
                    if keyword in keywords or any(kw in keyword for kw in keywords):
                        food_score = max(food_score, 0.9)
                        break
    
    # CAPTION-CUISINE DIRECT MATCH (30% weight)
    caption_cuisine_score = 0.0
    if cuisine:
        # Check if caption contains cuisine name or related terms
        if cuisine in caption_lower or any(word in cuisine for word in re.findall(r'\b\w+\b', caption_lower) if len(word) > 3):
            caption_cuisine_score = 0.8
        
        # Check cuisine keywords
        for cuisine_type, keywords in CUISINE_KEYWORDS.items():
            if cuisine_type in cuisine:
                for keyword in keywords:
                    if keyword in caption_lower:
                        caption_cuisine_score = max(caption_cuisine_score, 1.0)
                        break
    
    # RESTAURANT NAME MATCH (10% weight)
    name_score = 0.0
    caption_words = re.findall(r'\b\w+\b', caption_lower)
    for word in caption_words:
        if len(word) > 3 and word in name:
            name_score = max(name_score, 0.7)
    
    # RATING BOOST (5% weight)
    rating = float(restaurant.get("overall_rating", 0) or 0)
    rating_boost = (rating / 5.0) if rating > 0 else 0
    
    # REVIEW COUNT BOOST (5% weight)
    review_count = float(restaurant.get("reviews", 0) or 0)
    review_boost = min(review_count / 50000.0, 1.0) if review_count > 0 else 0
    
    # Combined score for image search
    score = (
        food_score * 0.50 +
        caption_cuisine_score * 0.30 +
        name_score * 0.10 +
        rating_boost * 0.05 +
        review_boost * 0.05
    )
    
    return score


def search_restaurants_by_image(caption: str) -> Tuple[Dict, List[Dict]]:
    """
    Search restaurants based on image caption with food-focused matching.
    
    New flow:
    1. Interpret the caption as a whole to determine primary cuisine/food type
    2. Use that interpretation to find the best restaurant matches
    """
    df = load_restaurants()
    
    if not caption or caption.strip() == "":
        return {}, []
    
    # STEP 1: Interpret the caption as a whole to determine primary cuisine/food match
    interpretation = interpret_caption_for_cuisine(caption)
    primary_cuisine = interpretation.get("primary_cuisine")
    food_items = interpretation.get("food_items", [])
    confidence = interpretation.get("confidence", 0.0)
    
    print(f"[ImageSearch] Caption interpretation: cuisine={primary_cuisine}, food_items={food_items}, confidence={confidence:.2f}")
    
    # STEP 2: Filter restaurants based on the interpreted cuisine/food
    mask = pd.Series([False] * len(df))
    
    # Priority 1: Match primary cuisine if we have high confidence
    if primary_cuisine and confidence > 0.3:
        # Direct cuisine match
        mask |= df["cuisine"].str.lower().str.contains(primary_cuisine, na=False, regex=False)
        
        # Also check cuisine keywords for the primary cuisine
        if primary_cuisine in CUISINE_KEYWORDS:
            for keyword in CUISINE_KEYWORDS[primary_cuisine]:
                mask |= df["cuisine"].str.lower().str.contains(keyword, na=False, regex=False)
    
    # Priority 2: Match specific food items found in caption
    if food_items:
        for food_item in food_items:
            # Search in cuisine field
            mask |= df["cuisine"].str.lower().str.contains(food_item, na=False, regex=False)
            # Search in name field (restaurants often have food items in name)
            mask |= df["name"].str.lower().str.contains(food_item, na=False, regex=False)
    
    # Priority 3: Fallback - search for any food keywords if no primary match
    if not mask.any() or confidence < 0.3:
        food_keywords = extract_food_keywords_from_caption(caption)
        if food_keywords:
            for keyword in food_keywords:
                mask |= df["cuisine"].str.lower().str.contains(keyword, na=False, regex=False)
                mask |= df["name"].str.lower().str.contains(keyword, na=False, regex=False)
    
    # Priority 4: Broader search with caption words if still no matches
    if not mask.any():
        caption_words = re.findall(r'\b\w+\b', caption.lower())
        for word in caption_words:
            if len(word) > 3:
                mask |= df["cuisine"].str.lower().str.contains(word, na=False, regex=False)
                mask |= df["name"].str.lower().str.contains(word, na=False, regex=False)
    
    matches = df[mask].copy()
    
    if matches.empty:
        return {}, []
    
    # STEP 3: Calculate match scores prioritizing the interpreted cuisine
    matches["match_score"] = matches.apply(
        lambda row: calculate_image_match_score_with_interpretation(
            row.to_dict(), caption, interpretation
        ), axis=1
    )
    
    # Sort by match score, then by rating, then by review count
    matches = matches.sort_values(
        by=["match_score", "overall_rating", "reviews"],
        ascending=[False, False, False]
    )
    
    # Convert to list of dicts
    all_matches = matches.to_dict("records")
    
    # Calculate match score percentiles
    if all_matches:
        scores = [float(r.get("match_score", 0)) for r in all_matches]
        if scores:
            max_score = max(scores)
            min_score = min(scores)
            score_range = max_score - min_score
            
            for r in all_matches:
                score = float(r.get("match_score", 0))
                if score_range > 0:
                    percentile = ((score - min_score) / score_range) * 100
                else:
                    percentile = min(score * 100, 100.0)
                
                percentile = max(percentile, 5.0)
                r["match_score_percentile"] = round(percentile, 1)
        else:
            for r in all_matches:
                r["match_score_percentile"] = 0.0
    
    # Best match is the first one
    best_match = all_matches[0] if all_matches else {}
    
    # Use the same cleaning logic as search_restaurants
    # Import the clean_restaurant function logic inline
    def infer_dietary_accommodations(restaurant: Dict) -> str:
        """Infer dietary accommodations from cuisine, name, and dining style."""
        name = str(restaurant.get("name", "")).lower()
        cuisine = str(restaurant.get("cuisine", "")).lower()
        dining_style = str(restaurant.get("dining_style", "")).lower()
        
        accommodations = []
        
        if "vegan" in cuisine or "vegan" in name:
            accommodations.append("Vegan")
        elif "vegetarian" in cuisine or "vegetarian" in name:
            accommodations.append("Vegetarian")
        
        if "kosher" in cuisine or "kosher" in name:
            accommodations.append("Kosher")
        
        if "gluten" in cuisine or "gluten" in name or "senza gluten" in name:
            accommodations.append("Gluten-Free")
        
        if "halal" in cuisine or "halal" in name:
            accommodations.append("Halal")
        
        return ", ".join(accommodations) if accommodations else "None"
    
    # Ensure address lookup is loaded
    load_restaurants()
    
    # Clean up the data for JSON serialization (reuse logic from search_restaurants)
    def clean_restaurant(r: Dict) -> Dict:
        # Fix Google Maps URL to force English language
        url = str(r.get("url", "") or "")
        if url and "google.com/maps" in url:
            if "hl=" in url:
                url = re.sub(r'hl=[^&]+', 'hl=en', url)
            else:
                if "?" in url:
                    url += "&hl=en"
                else:
                    url += "?hl=en"
        
        price_cat = normalize_price_category(r.get("price_category"), r.get("price_per_person"))
        
        # Extract neighborhood from locality (simplified version)
        locality_raw = str(r.get("locality", "")).strip() if r.get("locality") and pd.notna(r.get("locality")) and str(r.get("locality")).lower() != "nan" else None
        
        borough = None
        neighborhood = None
        
        if locality_raw:
            locality_lower = locality_raw.lower()
            if locality_lower in BOROUGH_NAMES:
                borough = locality_lower.title()
                neighborhood = None
            else:
                for borough_name, neighborhoods in LOCATION_KEYWORDS.items():
                    for neighborhood_keyword in neighborhoods:
                        if neighborhood_keyword.lower() == locality_lower or neighborhood_keyword.lower() in locality_lower:
                            borough = borough_name.title()
                            neighborhood = locality_raw
                            break
                    if borough:
                        break
        
        # Normalize cuisine
        def normalize_cuisine(cuisine_val):
            if not cuisine_val or pd.isna(cuisine_val) or str(cuisine_val).lower() == "nan":
                return None
            cuisine_str = str(cuisine_val).strip()
            special_char_map = {'é': 'e', 'ï': 'i', 'à': 'a', 'è': 'e', 'ù': 'u', 'ô': 'o', 'ê': 'e', 'â': 'a', 'î': 'i', 'û': 'u', 'ç': 'c'}
            for special, normal in special_char_map.items():
                cuisine_str = cuisine_str.replace(special, normal)
                cuisine_str = cuisine_str.replace(special.upper(), normal.upper())
            parts = re.split(r'([()])', cuisine_str)
            normalized_parts = []
            for part in parts:
                if part in ['(', ')']:
                    normalized_parts.append(part)
                elif part.strip():
                    words = part.split()
                    normalized_part = ' '.join(word.capitalize() for word in words)
                    normalized_parts.append(normalized_part)
                else:
                    normalized_parts.append(part)
            return ''.join(normalized_parts)
        
        cuisine = normalize_cuisine(r.get("cuisine"))
        
        # Normalize dining style
        def normalize_dining_style(style_val):
            if not style_val or pd.isna(style_val) or str(style_val).lower() == "nan":
                return None
            style_str = str(style_val).strip()
            words = style_str.split()
            return ' '.join(word.capitalize() for word in words)
        
        dining_style = normalize_dining_style(r.get("dining_style"))
        dietary_accommodations = infer_dietary_accommodations(r)
        
        return {
            "name": str(r.get("name", "")),
            "overall_rating": float(r.get("overall_rating", 0)) if r.get("overall_rating") and pd.notna(r.get("overall_rating")) else None,
            "reviews": float(r.get("reviews", 0)) if r.get("reviews") and pd.notna(r.get("reviews")) else 0,
            "price_category": price_cat,
            "price_per_person": str(r.get("price_per_person", "")).strip() if r.get("price_per_person") and pd.notna(r.get("price_per_person")) and str(r.get("price_per_person")).lower() != "nan" else None,
            "borough": borough,
            "neighborhood": neighborhood,
            "cuisine": cuisine,
            "food": float(r.get("food", 0)) if r.get("food") and pd.notna(r.get("food")) and float(r.get("food", 0)) > 0 else None,
            "service": float(r.get("service", 0)) if r.get("service") and pd.notna(r.get("service")) and float(r.get("service", 0)) > 0 else None,
            "ambiance": float(r.get("ambiance", 0)) if r.get("ambiance") and pd.notna(r.get("ambiance")) and float(r.get("ambiance", 0)) > 0 else None,
            "dining_style": dining_style,
            "dietary_accommodations": dietary_accommodations,
            "lat": float(r.get("lat", 0)) if r.get("lat") and pd.notna(r.get("lat")) else None,
            "lon": float(r.get("lon", 0)) if r.get("lon") and pd.notna(r.get("lon")) else None,
            "zipcode": str(r.get("zipcode")).split('.')[0] if r.get("zipcode") and pd.notna(r.get("zipcode")) else None,
            "url": url,
            "match_score": float(r.get("match_score", 0)),
            "match_score_percentile": float(r.get("match_score_percentile", 0)),
        }
    
    best_match_clean = clean_restaurant(best_match) if best_match else {}
    all_matches_clean = [clean_restaurant(r) for r in all_matches]
    
    return best_match_clean, all_matches_clean


def search_restaurants(query: str) -> Tuple[Dict, List[Dict]]:
    """Search restaurants optimized for general discovery queries."""
    df = load_restaurants()
    
    if query.strip() == "":
        return {}, []
    
    query_lower = query.lower()
    query_words = re.findall(r'\b\w+\b', query_lower)
    
    # Extract keywords for better filtering
    cuisine_keywords, location_keywords = extract_query_keywords(query)
    
    # More flexible filtering - check if any keywords match
    # This allows queries like "asian food in queens" to work
    mask = pd.Series([False] * len(df))
    
    # Check cuisine keywords
    if cuisine_keywords:
        for keyword in cuisine_keywords:
            mask |= df["cuisine"].str.lower().str.contains(keyword, na=False, regex=False)
    
    # Check location keywords
    if location_keywords:
        for keyword in location_keywords:
            mask |= df["locality"].str.lower().str.contains(keyword, na=False, regex=False)
            mask |= df["address"].str.lower().str.contains(keyword, na=False, regex=False)
            # Also check borough field if it exists
            if "borough" in df.columns:
                mask |= df["borough"].str.lower().str.contains(keyword, na=False, regex=False)
    
    # Check individual words in all relevant fields (more inclusive)
    for word in query_words:
        if len(word) > 1:  # Reduced from 2 to 1 to catch more matches
            # Create word boundary pattern for more precise matching
            word_pattern = r'\b' + re.escape(word) + r'\b'
            mask |= (
                df["name"].str.lower().str.contains(word_pattern, na=False, regex=True) |
                df["cuisine"].str.lower().str.contains(word, na=False, regex=False) |
                df["locality"].str.lower().str.contains(word, na=False, regex=False) |
                df["address"].str.lower().str.contains(word, na=False, regex=False)
            )
            # Also check borough if it exists
            if "borough" in df.columns:
                mask |= df["borough"].str.lower().str.contains(word, na=False, regex=False)
    
    # If no matches found with strict filtering, try a more lenient approach
    if not mask.any():
        # Try partial matching for the full query
        full_query_pattern = re.escape(query_lower)
        mask |= (
            df["name"].str.lower().str.contains(full_query_pattern, na=False, regex=True) |
            df["cuisine"].str.lower().str.contains(full_query_pattern, na=False, regex=True) |
            df["locality"].str.lower().str.contains(full_query_pattern, na=False, regex=True) |
            df["address"].str.lower().str.contains(full_query_pattern, na=False, regex=True)
        )
    
    matches = df[mask].copy()
    
    if matches.empty:
        return {}, []
    
    # Calculate match scores
    matches["match_score"] = matches.apply(
        lambda row: calculate_match_score(row.to_dict(), query), axis=1
    )
    
    # Sort by match score, then by rating, then by review count
    matches = matches.sort_values(
        by=["match_score", "overall_rating", "reviews"],
        ascending=[False, False, False]
    )
    
    # Convert to list of dicts
    all_matches = matches.to_dict("records")
    
    # Calculate match score percentiles for all matches
    if all_matches:
        scores = [float(r.get("match_score", 0)) for r in all_matches]
        if scores:
            max_score = max(scores)
            min_score = min(scores)
            score_range = max_score - min_score
            
            # Calculate percentile for each match (0-100 scale)
            for r in all_matches:
                score = float(r.get("match_score", 0))
                if score_range > 0:
                    # Normal percentile calculation when there's a range
                    percentile = ((score - min_score) / score_range) * 100
                else:
                    # When all scores are the same, assign percentile based on absolute score
                    # Convert match score (0-1 range) to percentage (0-100)
                    percentile = min(score * 100, 100.0)
                
                # Ensure minimum percentile for any match found
                percentile = max(percentile, 5.0)  # Minimum 5% for any match
                r["match_score_percentile"] = round(percentile, 1)
        else:
            for r in all_matches:
                r["match_score_percentile"] = 0.0
    
    # Best match is the first one
    best_match = all_matches[0] if all_matches else {}
    
    def infer_dietary_accommodations(restaurant: Dict) -> str:
        """Infer dietary accommodations from cuisine, name, and dining style."""
        name = str(restaurant.get("name", "")).lower()
        cuisine = str(restaurant.get("cuisine", "")).lower()
        dining_style = str(restaurant.get("dining_style", "")).lower()
        
        accommodations = []
        
        # Check for vegan
        if "vegan" in cuisine or "vegan" in name:
            accommodations.append("Vegan")
        # Check for vegetarian
        elif "vegetarian" in cuisine or "vegetarian" in name:
            accommodations.append("Vegetarian")
        
        # Check for kosher
        if "kosher" in cuisine or "kosher" in name:
            accommodations.append("Kosher")
        
        # Check for gluten-free
        if "gluten" in cuisine or "gluten" in name or "senza gluten" in name:
            accommodations.append("Gluten-Free")
        
        # Check for halal (less common but possible)
        if "halal" in cuisine or "halal" in name:
            accommodations.append("Halal")
        
        return ", ".join(accommodations) if accommodations else "None"
    
    # Ensure address lookup is loaded
    load_restaurants()  # This will populate address_lookup if not already done
    
    # Clean up the data for JSON serialization
    def clean_restaurant(r: Dict) -> Dict:
        # Fix Google Maps URL to force English language
        url = str(r.get("url", "") or "")
        if url and "google.com/maps" in url:
            # Replace or add hl parameter to force English
            if "hl=" in url:
                # Replace existing language parameter
                url = re.sub(r'hl=[^&]+', 'hl=en', url)
            else:
                # Add language parameter
                if "?" in url:
                    url += "&hl=en"
                else:
                    url += "?hl=en"
        
        # Normalize price category to consistent format ($, $$, $$$, $$$$)
        price_cat = normalize_price_category(r.get("price_category"), r.get("price_per_person"))
        
        # Extract neighborhood from locality
        locality_raw = str(r.get("locality", "")).strip() if r.get("locality") and pd.notna(r.get("locality")) and str(r.get("locality")).lower() != "nan" else None
        
        borough = None
        neighborhood = None
        
        if locality_raw:
            locality_lower = locality_raw.lower()
            
            # First, check if locality is exactly a borough name
            if locality_lower in BOROUGH_NAMES:
                borough = locality_lower.title()
                neighborhood = None  # Borough and neighborhood cannot be the same
            else:
                # Check if locality matches any neighborhood keywords
                for borough_name, neighborhoods in LOCATION_KEYWORDS.items():
                    for neighborhood_keyword in neighborhoods:
                        # Check for exact match or if keyword is contained in locality
                        if neighborhood_keyword.lower() == locality_lower or neighborhood_keyword.lower() in locality_lower:
                            borough = borough_name.title()
                            neighborhood = locality_raw  # Keep original capitalization
                            break
                    if borough:
                        break
                
                # If no match found but locality exists, try to infer borough from common patterns
                if not borough:
                    # Many neighborhoods in the data are Manhattan neighborhoods
                    # Check if it looks like a Manhattan neighborhood
                    manhattan_indicators = ["midtown", "downtown", "uptown", "village", "soho", "tribeca", "chelsea"]
                    if any(indicator in locality_lower for indicator in manhattan_indicators):
                        borough = "Manhattan"
                        neighborhood = locality_raw
        
        # If no borough found, try to extract from address
        if not borough:
            address = str(r.get("address", "")).strip()
            if (not address or address.lower() == "nan") and address_lookup:
                # Try to find address by matching restaurant name
                restaurant_name = str(r.get("name", "")).strip()
                if restaurant_name:
                    normalized_name = normalize_name_for_matching(restaurant_name)
                    if normalized_name in address_lookup:
                        address = address_lookup[normalized_name]
                    else:
                        # Try fuzzy matching
                        for lookup_name, lookup_address in address_lookup.items():
                            similarity = difflib.SequenceMatcher(None, normalized_name, lookup_name).ratio()
                            if similarity > 0.85:  # High similarity threshold
                                address = lookup_address
                                break
            
            if address and address.lower() not in ["nan", "none", ""]:
                address_lower = address.lower()
                for borough_name, keywords in LOCATION_KEYWORDS.items():
                    # Check for borough name in address
                    if borough_name.lower() in address_lower:
                        borough = borough_name.title()
                        break
                    # Check for neighborhood keywords in address
                    for keyword in keywords:
                        if keyword in address_lower:
                            borough = borough_name.title()
                            break
                    if borough:
                        break
        
        # Final check: ensure borough and neighborhood are never the same
        if borough and neighborhood:
            if neighborhood.lower() == borough.lower():
                neighborhood = None  # Clear neighborhood if it matches borough
        
        # Clean and normalize cuisine
        def normalize_cuisine(cuisine_val):
            """Normalize cuisine: remove special characters, standardize capitalization."""
            if not cuisine_val or pd.isna(cuisine_val) or str(cuisine_val).lower() == "nan":
                return None
            
            cuisine_str = str(cuisine_val).strip()
            
            # Normalize special characters
            # Café -> Cafe, Thaï -> Thai, etc.
            special_char_map = {
                'é': 'e',
                'ï': 'i',
                'à': 'a',
                'è': 'e',
                'ù': 'u',
                'ô': 'o',
                'ê': 'e',
                'â': 'a',
                'î': 'i',
                'û': 'u',
                'ç': 'c',
            }
            for special, normal in special_char_map.items():
                cuisine_str = cuisine_str.replace(special, normal)
                cuisine_str = cuisine_str.replace(special.upper(), normal.upper())
            
            # Title case normalization - handle parenthetical text and multi-word
            import re
            # Split by parentheses to handle cases like "Regional Italian (sardinia)"
            parts = re.split(r'([()])', cuisine_str)
            normalized_parts = []
            
            for part in parts:
                if part in ['(', ')']:
                    normalized_parts.append(part)
                elif part.strip():
                    # Capitalize each word in the part
                    words = part.split()
                    normalized_part = ' '.join(word.capitalize() for word in words)
                    normalized_parts.append(normalized_part)
                else:
                    normalized_parts.append(part)
            
            cuisine_str = ''.join(normalized_parts)
            
            return cuisine_str
        
        cuisine = normalize_cuisine(r.get("cuisine"))
        
        # Clean and normalize dining style
        def normalize_dining_style(style_val):
            """Normalize dining style: standardize capitalization."""
            if not style_val or pd.isna(style_val) or str(style_val).lower() == "nan":
                return None
            
            style_str = str(style_val).strip()
            
            # Title case normalization (first letter of each word uppercase)
            words = style_str.split()
            style_str = ' '.join(word.capitalize() for word in words)
            
            return style_str
        
        dining_style = normalize_dining_style(r.get("dining_style"))
        
        # Infer dietary accommodations
        dietary_accommodations = infer_dietary_accommodations(r)
        
        return {
            "name": str(r.get("name", "")),
            "overall_rating": float(r.get("overall_rating", 0)) if r.get("overall_rating") and pd.notna(r.get("overall_rating")) else None,
            "reviews": float(r.get("reviews", 0)) if r.get("reviews") and pd.notna(r.get("reviews")) else 0,
            "price_category": price_cat,
            "price_per_person": str(r.get("price_per_person", "")).strip() if r.get("price_per_person") and pd.notna(r.get("price_per_person")) and str(r.get("price_per_person")).lower() != "nan" else None,
            "borough": borough,
            "neighborhood": neighborhood,
            "cuisine": cuisine,
            "food": float(r.get("food", 0)) if r.get("food") and pd.notna(r.get("food")) and float(r.get("food", 0)) > 0 else None,
            "service": float(r.get("service", 0)) if r.get("service") and pd.notna(r.get("service")) and float(r.get("service", 0)) > 0 else None,
            "ambiance": float(r.get("ambiance", 0)) if r.get("ambiance") and pd.notna(r.get("ambiance")) and float(r.get("ambiance", 0)) > 0 else None,
            "dining_style": dining_style,
            "dietary_accommodations": dietary_accommodations,
            "lat": float(r.get("lat", 0)) if r.get("lat") and pd.notna(r.get("lat")) else None,
            "lon": float(r.get("lon", 0)) if r.get("lon") and pd.notna(r.get("lon")) else None,
            "zipcode": str(r.get("zipcode")).split('.')[0] if r.get("zipcode") and pd.notna(r.get("zipcode")) else None,
            "url": url,
            "match_score": float(r.get("match_score", 0)),
            "match_score_percentile": float(r.get("match_score_percentile", 0)),
        }
    
    best_match_clean = clean_restaurant(best_match) if best_match else {}
    all_matches_clean = [clean_restaurant(r) for r in all_matches]
    
    return best_match_clean, all_matches_clean


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/image-search")
def image_search():
    """Image search page - placeholder for future version."""
    return render_template("image_search.html")

@app.route("/search")
def search():
    return render_template("index.html")


@app.get("/api/search")
def search_restaurants_api():
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query parameter 'query' is required."}), 400

    if len(query) > 100:
        return jsonify({"error": "Query too long (max 100 characters)."}), 400

    try:
        best_match, all_matches = search_restaurants(query)
        
        if not best_match:
            return jsonify({"error": f"No NYC restaurant found matching '{query}'."}), 404
        
        return jsonify({
            "best_match": best_match,
            "all_matches": all_matches,
            "total_matches": len(all_matches)
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.post("/api/image-search")
def image_search_api():
    """Search restaurants based on uploaded image."""
    if not IMAGE_CAPTIONER_AVAILABLE:
        return jsonify({"error": "Image captioning service not available"}), 500
    
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Get caption generation parameters
        max_length = int(request.form.get('max_length', 80))  # Default increased for detailed captions
        num_beams = int(request.form.get('num_beams', 5))      # Default increased for better quality
        
        # Validate parameters
        max_length = max(20, min(150, max_length))  # Increased range for more detailed descriptions
        num_beams = max(1, min(10, num_beams))      # Increased range for better quality
        
        # Load and process image
        image = Image.open(io.BytesIO(file.read()))
        image = image.convert("RGB")
        
        # Generate caption
        captioner = ImageCaptioner()
        caption = captioner.generate_caption(image, max_length=max_length, num_beams=num_beams)
        
        if not caption or caption.strip() == "":
            return jsonify({"error": "Failed to generate caption from image"}), 500
        
        # Search restaurants based on caption
        best_match, all_matches = search_restaurants_by_image(caption)
        
        if not best_match:
            return jsonify({
                "error": f"No restaurants found matching the image caption: '{caption}'",
                "caption": caption
            }), 404
        
        return jsonify({
            "caption": caption,
            "best_match": best_match,
            "all_matches": all_matches,
            "total_matches": len(all_matches)
        })
    except Exception as e:
        import traceback
        print(f"Error in image_search_api: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.get("/api/dashboard/stats")
def dashboard_stats():
    """Get overall statistics for the dashboard."""
    try:
        df = load_restaurants()
        
        # Basic stats
        total_restaurants = len(df)
        median_rating = df['overall_rating'].median() if 'overall_rating' in df.columns else 0
        total_reviews = df['reviews'].sum() if 'reviews' in df.columns else 0
        
        # Rating distribution (more granular intervals for meaningful insights)
        rating_dist = {}
        if 'overall_rating' in df.columns:
            # Define rating intervals: one broad range for low ratings, then 0.3 intervals for high ratings
            rating_intervals = [
                (0, 3.5, "0-3.5"),
                (3.5, 3.8, "3.5-3.8"), 
                (3.8, 4.1, "3.8-4.1"),
                (4.1, 4.4, "4.1-4.4"),
                (4.4, 4.7, "4.4-4.7"),
                (4.7, 5.0, "4.7-5.0")
            ]
            
            for min_rating, max_rating, label in rating_intervals:
                if max_rating == 5.0:
                    # Include 5.0 in the last interval
                    count = len(df[(df['overall_rating'] >= min_rating) & (df['overall_rating'] <= max_rating)])
                else:
                    count = len(df[(df['overall_rating'] >= min_rating) & (df['overall_rating'] < max_rating)])
                if count > 0:
                    rating_dist[label] = int(count)
        
        # Cuisine distribution (comprehensive with name-based inference)
        cuisine_dist = {}
        if 'cuisine' in df.columns:
            # Create enhanced cuisine data by inferring from names
            enhanced_cuisines = []
            inferred_count = 0
            
            for _, row in df.iterrows():
                original_cuisine = row.get('cuisine', '')
                
                # If cuisine is missing or empty, try to infer from name
                if pd.isna(original_cuisine) or str(original_cuisine).strip() == "":
                    inferred_cuisine = infer_cuisine_from_name(row.get('name', ''))
                    if inferred_cuisine:
                        enhanced_cuisines.append(inferred_cuisine)
                        inferred_count += 1
                    else:
                        enhanced_cuisines.append('Unknown')
                else:
                    enhanced_cuisines.append(str(original_cuisine).strip())
            
            # Get cuisine counts from enhanced data
            enhanced_series = pd.Series(enhanced_cuisines)
            cuisine_counts = enhanced_series.value_counts()
            
            # Create consolidated cuisine categories
            consolidated_cuisines = {}
            
            # Define cuisine consolidation rules
            consolidation_map = {
                # Italian (combine Italian + Pizza)
                'Italian': ['Italian', 'Pizza'],
                
                # American (combine American + Contemporary American + Steakhouse)  
                'American': ['American', 'Contemporary American', 'Steakhouse'],
                
                # Japanese (combine Japanese + Sushi)
                'Japanese': ['Japanese', 'Sushi'],
                
                # Mediterranean (combine Mediterranean + Greek)
                'Mediterranean': ['Mediterranean', 'Greek'],
                
                # Casual Dining (combine service-style categories)
                'Casual Dining': ['Deli', 'Coffee', 'Bakery', 'Burger', 'BBQ'],
                
                # Keep Asian cuisines separate for more detail
                'Chinese': ['Chinese'],
                'Thai': ['Thai'],
                'Korean': ['Korean'],
                'Vietnamese': ['Vietnamese'],
                
                # Keep these as individual categories
                'Mexican': ['Mexican'],
                'French': ['French'], 
                'Indian': ['Indian'],
                'Seafood': ['Seafood'],
                'Caribbean': ['Caribbean'],
                'Spanish': ['Spanish']
            }
            
            # Apply consolidation
            remaining_cuisines = dict(cuisine_counts)
            
            for consolidated_name, source_cuisines in consolidation_map.items():
                total_count = 0
                for source in source_cuisines:
                    if source in remaining_cuisines:
                        total_count += remaining_cuisines[source]
                        del remaining_cuisines[source]
                
                if total_count > 0:
                    consolidated_cuisines[consolidated_name] = int(total_count)
            
            # Handle remaining cuisines
            unknown_count = remaining_cuisines.pop('Unknown', 0)
            
            # Add significant remaining cuisines individually (>= 15 restaurants)
            for cuisine, count in remaining_cuisines.items():
                if count >= 15:
                    consolidated_cuisines[cuisine] = int(count)
                else:
                    # Add to "Other" category
                    if 'Other' not in consolidated_cuisines:
                        consolidated_cuisines['Other'] = 0
                    consolidated_cuisines['Other'] += int(count)
            
            # Sort by count and prepare final distribution
            sorted_cuisines = sorted(consolidated_cuisines.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 15 categories
            for cuisine, count in sorted_cuisines[:15]:
                cuisine_dist[cuisine] = int(count)
            
            # Add remaining as "Other" if we have more than 15 categories
            if len(sorted_cuisines) > 15:
                additional_other = sum(count for _, count in sorted_cuisines[15:])
                cuisine_dist['Other'] = int(cuisine_dist.get('Other', 0) + additional_other)
            
            # Add "Unknown" category
            if unknown_count > 0:
                cuisine_dist["Unknown"] = int(unknown_count)
            
            print(f"Cuisine inference: Successfully inferred {inferred_count} cuisines from restaurant names")
        
        # Price category distribution
        price_dist = {}
        if 'price_category' in df.columns:
            # Use normalize_price_category to get consistent price categories
            df['normalized_price'] = df.apply(lambda row: normalize_price_category(row.get("price_category"), row.get("price_per_person")), axis=1)
            price_counts = df['normalized_price'].value_counts(dropna=False)
            
            # Map NaN to "Unknown"
            price_dist = {str(k) if pd.notna(k) else "Unknown": int(v) for k, v in price_counts.items()}
        
        # Borough distribution (from locality, address, and zip codes)
        borough_dist = {}
        
        # Create a comprehensive borough extraction function
        def extract_borough_comprehensive(df):
            """Extract borough information from locality, address, and zip codes."""
            borough_counts = {"Manhattan": 0, "Brooklyn": 0, "Queens": 0, "Bronx": 0, "Staten Island": 0}
            
            # Track which restaurants have been assigned to avoid double counting
            assigned = pd.Series([False] * len(df))
            
            # 1. First, check locality field for Manhattan neighborhoods (most reliable)
            if 'locality' in df.columns:
                manhattan_keywords = ["manhattan", "midtown", "downtown", "uptown", "village", "soho", "tribeca", "chelsea", 
                                    "hell's kitchen", "financial district", "battery park", "gramercy", "flatiron", 
                                    "harlem", "nolita", "little italy", "hudson square", "upper east side", "upper west side",
                                    "east village", "west village", "lower east side"]
                
                manhattan_mask = pd.Series([False] * len(df))
                for keyword in manhattan_keywords:
                    manhattan_mask |= df['locality'].str.lower().str.contains(keyword, na=False, regex=False)
                
                borough_counts["Manhattan"] = manhattan_mask.sum()
                assigned |= manhattan_mask
            
            # 2. Check addresses for explicit borough names
            if 'address' in df.columns:
                address_lower = df['address'].str.lower().fillna('')
                
                # Brooklyn
                brooklyn_mask = address_lower.str.contains('brooklyn', na=False) & ~assigned
                borough_counts["Brooklyn"] += brooklyn_mask.sum()
                assigned |= brooklyn_mask
                
                # Queens  
                queens_mask = address_lower.str.contains('queens', na=False) & ~assigned
                borough_counts["Queens"] += queens_mask.sum()
                assigned |= queens_mask
                
                # Bronx
                bronx_mask = address_lower.str.contains('bronx', na=False) & ~assigned
                borough_counts["Bronx"] += bronx_mask.sum()
                assigned |= bronx_mask
                
                # Staten Island
                si_mask = address_lower.str.contains('staten island', na=False) & ~assigned
                borough_counts["Staten Island"] += si_mask.sum()
                assigned |= si_mask
                
                # Manhattan (for addresses that explicitly mention it and weren't caught by locality)
                manhattan_addr_mask = address_lower.str.contains('manhattan', na=False) & ~assigned
                borough_counts["Manhattan"] += manhattan_addr_mask.sum()
                assigned |= manhattan_addr_mask
                
                # New York, NY addresses (likely Manhattan if not already assigned)
                ny_mask = (address_lower.str.contains('new york, ny', na=False) & ~assigned)
                borough_counts["Manhattan"] += ny_mask.sum()
                assigned |= ny_mask
            
            # 3. Use zip codes as fallback for remaining unassigned restaurants
            if 'zipcode' in df.columns:
                zipcode_str = df['zipcode'].astype(str).str.replace('.0', '', regex=False)
                
                # Manhattan zip codes (100xx, 101xx, 102xx)
                manhattan_zip_mask = (zipcode_str.str.match(r'^10[0-2]\d{2}$') & ~assigned)
                borough_counts["Manhattan"] += manhattan_zip_mask.sum()
                assigned |= manhattan_zip_mask
                
                # Bronx zip codes (104xx)
                bronx_zip_mask = (zipcode_str.str.match(r'^104\d{2}$') & ~assigned)
                borough_counts["Bronx"] += bronx_zip_mask.sum()
                assigned |= bronx_zip_mask
                
                # Brooklyn zip codes (112xx)
                brooklyn_zip_mask = (zipcode_str.str.match(r'^112\d{2}$') & ~assigned)
                borough_counts["Brooklyn"] += brooklyn_zip_mask.sum()
                assigned |= brooklyn_zip_mask
                
                # Queens zip codes (11xxx except 112xx)
                queens_zip_mask = (zipcode_str.str.match(r'^11[013-9]\d{2}$') & ~assigned)
                borough_counts["Queens"] += queens_zip_mask.sum()
                assigned |= queens_zip_mask
                
                # Staten Island zip codes (103xx)
                si_zip_mask = (zipcode_str.str.match(r'^103\d{2}$') & ~assigned)
                borough_counts["Staten Island"] += si_zip_mask.sum()
                assigned |= si_zip_mask
            
            return borough_counts
        
        borough_dist = extract_borough_comprehensive(df)
        # Only include boroughs with restaurants and convert to regular int for JSON serialization
        borough_dist = {k: int(v) for k, v in borough_dist.items() if v > 0}
        
        return jsonify({
            "total_restaurants": int(total_restaurants),
            "median_rating": round(float(median_rating), 2),
            "total_reviews": int(total_reviews),
            "rating_distribution": rating_dist,
            "cuisine_distribution": cuisine_dist,
            "price_distribution": price_dist,
            "borough_distribution": borough_dist
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.get("/api/dashboard/ratings-comparison")
def ratings_comparison():
    """Get ratings comparison by rating intervals."""
    try:
        df = load_restaurants()
        
        # Define the same rating intervals as in the pie chart
        rating_intervals = [
            (0, 3.5, "0-3.5"),
            (3.5, 3.8, "3.5-3.8"), 
            (3.8, 4.1, "3.8-4.1"),
            (4.1, 4.4, "4.1-4.4"),
            (4.4, 4.7, "4.4-4.7"),
            (4.7, 5.0, "4.7-5.0")
        ]
        
        interval_data = {}
        
        for min_rating, max_rating, label in rating_intervals:
            if max_rating == 5.0:
                # Include 5.0 in the last interval
                interval_mask = (df['overall_rating'] >= min_rating) & (df['overall_rating'] <= max_rating)
            else:
                interval_mask = (df['overall_rating'] >= min_rating) & (df['overall_rating'] < max_rating)
            
            interval_df = df[interval_mask]
            
            if len(interval_df) > 0:
                # Calculate averages for each rating type
                overall_avg = interval_df['overall_rating'].mean()
                food_avg = interval_df[interval_df['food'].notna() & (interval_df['food'] > 0)]['food'].mean()
                service_avg = interval_df[interval_df['service'].notna() & (interval_df['service'] > 0)]['service'].mean()
                ambiance_avg = interval_df[interval_df['ambiance'].notna() & (interval_df['ambiance'] > 0)]['ambiance'].mean()
                
                # Count of restaurants with each rating type
                food_count = len(interval_df[interval_df['food'].notna() & (interval_df['food'] > 0)])
                service_count = len(interval_df[interval_df['service'].notna() & (interval_df['service'] > 0)])
                ambiance_count = len(interval_df[interval_df['ambiance'].notna() & (interval_df['ambiance'] > 0)])
                
                interval_data[label] = {
                    "overall": {
                        "avg": round(float(overall_avg), 2) if pd.notna(overall_avg) else 0,
                        "count": len(interval_df)
                    },
                    "food": {
                        "avg": round(float(food_avg), 2) if pd.notna(food_avg) else 0,
                        "count": food_count
                    },
                    "service": {
                        "avg": round(float(service_avg), 2) if pd.notna(service_avg) else 0,
                        "count": service_count
                    },
                    "ambiance": {
                        "avg": round(float(ambiance_avg), 2) if pd.notna(ambiance_avg) else 0,
                        "count": ambiance_count
                    }
                }
        
        return jsonify(interval_data)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.get("/api/dashboard/top-restaurants")
def top_restaurants():
    """Get top restaurants by rating and review count."""
    try:
        df = load_restaurants()
        
        # Top by rating (min 10 reviews)
        top_by_rating = df[df['reviews'] >= 10].nlargest(10, 'overall_rating')[
            ['name', 'overall_rating', 'reviews', 'cuisine', 'locality']
        ].to_dict('records')
        
        # Top by review count
        top_by_reviews = df.nlargest(10, 'reviews')[
            ['name', 'overall_rating', 'reviews', 'cuisine', 'locality']
        ].to_dict('records')
        
        # Clean up the data
        def clean_restaurant_record(r):
            return {
                "name": str(r.get('name', '')),
                "rating": float(r.get('overall_rating', 0)) if pd.notna(r.get('overall_rating')) else 0,
                "reviews": int(r.get('reviews', 0)) if pd.notna(r.get('reviews')) else 0,
                "cuisine": str(r.get('cuisine', '')) if pd.notna(r.get('cuisine')) else 'N/A',
                "locality": str(r.get('locality', '')) if pd.notna(r.get('locality')) else 'N/A'
            }
        
        return jsonify({
            "top_by_rating": [clean_restaurant_record(r) for r in top_by_rating],
            "top_by_reviews": [clean_restaurant_record(r) for r in top_by_reviews]
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.get("/api/dashboard/geographic")
def geographic_data():
    """Get geographic distribution data."""
    try:
        # Use cached data for better performance
        df = load_restaurants()
        
        # For geographic data, we need to handle NaN values properly
        # So we'll work with a copy and convert lat/lon to numeric, keeping NaN for invalid values
        df = df.copy()
        if 'lat' in df.columns:
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        if 'lon' in df.columns:
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        
        # Filter restaurants with valid coordinates (not NaN and not 0)
        geo_df = df[(df['lat'].notna()) & (df['lon'].notna()) & (df['lat'] != 0) & (df['lon'] != 0)]
        
        geo_data = []
        for _, row in geo_df.iterrows():
            try:
                lat_val = float(row.get('lat', 0))
                lon_val = float(row.get('lon', 0))
                # Additional validation: ensure coordinates are in reasonable range for NYC
                if 40.0 <= lat_val <= 41.0 and -75.0 <= lon_val <= -73.0:
                    geo_data.append({
                        "name": str(row.get('name', '')) if pd.notna(row.get('name')) else 'Unknown',
                        "lat": lat_val,
                        "lon": lon_val,
                        "rating": float(row.get('overall_rating', 0)) if pd.notna(row.get('overall_rating')) else 0,
                        "reviews": int(row.get('reviews', 0)) if pd.notna(row.get('reviews')) else 0,
                        "cuisine": str(row.get('cuisine', '')) if pd.notna(row.get('cuisine')) and str(row.get('cuisine')).strip() != '' else 'N/A'
                    })
            except (ValueError, TypeError):
                # Skip rows with invalid coordinate data
                continue
        
        return jsonify({
            "restaurants": geo_data,
            "count": len(geo_data)
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# Yelp Analysis Routes for USA Tab
@app.get("/api/usa/yelp-charts")
def usa_yelp_charts():
    """Get Yelp analysis charts for USA tab."""
    global usa_charts_cache, usa_charts_json
    import time
    request_start = time.time()
    
    try:
        if not YELP_CHARTS_AVAILABLE:
            return jsonify({
                "error": "Yelp chart generator not available",
                "message": "The USA Yelp analysis requires the Yelp Academic Dataset JSON files. These files are not included in the repository due to size limitations."
            }), 503  # 503 Service Unavailable is more appropriate
        
        # Use pre-serialized JSON if available (ultra-fast path - no serialization overhead)
        if usa_charts_json is not None:
            from flask import Response
            import gzip
            
            # Compress response to reduce transfer time (331KB -> ~50-80KB)
            compress_start = time.time()
            compressed = gzip.compress(usa_charts_json.encode('utf-8'))
            compress_time = time.time() - compress_start
            
            response = Response(
                compressed,
                mimetype='application/json',
                headers={
                    'Cache-Control': 'public, max-age=3600',
                    'Content-Encoding': 'gzip',
                    'Content-Length': str(len(compressed))
                }
            )
            
            total_time = time.time() - request_start
            print(f"[PERF] USA charts response: {total_time*1000:.1f}ms (compress: {compress_time*1000:.1f}ms, size: {len(compressed)/1024:.1f}KB)")
            return response
        
        # Fallback: use cached dict (still fast, but requires jsonify)
        if usa_charts_cache is not None:
            response = jsonify(usa_charts_cache)
            response.headers['Cache-Control'] = 'public, max-age=3600'
            return response
        
        # Fallback: generate charts on demand (slower)
        print("[*] Generating charts on demand (cache not available)...")
        generator = YelpChartGenerator()
        charts = generator.generate_all_charts()
        
        if charts is None:
            return jsonify({"error": "Failed to load Yelp data or generate charts"}), 500
        
        # Pre-serialize Plotly charts to JSON strings for faster responses
        if isinstance(charts, dict):
            for key in ['sentiment_chart', 'word_frequency_chart', 'theme_chart']:
                if key in charts:
                    # Already a JSON string, skip
                    if isinstance(charts[key], str):
                        continue
                    # Convert Plotly figure to JSON string if it's a figure object
                    if hasattr(charts[key], 'to_json'):
                        charts[key] = charts[key].to_json()
        
        # Cache for next request
        usa_charts_cache = charts
        
        # Pre-serialize to JSON string for future requests
        import json
        try:
            usa_charts_json = json.dumps(charts)
        except (TypeError, ValueError) as json_error:
            return jsonify({"error": f"Data serialization error: {str(json_error)}"}), 500
        
        # Return pre-serialized response
        from flask import Response
        response = Response(
            usa_charts_json,
            mimetype='application/json',
            headers={'Cache-Control': 'public, max-age=3600'}
        )
        return response
        
    except Exception as e:
        import traceback
        print(f"Error in usa_yelp_charts: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.get("/api/usa/stats")
def usa_stats():
    """Get basic stats for USA Yelp data."""
    try:
        if not YELP_CHARTS_AVAILABLE:
            return jsonify({"error": "Yelp data not available"}), 500
        
        generator = YelpChartGenerator()
        if generator.load_data() is None:
            return jsonify({"error": "Failed to load Yelp data"}), 500
        
        # Get basic statistics
        df = generator.df
        sentiment_counts = df['sentiment'].value_counts()
        
        stats = {
            "total_restaurants": 150000,  # Known Yelp dataset restaurant count
            "sample_size": len(df),       # Actual sample processed
            "total_reviews": 8000000,     # Known Yelp dataset total reviews
            "positive_reviews": int(sentiment_counts.get('positive', 0)),
            "negative_reviews": int(sentiment_counts.get('negative', 0)),
            "neutral_reviews": int(sentiment_counts.get('neutral', 0)),
            "positive_percentage": round(sentiment_counts.get('positive', 0) / len(df) * 100, 1),
            "data_source": "Yelp Academic Dataset"
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    import sys
    import traceback
    import io
    
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    try:
        print("Starting Flask app...")
        print("App initialized successfully")
        
        # Download required NLTK data at startup
        print("[*] Ensuring NLTK data is available...")
        try:
            import nltk
            # Download punkt_tab (newer NLTK) or punkt (older)
            try:
                nltk.data.find('tokenizers/punkt_tab')
                print("[OK] NLTK punkt_tab found")
            except LookupError:
                try:
                    nltk.data.find('tokenizers/punkt')
                    print("[OK] NLTK punkt found")
                except LookupError:
                    print("[*] Downloading NLTK punkt_tab...")
                    try:
                        nltk.download('punkt_tab', quiet=True)
                        print("[OK] NLTK punkt_tab downloaded")
                    except:
                        nltk.download('punkt', quiet=True)
                        print("[OK] NLTK punkt downloaded")
        except Exception as e:
            print(f"[WARNING] NLTK setup issue: {e}")
        
        # Check if Yelp charts are available
        if YELP_CHARTS_AVAILABLE:
            print("[OK] Yelp chart generator available")
            # Preload USA charts cache at startup for fast access
            load_usa_charts_cache()
        else:
            print("[WARNING] Yelp chart generator not available")
        
        # Run the app with environment variable support for production
        import os
        host = os.environ.get('HOST', '127.0.0.1')
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('FLASK_ENV', 'development') != 'production'
        
        print(f"Starting server on http://{host}:{port}")
        print(f"Debug mode: {debug}")
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)
