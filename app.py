from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template, request

# CSV file path - update this to point to your merged restaurants CSV file
# The CSV should be created by running merge_restaurants.py
CSV_PATH = Path("nyc_restaurants_merged.csv")

app = Flask(__name__)

# Load restaurants data once at startup
restaurants_df = None
address_lookup = None


def normalize_name_for_matching(name: str) -> str:
    """Normalize restaurant name for matching."""
    if pd.isna(name) or name == "":
        return ""
    return str(name).lower().strip().replace("'", "").replace("&", "and").replace("-", " ")


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
                    nyc_path = Path("nyc_restaurants.csv")
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
    
    # CUISINE MATCHING (40% weight) - Most important for discovery
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
    
    # LOCATION MATCHING (35% weight) - Second most important
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
    
    # NAME MATCHING (10% weight) - Reduced importance for discovery
    name_score = 0.0
    if query_lower in name:
        name_score = 0.9
    else:
        # Check if any significant query words are in name
        query_words = re.findall(r'\b\w+\b', query_lower)
        for word in query_words:
            if len(word) > 3 and word in name:
                name_score = max(name_score, 0.5)
        
        # Similarity as fallback
        if name_score == 0:
            name_score = difflib.SequenceMatcher(None, name, query_lower).ratio() * 0.3
    
    # RATING BOOST (10% weight) - Quality indicator
    rating = float(restaurant.get("overall_rating", 0) or 0)
    rating_boost = (rating / 5.0) if rating > 0 else 0
    
    # REVIEW COUNT BOOST (5% weight) - Popularity indicator
    review_count = float(restaurant.get("reviews", 0) or 0)
    review_boost = min(review_count / 50000.0, 1.0) if review_count > 0 else 0
    
    # Combined score optimized for discovery queries
    # Cuisine (40%), Location (35%), Name (10%), Rating (10%), Reviews (5%)
    score = (
        cuisine_score * 0.40 +
        location_score * 0.35 +
        name_score * 0.10 +
        rating_boost * 0.10 +
        review_boost * 0.05
    )
    
    return score


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
    
    # Check cuisine
    if cuisine_keywords:
        for keyword in cuisine_keywords:
            mask |= df["cuisine"].str.lower().str.contains(keyword, na=False, regex=False)
    
    # Check location
    if location_keywords:
        for keyword in location_keywords:
            mask |= df["locality"].str.lower().str.contains(keyword, na=False, regex=False)
            mask |= df["address"].str.lower().str.contains(keyword, na=False, regex=False)
    
    # Also check individual words in name, cuisine, locality, address
    for word in query_words:
        if len(word) > 2:  # Skip very short words
            mask |= (
                df["name"].str.lower().str.contains(word, na=False, regex=False) |
                df["cuisine"].str.lower().str.contains(word, na=False, regex=False) |
                df["locality"].str.lower().str.contains(word, na=False, regex=False) |
                df["address"].str.lower().str.contains(word, na=False, regex=False)
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
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            # Calculate percentile for each match (0-100 scale)
            for r in all_matches:
                score = float(r.get("match_score", 0))
                if score_range > 0:
                    percentile = ((score - min_score) / score_range) * 100
                else:
                    percentile = 100.0
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


if __name__ == "__main__":
    app.run(debug=True)
