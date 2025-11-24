"""Script to merge nyc_restaurants.csv and kayak_data.csv, checking for duplicates."""

import pandas as pd
from pathlib import Path
import difflib


def normalize_name(name: str) -> str:
    """Normalize restaurant name for comparison."""
    if pd.isna(name) or name == "":
        return ""
    return str(name).lower().strip().replace("'", "").replace("&", "and")


def are_similar(name1: str, name2: str, threshold: float = 0.85) -> bool:
    """Check if two restaurant names are similar."""
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    if not norm1 or not norm2:
        return False
    ratio = difflib.SequenceMatcher(None, norm1, norm2).ratio()
    return ratio >= threshold


def merge_restaurants():
    """Merge the two CSV files, handling duplicates."""
    # Update these paths to point to your CSV files
    nyc_path = Path("nyc_restaurants.csv")
    kayak_path = Path("kayak_data.csv")
    output_path = Path("nyc_restaurants_merged.csv")
    
    print("Loading CSV files...")
    nyc_df = pd.read_csv(nyc_path)
    kayak_df = pd.read_csv(kayak_path)
    
    print(f"Loaded {len(nyc_df)} restaurants from nyc_restaurants.csv")
    print(f"Loaded {len(kayak_df)} restaurants from kayak_data.csv")
    
    # Standardize column names for nyc_df
    nyc_df = nyc_df.rename(columns={
        "Name": "name",
        "Rating": "overall_rating",
        "Rating Count": "reviews",
        "Price Category": "price_category",
        "Address": "address",
        "Lat": "lat",
        "Lon": "lon",
        "ZipCode": "zipcode",
        "URL": "url"
    })
    
    # Standardize column names for kayak_df
    kayak_df = kayak_df.rename(columns={
        "locality": "locality",
        "cuisine": "cuisine",
        "price_per_person": "price_per_person",
        "overall_rating": "overall_rating",
        "food": "food",
        "service": "service",
        "ambiance": "ambiance",
        "reviews": "reviews",
        "noise": "noise",
        "dining_style": "dining_style"
    })
    
    # Add missing columns to nyc_df
    for col in ["cuisine", "price_per_person", "food", "service", "ambiance", "noise", "dining_style", "locality"]:
        if col not in nyc_df.columns:
            nyc_df[col] = None
    
    # Add missing columns to kayak_df
    for col in ["lat", "lon", "zipcode", "price_category"]:
        if col not in kayak_df.columns:
            kayak_df[col] = None
    
    # Convert numeric columns
    for df in [nyc_df, kayak_df]:
        df["overall_rating"] = pd.to_numeric(df["overall_rating"], errors="coerce")
        df["reviews"] = pd.to_numeric(df["reviews"], errors="coerce")
        df["food"] = pd.to_numeric(df["food"], errors="coerce")
        df["service"] = pd.to_numeric(df["service"], errors="coerce")
        df["ambiance"] = pd.to_numeric(df["ambiance"], errors="coerce")
    
    # Find duplicates by comparing names
    print("\nChecking for duplicates...")
    merged_rows = []
    kayak_used = set()
    
    # Start with all nyc restaurants
    for idx, nyc_row in nyc_df.iterrows():
        merged_rows.append(nyc_row.to_dict())
    
    # Add kayak restaurants, checking for duplicates
    duplicates_found = 0
    for idx, kayak_row in kayak_df.iterrows():
        kayak_name = kayak_row["name"]
        is_duplicate = False
        
        # Check if this restaurant already exists
        for existing_row in merged_rows:
            existing_name = existing_row.get("name", "")
            if are_similar(kayak_name, existing_name):
                # Merge data: prefer non-null values from kayak
                for key, value in kayak_row.items():
                    if pd.notna(value) and value != "":
                        if existing_row.get(key) is None or existing_row.get(key) == "":
                            existing_row[key] = value
                is_duplicate = True
                duplicates_found += 1
                break
        
        if not is_duplicate:
            merged_rows.append(kayak_row.to_dict())
    
    print(f"Found {duplicates_found} duplicates (merged)")
    print(f"Total unique restaurants: {len(merged_rows)}")
    
    # Create merged dataframe
    merged_df = pd.DataFrame(merged_rows)
    
    # Ensure consistent column order
    columns_order = [
        "name", "overall_rating", "reviews", "price_category", "price_per_person",
        "address", "locality", "cuisine", "food", "service", "ambiance",
        "dining_style", "noise", "lat", "lon", "zipcode", "url"
    ]
    
    # Add any missing columns
    for col in columns_order:
        if col not in merged_df.columns:
            merged_df[col] = None
    
    # Reorder columns
    existing_columns = [col for col in columns_order if col in merged_df.columns]
    merged_df = merged_df[existing_columns]
    
    # Save merged CSV
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged data saved to: {output_path}")
    print(f"Total restaurants: {len(merged_df)}")
    
    return merged_df


if __name__ == "__main__":
    merge_restaurants()

