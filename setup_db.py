#!/usr/bin/env python3
"""
Create a SQLite database containing California restaurants and their reviews
from the Yelp Academic Dataset.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CA restaurant SQLite DB from Yelp JSON files.")
    parser.add_argument(
        "--business-json",
        type=Path,
        required=True,
        help="Path to the Yelp business JSON lines file.",
    )
    parser.add_argument(
        "--review-json",
        type=Path,
        required=True,
        help="Path to the Yelp review JSON lines file.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("ny_restaurants.sqlite"),
        help="SQLite DB file to create.",
    )
    parser.add_argument(
        "--business-limit",
        type=int,
        default=None,
        help="Limit number of business rows processed (for testing).",
    )
    parser.add_argument(
        "--review-limit",
        type=int,
        default=None,
        help="Limit number of review rows processed (for testing).",
    )
    return parser.parse_args()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;

        CREATE TABLE IF NOT EXISTS businesses (
            business_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            address TEXT,
            city TEXT,
            state TEXT,
            postal_code TEXT,
            latitude REAL,
            longitude REAL,
            stars REAL,
            review_count INTEGER,
            is_open INTEGER,
            categories TEXT
        );

        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY,
            business_id TEXT NOT NULL,
            stars REAL NOT NULL,
            useful INTEGER,
            funny INTEGER,
            cool INTEGER,
            text TEXT,
            date TEXT,
            FOREIGN KEY (business_id) REFERENCES businesses (business_id)
        );

        CREATE INDEX IF NOT EXISTS idx_businesses_name ON businesses (name);
        CREATE INDEX IF NOT EXISTS idx_reviews_business ON reviews (business_id);
        """
    )


def iter_json_lines(path: Path, limit: Optional[int]) -> Iterator[dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                break


def load_businesses(conn: sqlite3.Connection, path: Path, limit: Optional[int]) -> Set[str]:
    business_ids: Set[str] = set()
    rows: List[Tuple] = []

    for record in iter_json_lines(path, limit):
        if record.get("state") != "CA":
            continue
        categories = record.get("categories") or ""
        if "Restaurants" not in categories:
            continue

        business_id = record["business_id"]
        business_ids.add(business_id)
        rows.append(
            (
                business_id,
                record.get("name"),
                record.get("address"),
                record.get("city"),
                record.get("state"),
                record.get("postal_code"),
                record.get("latitude"),
                record.get("longitude"),
                record.get("stars"),
                record.get("review_count"),
                record.get("is_open"),
                categories,
            )
        )

        if len(rows) >= 1000:
            conn.executemany(
                """
                INSERT OR REPLACE INTO businesses (
                    business_id, name, address, city, state, postal_code,
                    latitude, longitude, stars, review_count, is_open, categories
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            rows.clear()

    if rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO businesses (
                business_id, name, address, city, state, postal_code,
                latitude, longitude, stars, review_count, is_open, categories
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    return business_ids


def load_reviews(
    conn: sqlite3.Connection,
    path: Path,
    valid_business_ids: Set[str],
    limit: Optional[int],
) -> int:
    rows: List[Tuple] = []
    inserted = 0
    for record in iter_json_lines(path, limit):
        if record["business_id"] not in valid_business_ids:
            continue
        rows.append(
            (
                record["review_id"],
                record["business_id"],
                record.get("stars"),
                record.get("useful"),
                record.get("funny"),
                record.get("cool"),
                record.get("text"),
                record.get("date"),
            )
        )

        if len(rows) >= 1000:
            conn.executemany(
                """
                INSERT OR REPLACE INTO reviews (
                    review_id, business_id, stars, useful, funny, cool, text, date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            inserted += len(rows)
            rows.clear()

    if rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO reviews (
                review_id, business_id, stars, useful, funny, cool, text, date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        inserted += len(rows)

    return inserted


def main() -> None:
    args = parse_args()
    conn = sqlite3.connect(args.db)
    try:
        ensure_schema(conn)
        business_ids = load_businesses(conn, args.business_json, args.business_limit)
        review_count = load_reviews(conn, args.review_json, business_ids, args.review_limit)
    finally:
        conn.commit()
        conn.close()

    print(f"Loaded {len(business_ids)} CA restaurants and {review_count} reviews into {args.db}")


if __name__ == "__main__":
    main()

