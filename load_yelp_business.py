#!/usr/bin/env python3
"""Load Yelp business JSON into a normalized SQLite database."""

from __future__ import annotations

import argparse
import ast
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

BusinessRecord = Dict[str, Union[str, int, float, Dict, None]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Yelp business JSON into a SQLite database."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("yelp_academic_dataset_business.json"),
        help="Path to the Yelp business JSON file (default: %(default)s).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("yelp_business.sqlite"),
        help="Path to the SQLite database to write (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of JSON rows to import (useful for testing).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of rows per transaction batch (default: %(default)s).",
    )
    return parser.parse_args()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;

        CREATE TABLE IF NOT EXISTS businesses (
            business_id TEXT PRIMARY KEY,
            name TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            postal_code TEXT,
            latitude REAL,
            longitude REAL,
            stars REAL,
            review_count INTEGER,
            is_open INTEGER
        );

        CREATE TABLE IF NOT EXISTS business_attributes (
            business_id TEXT,
            attribute TEXT,
            value TEXT,
            FOREIGN KEY (business_id) REFERENCES businesses (business_id)
        );

        CREATE TABLE IF NOT EXISTS business_categories (
            business_id TEXT,
            category TEXT,
            FOREIGN KEY (business_id) REFERENCES businesses (business_id)
        );

        CREATE TABLE IF NOT EXISTS business_hours (
            business_id TEXT,
            day TEXT,
            opens TEXT,
            closes TEXT,
            FOREIGN KEY (business_id) REFERENCES businesses (business_id)
        );

        CREATE INDEX IF NOT EXISTS idx_business_categories_business
            ON business_categories (business_id);
        CREATE INDEX IF NOT EXISTS idx_business_attributes_business
            ON business_attributes (business_id);
        CREATE INDEX IF NOT EXISTS idx_business_hours_business
            ON business_hours (business_id);
        """
    )


def iter_json(path: Path, limit: Optional[int]) -> Iterator[BusinessRecord]:
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


def normalize_literal(value: Union[str, int, float, bool, None, Dict]) -> Union[str, int, float, bool, None, Dict]:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if stripped in {"True", "False"}:
        return stripped == "True"
    if stripped in {"None", "null"}:
        return None

    try:
        # Handle strings like "u'free'" or "{'garage': False}"
        literal = ast.literal_eval(stripped)
        return literal
    except (ValueError, SyntaxError):
        return value


def flatten_attributes(data: Optional[Dict]) -> List[Tuple[str, Optional[str]]]:
    if not data:
        return []

    rows: List[Tuple[str, Optional[str]]] = []

    def _walk(prefix: str, value) -> None:
        normalized = normalize_literal(value)
        if isinstance(normalized, dict):
            for key, inner in normalized.items():
                child_prefix = f"{prefix}.{key}" if prefix else key
                _walk(child_prefix, inner)
        else:
            if normalized is None:
                rows.append((prefix, None))
            elif isinstance(normalized, (bool, int, float)):
                rows.append((prefix, json.dumps(normalized)))
            else:
                rows.append((prefix, str(normalized)))

    for key, value in data.items():
        _walk(key, value)

    return rows


def split_categories(categories: Optional[str]) -> List[str]:
    if not categories:
        return []
    return [cat.strip() for cat in categories.split(",") if cat.strip()]


def decode_hours(hours: Optional[Dict[str, str]]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    if not hours:
        return []
    processed: List[Tuple[str, Optional[str], Optional[str]]] = []
    for day, window in hours.items():
        if not window:
            processed.append((day, None, None))
            continue
        if "-" in window:
            opens, closes = window.split("-", 1)
            processed.append((day, opens or None, closes or None))
        else:
            processed.append((day, window, None))
    return processed


def insert_batch(
    conn: sqlite3.Connection,
    business_rows: List[Tuple],
    attribute_rows: List[Tuple],
    category_rows: List[Tuple],
    hour_rows: List[Tuple],
) -> None:
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT OR REPLACE INTO businesses (
            business_id, name, address, city, state, postal_code,
            latitude, longitude, stars, review_count, is_open
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        business_rows,
    )
    if attribute_rows:
        cursor.executemany(
            "INSERT INTO business_attributes (business_id, attribute, value) VALUES (?, ?, ?)",
            attribute_rows,
        )
    if category_rows:
        cursor.executemany(
            "INSERT INTO business_categories (business_id, category) VALUES (?, ?)",
            category_rows,
        )
    if hour_rows:
        cursor.executemany(
            "INSERT INTO business_hours (business_id, day, opens, closes) VALUES (?, ?, ?, ?)",
            hour_rows,
        )
    conn.commit()


def load_data(
    conn: sqlite3.Connection,
    json_path: Path,
    limit: Optional[int],
    batch_size: int,
) -> int:
    business_batch: List[Tuple] = []
    attribute_batch: List[Tuple] = []
    category_batch: List[Tuple] = []
    hour_batch: List[Tuple] = []
    imported = 0

    for record in iter_json(json_path, limit):
        business_batch.append(
            (
                record.get("business_id"),
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
            )
        )

        for attr_name, attr_value in flatten_attributes(record.get("attributes")):
            attribute_batch.append((record.get("business_id"), attr_name, attr_value))

        for category in split_categories(record.get("categories")):
            category_batch.append((record.get("business_id"), category))

        for day, opens, closes in decode_hours(record.get("hours")):
            hour_batch.append((record.get("business_id"), day, opens, closes))

        imported += 1

        if imported % batch_size == 0:
            insert_batch(conn, business_batch, attribute_batch, category_batch, hour_batch)
            business_batch.clear()
            attribute_batch.clear()
            category_batch.clear()
            hour_batch.clear()

    if business_batch or attribute_batch or category_batch or hour_batch:
        insert_batch(conn, business_batch, attribute_batch, category_batch, hour_batch)

    return imported


def main() -> None:
    args = parse_args()

    if not args.json.exists():
        raise SystemExit(f"JSON file not found: {args.json}")

    conn = sqlite3.connect(args.db)
    try:
        ensure_schema(conn)
        row_count = load_data(conn, args.json, args.limit, args.batch_size)
    finally:
        conn.close()

    print(f"Imported {row_count} business rows into {args.db}")


if __name__ == "__main__":
    main()


