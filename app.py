from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, render_template, request

DB_PATH = Path("ny_restaurants.sqlite")

app = Flask(__name__)


def get_connection() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise RuntimeError(
            f"Database {DB_PATH} not found. Run setup_db.py before starting the server."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_reviews(conn: sqlite3.Connection, business_id: str, order: str) -> List[Dict]:
    if order not in {"DESC", "ASC"}:
        raise ValueError("order must be 'ASC' or 'DESC'")
    rows = conn.execute(
        f"""
        SELECT review_id, stars, text, date
        FROM reviews
        WHERE business_id = ?
        ORDER BY stars {order}, date DESC
        LIMIT 5
        """,
        (business_id,),
    ).fetchall()
    return [dict(row) for row in rows]


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/search")
def search_business():
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query parameter 'query' is required."}), 400

    conn = get_connection()
    try:
        business = conn.execute(
            """
            SELECT business_id, name, address, city, state, postal_code,
                   latitude, longitude, stars, review_count, categories
            FROM businesses
            WHERE name LIKE ?
            ORDER BY review_count DESC
            LIMIT 1
            """,
            (f"%{query}%",),
        ).fetchone()

        if business is None:
            return (
                jsonify({"error": f"No New York restaurant found matching '{query}'."}),
                404,
            )

        business_dict = dict(business)
        business_dict["categories"] = [
            c.strip() for c in (business_dict.get("categories") or "").split(",") if c.strip()
        ]

        top_positive = fetch_reviews(conn, business["business_id"], "DESC")
        top_negative = fetch_reviews(conn, business["business_id"], "ASC")
    finally:
        conn.close()

    return jsonify(
        {
            "business": business_dict,
            "top_positive_reviews": top_positive,
            "top_negative_reviews": top_negative,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

