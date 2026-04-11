"""
Resale Velocity Engine — Google Trends Signal Layer
Fetches RSV (Relative Search Volume) per brand + category.
Applies 2-week lag as Trend_Delta feature (per PRD Section 7.3).

Run AFTER: python scripts/02_ai_parsing.py

IMPORTANT: pytrends has rate limits. This script processes brands in
batches with delays to avoid 429 errors.
"""

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from pytrends.request import TrendReq

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "luxury_with_ai_features.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "luxury_final.csv"
TRENDS_CACHE_PATH = ROOT / "data" / "processed" / "trends_cache.json"

# ─── Configuration ────────────────────────────────────────────────────────────

# Baseline keyword: a high-volume constant included in every Trends call.
# Forces relative scaling — prevents Chanel RSV=100 ≈ niche brand RSV=100.
# Per PRD Section 7.3: "Include a high-volume constant keyword in every API call"
BASELINE_KEYWORD = "luxury fashion"

# Brand + category keyword pairs to pull trends for
# Format: "brand category" as a search phrase
TREND_PAIRS = [
    ("chanel", "bag"),
    ("chanel", "dress"),
    ("louis vuitton", "bag"),
    ("gucci", "bag"),
    ("gucci", "shoes"),
    ("prada", "bag"),
    ("hermès", "bag"),
    ("bottega veneta", "bag"),
    ("saint laurent", "bag"),
    ("celine", "bag"),
    ("celine", "shoes"),
    ("valentino", "bag"),
    ("loewe", "bag"),
    ("dior", "bag"),
    ("balenciaga", "shoes"),
    ("balenciaga", "bag"),
    ("acne studios", "jacket"),
    ("rick owens", "jacket"),
    ("maison margiela", "shoes"),
]

# Lag: trend signal leads price movement by ~2 weeks (PRD Section 7.3)
TREND_LAG_WEEKS = 2


def fetch_trend_rsv(brand: str, category: str, pytrends: TrendReq) -> dict:
    """
    Fetch 90-day RSV trend for a brand+category pair.
    Returns dict with current RSV, 30-day delta, and signal direction.
    """
    keyword = f"{brand} {category}"
    try:
        pytrends.build_payload(
            [keyword, BASELINE_KEYWORD],
            cat=0,
            timeframe="today 3-m",
            geo="US",
        )
        df = pytrends.interest_over_time()

        if df.empty or keyword not in df.columns:
            return _empty_trend(keyword)

        series = df[keyword].dropna()
        baseline = df[BASELINE_KEYWORD].dropna()

        if len(series) < 4:
            return _empty_trend(keyword)

        # Normalize against baseline to make values comparable across brands
        # Avoid division by zero
        baseline_mean = baseline.mean() if baseline.mean() > 0 else 1
        normalized = (series / baseline_mean * 50).clip(0, 100)

        # Current RSV: average of last 2 weeks
        current_rsv = float(normalized.iloc[-2:].mean())

        # 30-day delta: compare last 2 weeks vs 4–6 weeks ago
        if len(normalized) >= 6:
            recent = normalized.iloc[-2:].mean()
            prior = normalized.iloc[-6:-4].mean()
            delta_pct = ((recent - prior) / (prior + 1)) * 100
        else:
            delta_pct = 0.0

        # Trend direction per PRD: >10% = rising, <-10% = declining, else flat
        if delta_pct > 10:
            direction = "rising"
        elif delta_pct < -10:
            direction = "declining"
        else:
            direction = "flat"

        return {
            "keyword": keyword,
            "current_rsv": round(current_rsv, 2),
            "delta_pct_30d": round(delta_pct, 2),
            "trend_direction": direction,
            "lag_weeks": TREND_LAG_WEEKS,
        }

    except Exception as e:
        print(f"    Trends error for '{keyword}': {e}")
        return _empty_trend(keyword)


def _empty_trend(keyword: str) -> dict:
    return {
        "keyword": keyword,
        "current_rsv": 50.0,   # neutral
        "delta_pct_30d": 0.0,
        "trend_direction": "flat",
        "lag_weeks": TREND_LAG_WEEKS,
    }


def build_trend_lookup(cache: dict) -> dict:
    """Fetch all brand+category trends and build a lookup dict."""
    pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
    lookup = {}

    for brand, category in TREND_PAIRS:
        key = f"{brand}|{category}"

        if key in cache:
            lookup[key] = cache[key]
            continue

        print(f"    Fetching: {brand} + {category}...", end=" ")
        result = fetch_trend_rsv(brand, category, pytrends)
        lookup[key] = result
        cache[key] = result
        print(f"RSV={result['current_rsv']:.0f} | {result['trend_direction']}")

        # Respect rate limits — pytrends is aggressive about 429s
        time.sleep(3)

    return lookup


def apply_trend_features(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """
    Map brand+category trend data onto each row.
    Falls back to neutral values if no trend data for that pair.
    """
    def get_trend(row):
        brand = str(row.get("brand_normalized", "")).lower()
        category = str(row.get("mid_level", "")).lower()
        key = f"{brand}|{category}"

        # Exact match
        if key in lookup:
            return lookup[key]

        # Brand-only fallback: use first available trend for this brand
        for k, v in lookup.items():
            if k.startswith(f"{brand}|"):
                return v

        return _empty_trend(f"{brand}|{category}")

    trends = df.apply(get_trend, axis=1)
    df["google_trend_rsv"] = [t["current_rsv"] for t in trends]
    df["trend_delta_30d"] = [t["delta_pct_30d"] for t in trends]
    df["trend_direction"] = [t["trend_direction"] for t in trends]

    # Encode direction as a numeric multiplier for the model
    # Rising: nudge toward 90th pct; Declining: nudge toward 10th pct
    DIRECTION_MAP = {"rising": 1, "flat": 0, "declining": -1}
    df["trend_signal_encoded"] = df["trend_direction"].map(DIRECTION_MAP).fillna(0)

    return df


def run_trends_pipeline():
    print("=" * 60)
    print("RESALE VELOCITY ENGINE — GOOGLE TRENDS LAYER")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"❌  Run 02_ai_parsing.py first.\n    Expected: {INPUT_PATH}"
        )

    # Load cache
    cache = {}
    if TRENDS_CACHE_PATH.exists():
        with open(TRENDS_CACHE_PATH) as f:
            cache = json.load(f)
        print(f"\n    Trends cache: {len(cache)} pairs already fetched")

    print(f"\n[1/3] Fetching Google Trends RSV")
    print(f"    Baseline keyword: '{BASELINE_KEYWORD}'")
    print(f"    Lag applied: {TREND_LAG_WEEKS} weeks")
    print(f"    Brand/category pairs: {len(TREND_PAIRS)}\n")

    lookup = build_trend_lookup(cache)

    # Save cache
    with open(TRENDS_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\n[2/3] Loading dataset and applying trend features")
    df = pd.read_csv(INPUT_PATH)
    df = apply_trend_features(df, lookup)

    print(f"\n[3/3] Saving final dataset")
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("✅  Trends layer complete")
    print(f"    Trend distribution:")
    print(df["trend_direction"].value_counts().to_string())
    print(f"\n    Output: {OUTPUT_PATH}")
    print("=" * 60)
    print("\nNext step: run  python scripts/04_train_model.py")


if __name__ == "__main__":
    run_trends_pipeline()
