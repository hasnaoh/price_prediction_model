"""
Resale Velocity Engine — Data Pipeline v2
Step 1: Filter Mercari dataset to luxury brands
Step 2: Normalize fields (condition score, category)
Step 3: Engineer features — including V2-ready placeholders
Step 4: Export clean dataset ready for model training + Cox model

New in v2:
  - days_to_sell        : synthetic proxy from price/market ratio
  - price_to_market_ratio: listed price / category median (key Cox feature)
  - platform_tier       : placeholder for eBay/Grailed/Vestiaire source signal
  - event_observed      : censoring flag for Cox model (1 = sold, 0 = censored)

V2 SWAP: Replace synthetic days_to_sell with scraped eBay sold timestamps
V2 SWAP: Replace platform_tier placeholder with actual scrape source
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "train.tsv"
PROCESSED_PATH = ROOT / "data" / "processed" / "luxury_clean.csv"
STATS_PATH = ROOT / "data" / "processed" / "pipeline_stats.json"

# ─── Brand Configuration ──────────────────────────────────────────────────────
BRAND_TIERS = {
    "ultra_high": [
        "hermès", "hermes", "chanel", "rolex", "patek philippe",
        "van cleef", "cartier", "bottega veneta", "loro piana",
    ],
    "high": [
        "louis vuitton", "gucci", "prada", "dior", "christian dior",
        "saint laurent", "ysl", "valentino", "loewe", "givenchy",
        "balenciaga", "alexander mcqueen", "burberry", "fendi",
        "celine", "céline", "off-white", "jacquemus", "the row",
        "rick owens", "maison margiela", "acne studios",
    ],
    "contemporary": [
        "ganni", "staud", "aritzia", "nanushka", "toteme",
        "sandro", "maje", "rag & bone", "ba&sh", "ba and sh",
        "reformation", "frame", "theory", "vince",
    ],
}

BRAND_TIER_MAP = {}
for tier, brands in BRAND_TIERS.items():
    for brand in brands:
        BRAND_TIER_MAP[brand] = tier

BRAND_ORP_MEDIANS = {"ultra_high": 4500, "high": 1200, "contemporary": 350}

CONDITION_MAP = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize_brand(raw):
    if pd.isna(raw):
        return ""
    return " ".join(str(raw).lower().strip().split())

def assign_tier(normalized_brand):
    return BRAND_TIER_MAP.get(normalized_brand)

def normalize_category(cat):
    if pd.isna(cat):
        return {"top_level": None, "mid_level": None, "item_type": None}
    parts = [p.strip() for p in str(cat).split("/")]
    return {
        "top_level": parts[0] if len(parts) > 0 else None,
        "mid_level": parts[1] if len(parts) > 1 else None,
        "item_type": parts[2] if len(parts) > 2 else None,
    }

def compute_saturation_index(df):
    bucket = df["brand_tier"].astype(str) + "|" + df["mid_level"].astype(str)
    counts = bucket.map(bucket.value_counts())
    log_counts = np.log1p(counts)
    lo, hi = log_counts.min(), log_counts.max()
    if hi == lo:
        return pd.Series(1.0, index=df.index)
    return 0.1 + (log_counts - lo) / (hi - lo) * 2.9


def compute_price_to_market_ratio(df):
    """
    price_to_market_ratio = item price / median price for that brand+category bucket.

    < 0.8  = priced aggressively below market (faster sell-through expected)
    0.8-1.2 = at market (normal velocity)
    > 1.2  = priced above market (slower sell-through, markdown risk)

    This is the primary pricing signal for the Cox model.
    V2 SWAP: Use live eBay/Grailed median instead of within-dataset median.
    """
    bucket = df["brand_normalized"].astype(str) + "|" + df["mid_level"].astype(str)
    bucket_median = df.groupby(bucket)["price"].transform("median")
    ratio = df["price"] / bucket_median.replace(0, np.nan)
    return ratio.fillna(1.0).clip(0.1, 5.0)


def synthesize_days_to_sell(df):
    """
    Synthetic days_to_sell proxy for V1.

    Logic grounded in resale market dynamics (PRD Section 2.2):
    - Items priced at/below market comp sell faster
    - Higher brand tier = faster baseline velocity (more demand)
    - Worse condition = slower sell
    - Accessories included = faster sell (buyers prefer complete sets)

    Formula encodes the J-curve relationship:
      aggressive pricing -> fast sell
      market pricing     -> moderate sell
      above market       -> slow sell + markdown spiral risk

    V2 SWAP: Replace with actual (listing_timestamp - sold_timestamp) from eBay scrape.
    All downstream Cox model code is already wired to use this column name.
    """
    # Base days by brand tier (ultra_high has deepest demand pool)
    tier_base = df["brand_tier"].map({"ultra_high": 14, "high": 21, "contemporary": 35}).fillna(21)

    # Price ratio effect: below market sells faster, above market sells slower
    # Encoding the J-curve: ratio=0.7 -> ~0.6x days, ratio=1.3 -> ~2x days
    price_multiplier = df["price_to_market_ratio"].apply(
        lambda r: max(0.4, min(3.0, 0.5 + r ** 1.8))
    )

    # Condition effect
    condition_multiplier = df["condition_score"].apply(
        lambda c: {10: 0.8, 8: 1.0, 6: 1.2, 4: 1.6, 2: 2.2}.get(c, 1.0)
    )

    # Accessories accelerate sale
    accessory_multiplier = df["accessories_flag"].apply(lambda a: 0.85 if a == 1 else 1.0)

    # Compute and add realistic noise
    raw_days = tier_base * price_multiplier * condition_multiplier * accessory_multiplier
    noise = np.random.normal(1.0, 0.15, size=len(df))
    days = (raw_days * noise).clip(1, 180).round().astype(int)

    return days


def compute_censoring_flag(df):
    """
    event_observed: Cox model censoring flag.
    1 = item sold (event observed)
    0 = item did not sell / still listed (censored — we know it survived X days)

    V1 proxy: assume all Mercari training items eventually sold (they're in the
    sold price dataset). Mark ~15% as censored to reflect real-world unsold rate.

    V2 SWAP: Use actual sold/unsold flag from eBay scrape.
    Items still listed after 90 days are censored observations.
    """
    np.random.seed(42)
    # 85% sold, 15% censored — approximates TRR's reported sell-through rate
    return np.random.choice([1, 0], size=len(df), p=[0.85, 0.15])


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(sample=None):
    print("=" * 60)
    print("RESALE VELOCITY ENGINE — DATA PIPELINE v2")
    print("=" * 60)

    print(f"\n[1/6] Loading raw data from {RAW_PATH}")
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"\n  File not found: {RAW_PATH}\n\n"
            "  Download train.tsv from:\n"
            "  https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data\n"
            "  Place at: data/raw/train.tsv\n"
        )

    df = pd.read_csv(
        RAW_PATH, sep="\t", nrows=sample,
        dtype={"train_id": int, "name": str, "item_condition_id": int,
               "category_name": str, "brand_name": str, "price": float,
               "shipping": int, "item_description": str},
    )
    print(f"    Loaded {len(df):,} rows")

    print("\n[2/6] Filtering to luxury brands")
    df["brand_normalized"] = df["brand_name"].apply(normalize_brand)
    df["brand_tier"] = df["brand_normalized"].apply(assign_tier)
    before = len(df)
    df = df[df["brand_tier"].notna()].copy()
    after = len(df)
    print(f"    {before:,} -> {after:,} rows ({after/before*100:.1f}% retained as luxury)")

    print("\n[3/6] Cleaning price data")
    df = df[df["price"] > 0].copy()
    p99 = df["price"].quantile(0.99)
    df = df[df["price"] <= p99].copy()
    print(f"    Price range: ${df['price'].min():.0f} - ${df['price'].max():.0f}")
    print(f"    Median: ${df['price'].median():.0f}")

    print("\n[4/6] Engineering features")

    # Standard features
    df["condition_score"] = df["item_condition_id"].map(CONDITION_MAP).fillna(5)
    print("    condition_score")

    cat_parsed = df["category_name"].apply(normalize_category)
    df["top_level"] = cat_parsed.apply(lambda x: x["top_level"])
    df["mid_level"] = cat_parsed.apply(lambda x: x["mid_level"])
    df["item_type"] = cat_parsed.apply(lambda x: x["item_type"])
    print("    category hierarchy")

    df["orp_proxy"] = df["brand_tier"].map(BRAND_ORP_MEDIANS)
    df["msrp_ratio"] = df["price"] / df["orp_proxy"]
    print("    msrp_ratio")

    df["saturation_index"] = compute_saturation_index(df)
    print("    saturation_index")

    df["shipping_included"] = df["shipping"].astype(int)
    df["description_length"] = df["item_description"].str.len().fillna(0)
    df["has_description"] = (
        df["item_description"].notna()
        & (df["item_description"] != "No description yet")
        & (df["item_description"].str.len() > 10)
    ).astype(int)
    print("    shipping_included, description_length, has_description")

    # AI feature placeholders (populated by 02_ai_parsing.py)
    df["archive_score"] = 0
    df["accessories_flag"] = 0
    df["collab_flag"] = 0
    print("    AI placeholders (archive_score, accessories_flag, collab_flag)")

    # ── NEW V2-READY FEATURES ──────────────────────────────────────

    # price_to_market_ratio: key Cox model input + pricing signal
    df["price_to_market_ratio"] = compute_price_to_market_ratio(df)
    print("    price_to_market_ratio (price / category median)")

    # platform_tier: source platform signal
    # V2 SWAP: set from actual scrape source (vestiaire=3, grailed=2, ebay=1, mercari=0)
    df["platform_tier"] = 0  # 0 = Mercari (V1 proxy)
    print("    platform_tier (placeholder — V2 SWAP: set from scrape source)")

    # days_to_sell: Cox model target label
    # depends on accessories_flag, so must come after AI placeholders
    df["days_to_sell"] = synthesize_days_to_sell(df)
    print("    days_to_sell (synthetic proxy — V2 SWAP: actual listing/sold timestamps)")

    # event_observed: Cox censoring flag
    df["event_observed"] = compute_censoring_flag(df)
    print("    event_observed (censoring flag for Cox model)")

    print("\n[5/6] Selecting final feature set")
    FINAL_COLS = [
        "train_id", "name", "item_description",
        # Target labels
        "price", "days_to_sell", "event_observed",
        # Core features
        "brand_normalized", "brand_tier", "condition_score",
        "top_level", "mid_level", "item_type",
        "msrp_ratio", "saturation_index", "shipping_included",
        "description_length", "has_description",
        # AI-parsed (populated in step 02)
        "archive_score", "accessories_flag", "collab_flag",
        # V2-ready features
        "price_to_market_ratio", "platform_tier",
        # google_trend features added in step 03
    ]
    df_final = df[FINAL_COLS].copy()
    print(f"    {len(df_final):,} rows x {len(df_final.columns)} columns")

    print(f"\n[6/6] Saving to {PROCESSED_PATH}")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(PROCESSED_PATH, index=False)

    stats = {
        "total_raw_rows": before,
        "luxury_rows": after,
        "retention_pct": round(after/before*100, 2),
        "price_min": round(df_final["price"].min(), 2),
        "price_max": round(df_final["price"].max(), 2),
        "price_median": round(df_final["price"].median(), 2),
        "days_to_sell_median": round(df_final["days_to_sell"].median(), 1),
        "event_observed_rate": round(df_final["event_observed"].mean(), 3),
        "brand_tier_counts": df_final["brand_tier"].value_counts().to_dict(),
        "columns": list(df_final.columns),
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("Pipeline complete")
    print(f"    Output:  {PROCESSED_PATH}")
    print(f"    Columns: {len(df_final.columns)}")
    print(f"    Median days_to_sell: {stats['days_to_sell_median']} days")
    print(f"    Event observed rate: {stats['event_observed_rate']*100:.1f}%")
    print("=" * 60)
    print("\nNext step: python scripts/02_ai_parsing.py")

    return df_final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    run_pipeline(sample=args.sample)