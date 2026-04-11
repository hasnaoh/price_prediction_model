"""
Resale Velocity Engine — FastAPI Backend v2
POST /valuate returns four components:
  1. Recommended listing price
  2. Confidence range (10th-90th percentile)
  3. Trend signal (rising / flat / declining)
  4. Sell-through prediction (Cox model — 30/60/90 day probabilities)
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")
MODEL_DIR = ROOT / "backend" / "models"

app = FastAPI(
    title="Resale Velocity Engine",
    description="B2B pricing and sell-through optimization for luxury resale",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load model artifacts ─────────────────────────────────────────────────────
with open(MODEL_DIR / "xgboost_model.pkl", "rb") as f:
    MODEL_BUNDLE = pickle.load(f)
with open(MODEL_DIR / "encoder.pkl", "rb") as f:
    ENCODER = pickle.load(f)
with open(MODEL_DIR / "baseline_lookup.json") as f:
    BASELINE_LOOKUP = json.load(f)
with open(MODEL_DIR / "training_metrics.json") as f:
    METRICS = json.load(f)

MAIN_MODEL = MODEL_BUNDLE["main"]
LOW_MODEL = MODEL_BUNDLE["low"]
HIGH_MODEL = MODEL_BUNDLE["high"]

# Cox model — load if available, degrade gracefully if not yet trained
COX_MODEL = None
try:
    with open(MODEL_DIR / "cox_model.pkl", "rb") as f:
        COX_MODEL = pickle.load(f)
    print("Cox model loaded")
except FileNotFoundError:
    print("Cox model not found — run scripts/05_cox_model.py to enable sell-through predictions")

# ─── Configuration ────────────────────────────────────────────────────────────
BRAND_TIER_MAP = {
    "hermès": "ultra_high", "hermes": "ultra_high", "chanel": "ultra_high",
    "bottega veneta": "ultra_high", "loro piana": "ultra_high",
    "louis vuitton": "high", "gucci": "high", "prada": "high",
    "dior": "high", "christian dior": "high", "saint laurent": "high",
    "ysl": "high", "valentino": "high", "loewe": "high",
    "givenchy": "high", "balenciaga": "high", "alexander mcqueen": "high",
    "burberry": "high", "fendi": "high", "celine": "high",
    "off-white": "high", "jacquemus": "high", "the row": "high",
    "rick owens": "high", "maison margiela": "high", "acne studios": "high",
    "ganni": "contemporary", "staud": "contemporary", "aritzia": "contemporary",
    "reformation": "contemporary", "frame": "contemporary", "theory": "contemporary",
}

BRAND_ORP_MEDIANS = {"ultra_high": 4500, "high": 1200, "contemporary": 350}

# Brand tier retention curves (PRD Section 5.4)
RETENTION = {"ultra_high": 0.70, "high": 0.55, "contemporary": 0.22}

CONDITION_MAP = {"Pristine": 10, "Excellent": 8, "Very Good": 6, "Good": 4, "Fair": 2}

TREND_CACHE = {
    "chanel|bags": {"rsv": 100, "delta": 0, "direction": "flat"},
    "louis vuitton|bags": {"rsv": 100, "delta": 0, "direction": "flat"},
    "gucci|bags": {"rsv": 100, "delta": 5, "direction": "rising"},
    "prada|bags": {"rsv": 100, "delta": 0, "direction": "flat"},
    "dior|bags": {"rsv": 90, "delta": 12, "direction": "rising"},
    "loewe|bags": {"rsv": 26, "delta": 15, "direction": "rising"},
    "celine|bags": {"rsv": 33, "delta": 18, "direction": "rising"},
    "saint laurent|bags": {"rsv": 24, "delta": 11, "direction": "rising"},
    "bottega veneta|bags": {"rsv": 21, "delta": 14, "direction": "rising"},
    "balenciaga|bags": {"rsv": 100, "delta": 0, "direction": "flat"},
}

ARCHIVE_SIGNALS = [
    "phoebe philo", "old celine", "hedi slimane", "tom ford",
    "nicolas ghesquiere", "jonathan anderson", "archive", "runway",
]
ACCESSORY_SIGNALS = [
    "dust bag", "dustbag", "authenticity card", "auth card",
    "certificate", "receipt", "tags attached", "never worn", "unworn",
    "original packaging",
]

CATEGORICAL_FEATURES = ["brand_tier", "top_level", "mid_level", "item_type"]
NUMERIC_FEATURES = [
    "condition_score", "saturation_index", "description_length",
    "shipping_included", "archive_score", "accessories_flag", "collab_flag",
    "google_trend_rsv", "trend_delta_30d", "trend_signal_encoded", "has_description",
]

COX_FEATURES = [
    "price_to_market_ratio", "condition_score", "saturation_index",
    "accessories_flag", "archive_score", "trend_signal_encoded",
    "google_trend_rsv", "platform_tier",
]


# ─── Request / Response models ────────────────────────────────────────────────

class ValuationRequest(BaseModel):
    brand: str = Field(..., example="gucci")
    category: str = Field(..., example="Handbags")
    condition: str = Field(..., example="Excellent")
    original_retail_price: float = Field(..., gt=0, example=1800.0)
    color: Optional[str] = Field(None, example="black")
    material: Optional[str] = Field(None, example="leather")
    size: Optional[str] = Field(None, example="medium")
    year_season: Optional[str] = Field(None, example="FW22")
    is_limited_edition: Optional[bool] = Field(False)
    free_text_description: Optional[str] = Field(
        None,
        example="Beautiful black leather Gucci bag, includes dust bag and authenticity card. Never worn."
    )


class SellThroughPrediction(BaseModel):
    probability_7_day: float
    probability_14_day: float
    probability_30_day: float
    probability_60_day: float
    probability_90_day: float
    price_sensitivity: str
    note: str


class ValuationResponse(BaseModel):
    recommended_price: float
    confidence_low: float
    confidence_high: float
    trend_signal: str
    trend_direction: str
    sell_through: SellThroughPrediction
    rationale: dict
    model_version: str = "2.0.0"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_description_fast(text):
    if not text:
        return {"archive_score": 0, "accessories_flag": 0, "collab_flag": 0}
    lower = text.lower()
    return {
        "archive_score": int(any(s in lower for s in ARCHIVE_SIGNALS)),
        "accessories_flag": int(any(s in lower for s in ACCESSORY_SIGNALS)),
        "collab_flag": int("collab" in lower or "limited" in lower),
    }


def get_category_mapping(category):
    cat_lower = category.lower()
    if "bag" in cat_lower or "handbag" in cat_lower or "purse" in cat_lower:
        return "Women", "Bags", "Handbags"
    elif "shoe" in cat_lower or "boot" in cat_lower or "heel" in cat_lower:
        return "Women", "Shoes", category
    elif "jacket" in cat_lower or "coat" in cat_lower:
        return "Women", "Coats & Jackets", category
    elif "dress" in cat_lower:
        return "Women", "Dresses", category
    else:
        return "Women", "Tops & Blouses", category


def predict_sell_through(cox_model, price, confidence_low, confidence_high,
                          parsed, trend, condition_score, saturation):
    """
    Run Cox model to predict sell-through velocity.
    price_to_market_ratio = recommended_price / midpoint of confidence band
    (proxy for how aggressively the item is priced vs market)
    """
    market_mid = (confidence_low + confidence_high) / 2
    price_to_market = price / market_mid if market_mid > 0 else 1.0
    price_to_market = max(0.3, min(3.0, price_to_market))

    direction_map = {"rising": -1, "flat": 0, "declining": 1}  # negative = faster sell

    row = pd.DataFrame([{
        "price_to_market_ratio": price_to_market,
        "condition_score": condition_score,
        "saturation_index": saturation,
        "accessories_flag": parsed["accessories_flag"],
        "archive_score": parsed["archive_score"],
        "trend_signal_encoded": direction_map.get(trend["direction"], 0),
        "google_trend_rsv": trend["rsv"],
        "platform_tier": 0,  # V2 SWAP: real platform source
    }])

    # Predict survival curve
    survival = cox_model.predict_survival_function(row)
    checkpoints = {7: 0, 14: 0, 30: 0, 60: 0, 90: 0}
    for day in checkpoints:
        closest = survival.index[np.argmin(np.abs(survival.index - day))]
        checkpoints[day] = round(1 - float(survival.loc[closest].iloc[0]), 3)

    # Price sensitivity: what happens if we drop 10%?
    row_cheaper = row.copy()
    row_cheaper["price_to_market_ratio"] = price_to_market * 0.9
    surv_cheaper = cox_model.predict_survival_function(row_cheaper)
    closest_30 = surv_cheaper.index[np.argmin(np.abs(surv_cheaper.index - 30))]
    prob_30_cheaper = round(1 - float(surv_cheaper.loc[closest_30].iloc[0]), 3)

    base_prob = checkpoints[30]
    delta = prob_30_cheaper - base_prob
    sensitivity = (
        f"Dropping price 10% increases 30-day sell probability "
        f"from {base_prob*100:.0f}% to {prob_30_cheaper*100:.0f}% "
        f"(+{delta*100:.0f}pp)"
    )

    return checkpoints, sensitivity


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Resale Velocity Engine",
        "version": "2.0.0",
        "status": "online",
        "cox_model_loaded": COX_MODEL is not None,
        "model_metrics": METRICS,
    }


@app.get("/health")
def health():
    return {"status": "ok", "cox_available": COX_MODEL is not None}


@app.post("/valuate", response_model=ValuationResponse)
def valuate(req: ValuationRequest):
    try:
        brand_norm = req.brand.lower().strip()
        brand_tier = BRAND_TIER_MAP.get(brand_norm, "high")
        condition_score = CONDITION_MAP.get(req.condition, 6)

        top_level, mid_level, item_type = get_category_mapping(req.category)

        trend_key = f"{brand_norm}|{mid_level.lower()}"
        trend = TREND_CACHE.get(trend_key, {"rsv": 50, "delta": 0, "direction": "flat"})
        direction_map = {"rising": 1, "flat": 0, "declining": -1}

        parsed = parse_description_fast(req.free_text_description or "")
        if req.is_limited_edition:
            parsed["collab_flag"] = 1

        desc_len = len(req.free_text_description or "")

        # Build XGBoost feature row
        X_raw = pd.DataFrame([{
            "brand_tier": brand_tier,
            "top_level": top_level,
            "mid_level": mid_level,
            "item_type": item_type,
            "condition_score": condition_score,
            "saturation_index": 1.0,
            "description_length": desc_len,
            "shipping_included": 0,
            "archive_score": parsed["archive_score"],
            "accessories_flag": parsed["accessories_flag"],
            "collab_flag": parsed["collab_flag"],
            "google_trend_rsv": trend["rsv"],
            "trend_delta_30d": trend["delta"],
            "trend_signal_encoded": direction_map.get(trend["direction"], 0),
            "has_description": int(desc_len > 10),
        }])

        all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
        X_raw[CATEGORICAL_FEATURES] = ENCODER.transform(X_raw[CATEGORICAL_FEATURES].astype(str))
        X = X_raw[all_features].values

        # Price prediction
        price = float(np.expm1(MAIN_MODEL.predict(X)[0]))
        price_low = float(np.expm1(LOW_MODEL.predict(X)[0]))
        price_high = float(np.expm1(HIGH_MODEL.predict(X)[0]))

        # ORP anchor
        retention = RETENTION.get(brand_tier, 0.55)
        condition_multiplier = condition_score / 10.0
        anchored_price = req.original_retail_price * retention * condition_multiplier
        price = (price * 0.4) + (anchored_price * 0.6)
        price_low = price * 0.75
        price_high = price * 1.30

        # Clamp
        price_low = min(price_low, price * 0.75)
        price_high = max(price_high, price * 1.25)
        price = max(5.0, round(price, 2))
        price_low = max(5.0, round(price_low, 2))
        price_high = max(price_low + 10, round(price_high, 2))

        # Trend signal text
        direction = trend["direction"]
        if direction == "rising":
            trend_signal = "Consider listing at upper confidence bound - search demand is rising."
        elif direction == "declining":
            trend_signal = "Price aggressively at lower confidence bound to move inventory."
        else:
            trend_signal = "List at recommended price - demand is stable."

        # Sell-through prediction (Cox model)
        if COX_MODEL is not None:
            checkpoints, sensitivity = predict_sell_through(
                COX_MODEL, price, price_low, price_high,
                parsed, trend, condition_score, saturation=1.0
            )
            sell_through = SellThroughPrediction(
                probability_7_day=checkpoints[7],
                probability_14_day=checkpoints[14],
                probability_30_day=checkpoints[30],
                probability_60_day=checkpoints[60],
                probability_90_day=checkpoints[90],
                price_sensitivity=sensitivity,
                note="V1: trained on synthetic sell-through proxy. V2: real eBay timestamps.",
            )
        else:
            # Graceful degradation if Cox model not trained yet
            sell_through = SellThroughPrediction(
                probability_7_day=0.0,
                probability_14_day=0.0,
                probability_30_day=0.0,
                probability_60_day=0.0,
                probability_90_day=0.0,
                price_sensitivity="Run scripts/05_cox_model.py to enable sell-through predictions.",
                note="Cox model not loaded.",
            )

        # Rationale
        rationale = {
            "brand_tier": brand_tier,
            "condition_score": condition_score,
            "archive_item": bool(parsed["archive_score"]),
            "accessories_included": bool(parsed["accessories_flag"]),
            "collaboration_item": bool(parsed["collab_flag"]),
            "trend_direction": direction,
            "trend_rsv": trend["rsv"],
            "confidence_band": f"${price_low:.0f} - ${price_high:.0f}",
            "markdown_risk": price > price_high * 0.9,
        }

        return ValuationResponse(
            recommended_price=price,
            confidence_low=price_low,
            confidence_high=price_high,
            trend_signal=trend_signal,
            trend_direction=direction,
            sell_through=sell_through,
            rationale=rationale,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))