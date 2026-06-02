"""
Resale Velocity Engine — Brand Floor Correction
================================================
Applies domain-knowledge price corrections as a post-processing layer
on top of XGBoost predictions.

WHY THIS EXISTS:
  The XGBoost model is trained on Mercari data, which structurally
  underrepresents ultra-luxury transaction prices. A Hermès Birkin
  on Mercari is rare, mis-categorized, or fake — so the model has
  no signal for what Hermès actually means at market.

  This module encodes that domain knowledge as an explicit, auditable
  prior — the same logic a seasoned consignment buyer applies manually.

ARCHITECTURE:
  Three tiers, three behaviors:

  ULTRA-HIGH (Hermès, Chanel, Rolex, etc.)
    → Model A: market base discounting
    → P_resale = (P_retail × retention_rate) × condition_multiplier × accessories_premium
    → XGBoost output is overridden entirely — Mercari data is too far from reality
    → Source: Rebag Clair Report 2025, Bernstein Research

  HIGH (Louis Vuitton, Gucci, Prada, etc.)
    → Soft floor only: if XGBoost comes in below floor, nudge up
    → If XGBoost is above floor, trust it — Mercari has real signal here
    → Blend: 60% XGBoost, 40% floor when XGBoost is below floor

  CONTEMPORARY (Coach, Ganni, Reformation, etc.)
    → Passthrough: XGBoost runs unmodified
    → Mercari data is actually representative for this tier

V2 NOTE:
  Replace BRAND_RETENTION_RATES with live secondary market comp medians
  pulled from eBay/Grailed scrape. The formula stays identical — only
  the retention anchor changes from a static prior to a dynamic comp.

DATA SOURCES:
  - Rebag Clair Report 2025 (brand retention rates)
  - Bernstein Research 2025 (Birkin premium compression 2.2x → 1.4x retail)
  - Primary market retail pricing (brand websites, 2025)
  - Research synthesis: condition multiplier compression for ultra-luxury
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ─── Brand Retention Rates ────────────────────────────────────────────────────
# Source: Rebag Clair Report 2025
# Definition: average secondary market price / current retail price
# Note: >1.0 means the brand trades ABOVE retail on secondary market
#
# Hermès 1.38 means a $10,000 retail Hermès item sells for ~$13,800 on resale
# Chanel 0.92 means a $11,300 Classic Flap sells for ~$10,400 on resale
#
# V2 SWAP: Replace these static rates with live 90-day rolling median
# from eBay/Grailed sold listings, refreshed weekly.

BRAND_RETENTION_RATES: dict[str, float] = {
    # Ultra-high tier — appreciating assets
    "hermès":           1.38,
    "hermes":           1.38,
    "chanel":           0.92,
    "rolex":            1.04,
    "goyard":           1.32,
    "the row":          0.97,

    # Ultra-high tier — high retention, no appreciation
    # Jewelry: anchored by precious metal spot price floor
    # Apply brand equity premium on top of material value
    "van cleef":        0.87,   # Alhambra: 80-90% retention, avg 87%
    "van cleef & arpels": 0.87,
    "cartier":          0.87,   # Portfolio avg 79-97%, weighted to 87%
    "patek philippe":   1.10,   # Waitlist-driven; trades above retail
    "bottega veneta":   0.75,   # Strong but not scarce; no waitlist premium

    # Loro Piana: quiet luxury RTW — DOES NOT appreciate
    # Sized item: requires liquidity markdown vs handbag floor logic
    # See SIZED_RTW_BRANDS below for special handling
    "loro piana":       0.68,   # 60-75% retention, sized discount applied separately

    # High tier — solid retention, Mercari has partial signal
    "louis vuitton":    0.88,   # Core canvas: 85-92% retention; avg 88%
    "gucci":            0.72,
    "prada":            0.74,
    "dior":             0.76,
    "christian dior":   0.76,
    "saint laurent":    0.70,
    "ysl":              0.70,
    "valentino":        0.65,
    "loewe":            0.68,
    "givenchy":         0.60,
    "balenciaga":       0.58,
    "alexander mcqueen":0.62,
    "burberry":         0.60,
    "fendi":            0.65,
    "celine":           0.68,
    "céline":           0.68,
    "off-white":        0.50,   # Trend-dependent; declining post-Abloh
    "jacquemus":        0.55,
    "rick owens":       0.60,
    "maison margiela":  0.58,
    "acne studios":     0.52,
}

# Brands where sizing friction requires an additional liquidity markdown
# to hit 30-day sell-through. Applied on top of retention-based floor.
# Source: RTW structural analysis — sized items lose 10-20% on outlier sizes
SIZED_RTW_BRANDS: set[str] = {"loro piana"}

# Soft floor thresholds for HIGH tier brands.
# If XGBoost predicts below this % of retention-implied floor,
# we blend rather than fully override.
HIGH_TIER_FLOOR_BLEND = 0.60   # XGBoost weight when below floor
HIGH_TIER_XGBOOST_BLEND = 0.40  # Retention floor weight when below floor


# ─── Condition Multipliers ────────────────────────────────────────────────────
# Source: Research synthesis (Clair Report 2025 + authenticator grade standards)
#
# CRITICAL INSIGHT: Ultra-luxury has a COMPRESSED condition band vs contemporary.
# The difference between Excellent and Very Good is only 15-25% for Hermès —
# vs 40-50% for a Coach bag. This reflects deeper buyer demand at every grade.
#
# Model A discount = discount FROM market base (not from retail)
# i.e. Excellent Hermès = market_base × 0.95 (only 5% off market)
#
# condition_score mapping (from CONDITION_MAP in main.py):
#   Pristine  = 10
#   Excellent = 8
#   Very Good = 6
#   Good      = 4
#   Fair      = 2

CONDITION_MULTIPLIERS: dict[str, dict[int, float]] = {
    "ultra_high": {
        10: 1.05,   # Pristine: rare, commands small premium over market base
        8:  0.95,   # Excellent: 0-10% off market base (research: 0-10% discount)
        6:  0.82,   # Very Good: 15-25% off market base; midpoint = 82%
        4:  0.60,   # Good: 30-50% off market base; midpoint = 60%
        2:  0.42,   # Fair: 40-60% off market base; midpoint ~42%
    },
    "high": {
        10: 1.02,
        8:  0.90,
        6:  0.75,
        4:  0.55,
        2:  0.35,
    },
    "contemporary": {
        # Passthrough tier — these multipliers are not used (XGBoost runs unmodified)
        # Included for completeness and potential future use
        10: 1.00,
        8:  0.82,
        6:  0.65,
        4:  0.45,
        2:  0.28,
    },
}

# Default condition multiplier if score not found
CONDITION_MULTIPLIER_DEFAULT = 0.75


# ─── Accessories Premium ──────────────────────────────────────────────────────
# Source: Research synthesis
# Full set = original box, dust bag, authenticity card, receipt
# Note: jewelry full set premium is higher (certificate is authentication proof)

ACCESSORIES_PREMIUM: dict[str, float] = {
    "handbag":  0.075,   # +5-10% for full set; midpoint 7.5%
    "jewelry":  0.20,    # +15-25% for original cert; midpoint 20%
    "watch":    0.12,    # +10-15% for box & papers
    "default":  0.075,
}

# ─── Core Correction Logic ────────────────────────────────────────────────────

@dataclass
class FloorCorrectionResult:
    corrected_price: float
    confidence_low: float
    confidence_high: float
    floor_applied: bool
    floor_logic: str          # Human-readable explanation for rationale block
    retention_rate: float
    market_base: float        # P_retail × retention_rate (Model A step 1)


def get_accessories_premium(category: str, has_accessories: bool) -> float:
    """Return the accessories premium multiplier for a given category."""
    if not has_accessories:
        return 1.0
    cat = category.lower()
    if any(w in cat for w in ["bag", "handbag", "purse", "clutch"]):
        return 1.0 + ACCESSORIES_PREMIUM["handbag"]
    elif any(w in cat for w in ["jewelry", "bracelet", "necklace", "ring", "earring"]):
        return 1.0 + ACCESSORIES_PREMIUM["jewelry"]
    elif any(w in cat for w in ["watch"]):
        return 1.0 + ACCESSORIES_PREMIUM["watch"]
    return 1.0 + ACCESSORIES_PREMIUM["default"]


def apply_sized_rtw_markdown(price: float, brand: str, size: Optional[str]) -> tuple[float, bool]:
    """
    Apply liquidity markdown for sized RTW brands (e.g. Loro Piana).
    Outlier sizes (XXS, XL, XXL) get a steeper markdown to hit 30-day clearance.
    Standard sizes get a base markdown reflecting the smaller buyer pool vs handbags.

    Returns (adjusted_price, markdown_applied).
    """
    if brand.lower() not in SIZED_RTW_BRANDS:
        return price, False

    outlier_sizes = {"xxs", "xs/xxs", "xl", "xxl", "1x", "2x", "3x", "4x", "14", "16", "18"}
    size_str = (size or "").lower().strip()

    if size_str in outlier_sizes:
        # 15% markdown for outlier sizes — deeper buyer pool friction
        return round(price * 0.85, 2), True
    else:
        # 8% base markdown for standard sized RTW vs handbag liquidity
        return round(price * 0.92, 2), True


def apply_brand_floor(
    xgboost_price: float,
    xgboost_price_low: float,
    xgboost_price_high: float,
    brand: str,
    brand_tier: str,
    condition_score: int,
    original_retail_price: float,
    category: str,
    has_accessories: bool,
    size: Optional[str] = None,
) -> FloorCorrectionResult:
    """
    Main entry point. Apply brand floor correction based on tier.

    ULTRA-HIGH: Full Model A override
    HIGH:       Soft floor blend
    CONTEMPORARY: Passthrough

    Parameters
    ----------
    xgboost_price : float
        Raw price from XGBoost + ORP anchor blend (main.py line 303)
    brand : str
        Normalized brand name (lowercase)
    brand_tier : str
        One of: "ultra_high", "high", "contemporary"
    condition_score : int
        Numeric condition score (2, 4, 6, 8, or 10)
    original_retail_price : float
        ORP supplied by user in the /valuate request
    category : str
        Item category string from request
    has_accessories : bool
        Whether full set / accessories are included
    size : str, optional
        Item size, used for RTW liquidity markdown
    """
    brand_norm = brand.lower().strip()
    retention = BRAND_RETENTION_RATES.get(brand_norm)

    # ── CONTEMPORARY: pure passthrough ────────────────────────────
    if brand_tier == "contemporary" or retention is None:
        return FloorCorrectionResult(
            corrected_price=round(xgboost_price, 2),
            confidence_low=round(xgboost_price_low, 2),
            confidence_high=round(xgboost_price_high, 2),
            floor_applied=False,
            floor_logic=(
                "Contemporary tier: XGBoost prediction used directly. "
                "Mercari training data is representative for this brand."
            ),
            retention_rate=0.0,
            market_base=0.0,
        )

    # ── Model A: compute market base ──────────────────────────────
    # Step 1: P_retail × R_brand
    market_base = original_retail_price * retention

    # Step 2: × condition multiplier (compressed band for ultra-luxury)
    cond_multipliers = CONDITION_MULTIPLIERS.get(brand_tier, CONDITION_MULTIPLIERS["high"])
    cond_mult = cond_multipliers.get(condition_score, CONDITION_MULTIPLIER_DEFAULT)

    # Step 3: × accessories premium
    accessories_mult = get_accessories_premium(category, has_accessories)

    # Full Model A price
    model_a_price = market_base * cond_mult * accessories_mult

    # ── ULTRA-HIGH: full override ──────────────────────────────────
    if brand_tier == "ultra_high":
        corrected = model_a_price

        # Sized RTW markdown (e.g. Loro Piana)
        corrected, rtw_applied = apply_sized_rtw_markdown(corrected, brand_norm, size)

        # Confidence band: tighter for ultra-luxury (compressed condition band)
        conf_low = round(corrected * 0.88, 2)
        conf_high = round(corrected * 1.12, 2)

        acc_note = f" Full-set premium applied (+{(accessories_mult-1)*100:.0f}%)." if has_accessories else ""
        rtw_note = " Sized RTW liquidity markdown applied." if rtw_applied else ""

        logic = (
            f"Ultra-high tier: Model A override. "
            f"Market base = ${original_retail_price:,.0f} × {retention:.0%} retention = ${market_base:,.0f}. "
            f"Condition multiplier ({condition_score}/10): {cond_mult:.0%}. "
            f"Final: ${corrected:,.0f}."
            f"{acc_note}{rtw_note} "
            f"Source: Rebag Clair Report 2025."
        )

        return FloorCorrectionResult(
            corrected_price=round(corrected, 2),
            confidence_low=conf_low,
            confidence_high=conf_high,
            floor_applied=True,
            floor_logic=logic,
            retention_rate=retention,
            market_base=round(market_base, 2),
        )

    # ── HIGH: soft floor blend ─────────────────────────────────────
    # If XGBoost is above Model A floor → trust XGBoost
    # If XGBoost is below Model A floor → blend toward floor
    if brand_tier == "high":
        if xgboost_price >= model_a_price:
            # XGBoost is already at or above the floor — no correction needed
            return FloorCorrectionResult(
                corrected_price=round(xgboost_price, 2),
                confidence_low=round(xgboost_price_low, 2),
                confidence_high=round(xgboost_price_high, 2),
                floor_applied=False,
                floor_logic=(
                    f"High tier: XGBoost prediction (${xgboost_price:,.0f}) is at or above "
                    f"retention floor (${model_a_price:,.0f}). No correction applied."
                ),
                retention_rate=retention,
                market_base=round(market_base, 2),
            )
        else:
            # XGBoost is below floor — blend
            blended = (
                xgboost_price * HIGH_TIER_FLOOR_BLEND
                + model_a_price * HIGH_TIER_XGBOOST_BLEND
            )
            conf_low = round(blended * 0.80, 2)
            conf_high = round(blended * 1.25, 2)

            return FloorCorrectionResult(
                corrected_price=round(blended, 2),
                confidence_low=conf_low,
                confidence_high=conf_high,
                floor_applied=True,
                floor_logic=(
                    f"High tier: XGBoost (${xgboost_price:,.0f}) below retention floor "
                    f"(${model_a_price:,.0f}). Blended: "
                    f"60% XGBoost + 40% floor = ${blended:,.0f}. "
                    f"Retention rate: {retention:.0%}."
                ),
                retention_rate=retention,
                market_base=round(market_base, 2),
            )

    # Fallback — should not be reached
    return FloorCorrectionResult(
        corrected_price=round(xgboost_price, 2),
        confidence_low=round(xgboost_price_low, 2),
        confidence_high=round(xgboost_price_high, 2),
        floor_applied=False,
        floor_logic="No floor correction applied.",
        retention_rate=0.0,
        market_base=0.0,
    )
