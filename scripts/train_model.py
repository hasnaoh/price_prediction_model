"""
Resale Velocity Engine — Model Training
Trains the baseline (median comp) and XGBoost models.
Evaluates both on MAPE — goal is 15–20% improvement over baseline.
Outputs model artifacts to backend/models/.

Run AFTER: python scripts/03_trends.py
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "luxury_final.csv"
MODEL_DIR = ROOT / "backend" / "models"
BASELINE_PATH = MODEL_DIR / "baseline_lookup.json"
XGBOOST_PATH = MODEL_DIR / "xgboost_model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

# ─── Feature Configuration ────────────────────────────────────────────────────

CATEGORICAL_FEATURES = ["brand_tier", "top_level", "mid_level", "item_type"]
NUMERIC_FEATURES = [
    "condition_score",
    "saturation_index",
    "description_length",
    "shipping_included",
    "archive_score",
    "accessories_flag",
    "collab_flag",
    "google_trend_rsv",
    "trend_delta_30d",
    "trend_signal_encoded",
    "has_description",
]
TARGET = "price"

# Quantile regression: predict confidence band
QUANTILE_LOW = 0.10
QUANTILE_HIGH = 0.90

# ─── Baseline Model ───────────────────────────────────────────────────────────

def build_baseline_lookup(df: pd.DataFrame) -> dict:
    """
    Median comp model: Price = Median(last_6_months_sold | brand + category + condition_bucket)
    Returns a lookup dict. Fallback hierarchy: brand+cat+cond → brand+cat → brand_tier
    """
    lookup = {}

    # Bucket condition into 3 bins for lookup (otherwise too sparse)
    df["cond_bucket"] = pd.cut(
        df["condition_score"],
        bins=[0, 4, 7, 10],
        labels=["fair", "good", "excellent"],
    )

    for (brand, mid, cond), group in df.groupby(
        ["brand_normalized", "mid_level", "cond_bucket"], observed=True
    ):
        key = f"{brand}|{mid}|{cond}"
        lookup[key] = float(group[TARGET].median())

    # Brand + category fallback
    for (brand, mid), group in df.groupby(["brand_normalized", "mid_level"]):
        key = f"{brand}|{mid}"
        lookup[key] = float(group[TARGET].median())

    # Brand tier fallback
    for tier, group in df.groupby("brand_tier"):
        lookup[f"tier|{tier}"] = float(group[TARGET].median())

    return lookup


def predict_baseline(row: dict, lookup: dict) -> float:
    """Predict using baseline lookup with fallback hierarchy."""
    brand = str(row.get("brand_normalized", ""))
    mid = str(row.get("mid_level", ""))
    cond_score = float(row.get("condition_score", 6))
    tier = str(row.get("brand_tier", "high"))

    # Bucket condition
    if cond_score <= 4:
        cond = "fair"
    elif cond_score <= 7:
        cond = "good"
    else:
        cond = "excellent"

    # Try most specific key first, fall back progressively
    for key in [f"{brand}|{mid}|{cond}", f"{brand}|{mid}", f"tier|{tier}"]:
        if key in lookup:
            return lookup[key]

    return 150.0  # last-resort fallback


# ─── XGBoost Model ────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train):
    """Train median regression model."""
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="mape",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )
    return model


def train_quantile_model(X_train, y_train, quantile: float):
    """Train a quantile regression model for confidence bands."""
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def evaluate_mape(y_true, y_pred, label: str) -> float:
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"    {label:<35} MAPE = {mape:.2f}%")
    return mape


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_training():
    print("=" * 60)
    print("RESALE VELOCITY ENGINE — MODEL TRAINING")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"❌  Run 03_trends.py first.\n    Expected: {INPUT_PATH}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────
    print(f"\n[1/5] Loading {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=[TARGET])
    print(f"    {len(df):,} rows for training")

    # ── Train/test split ──────────────────────────────────────────
    print("\n[2/5] Splitting train / test (80/20)")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"    Train: {len(train_df):,} | Test: {len(test_df):,}")

    # ── Baseline model ────────────────────────────────────────────
    print("\n[3/5] Training baseline (median comp) model")
    baseline_lookup = build_baseline_lookup(train_df)
    baseline_preds = test_df.apply(
        lambda row: predict_baseline(row.to_dict(), baseline_lookup), axis=1
    )
    baseline_mape = evaluate_mape(test_df[TARGET], baseline_preds, "Baseline (median comp)")
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline_lookup, f)
    print(f"    Saved: {BASELINE_PATH}")

    # ── XGBoost model ─────────────────────────────────────────────
    print("\n[4/5] Training XGBoost model")

    # Encode categoricals
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    
    all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    def prepare_X(df_subset):
        X = df_subset[all_features].copy()
        # Fill missing numeric with median
        for col in NUMERIC_FEATURES:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(
                    df_subset[col].median() if col in df_subset else 0
                )
        return X

    X_train_raw = prepare_X(train_df)
    X_test_raw = prepare_X(test_df)
    y_train = np.log1p(train_df[TARGET].values)
    y_test = np.log1p(test_df[TARGET].values)

    # Fit encoder on train, transform both
    X_train_raw[CATEGORICAL_FEATURES] = encoder.fit_transform(
        X_train_raw[CATEGORICAL_FEATURES].astype(str)
    )
    X_test_raw[CATEGORICAL_FEATURES] = encoder.transform(
        X_test_raw[CATEGORICAL_FEATURES].astype(str)
    )

    X_train = X_train_raw.values
    X_test = X_test_raw.values

    # Main regression model
    print("    Training main regression model...")
    main_model = train_xgboost(X_train, y_train)
    xgb_preds = np.expm1(main_model.predict(X_test))
    y_test_actual = np.expm1(y_test)
    xgb_mape = evaluate_mape(y_test_actual, xgb_preds, "XGBoost (main)")

    # Quantile models for confidence band
    print("    Training confidence band models (10th + 90th percentile)...")
    low_model = train_quantile_model(X_train, y_train, QUANTILE_LOW)
    high_model = train_quantile_model(X_train, y_train, QUANTILE_HIGH)
    evaluate_mape(y_test, low_model.predict(X_test), f"XGBoost ({int(QUANTILE_LOW*100)}th percentile)")
    evaluate_mape(y_test, high_model.predict(X_test), f"XGBoost ({int(QUANTILE_HIGH*100)}th percentile)")

    # Compute improvement over baseline
    improvement = (baseline_mape - xgb_mape) / baseline_mape * 100
    target_met = improvement >= 15
    print(f"\n    MAPE improvement over baseline: {improvement:.1f}%")
    print(f"    Target (≥15%): {'✅ MET' if target_met else '❌ NOT MET — needs tuning'}")

    # Save models
    with open(XGBOOST_PATH, "wb") as f:
        pickle.dump(
            {
                "main": main_model,
                "low": low_model,
                "high": high_model,
                "feature_names": all_features,
            },
            f,
        )
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    # Save metrics
    metrics = {
        "baseline_mape": round(baseline_mape, 2),
        "xgboost_mape": round(xgb_mape, 2),
        "improvement_pct": round(improvement, 2),
        "target_met": target_met,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "features": all_features,
        "quantile_low": QUANTILE_LOW,
        "quantile_high": QUANTILE_HIGH,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n[5/5] Feature importance (top 10)")
    importances = dict(
        zip(all_features, main_model.feature_importances_)
    )
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, imp in top:
        bar = "█" * int(imp * 200)
        print(f"    {feat:<30} {bar} {imp:.4f}")

    print("\n" + "=" * 60)
    print("✅  Training complete")
    print(f"    Models saved to: {MODEL_DIR}")
    print(f"    Metrics: {METRICS_PATH}")
    print("=" * 60)
    print("\nNext step: run  uvicorn backend.app.main:app --reload")


if __name__ == "__main__":
    run_training()
