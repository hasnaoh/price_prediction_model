"""
Resale Velocity Engine — Cox Proportional Hazards Model
Predicts sell-through velocity: probability of selling within 7/14/30/60/90 days.

This is the feature that separates the Velocity Engine from TRUSS and Athena.
TRUSS returns a price. This returns a price AND a sell-through curve.

Run AFTER: python scripts/04_train_model.py (needs luxury_final.csv)

Output:
  - backend/models/cox_model.pkl
  - backend/models/cox_baseline_survival.pkl

V2 SWAP: Replace synthetic days_to_sell with real eBay listing/sold timestamps.
         The model architecture does not need to change — just the input data.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "luxury_final.csv"
MODEL_DIR = ROOT / "backend" / "models"
COX_MODEL_PATH = MODEL_DIR / "cox_model.pkl"
COX_BASELINE_PATH = MODEL_DIR / "cox_baseline_survival.pkl"
COX_METRICS_PATH = MODEL_DIR / "cox_metrics.json"

# ─── Features for Cox model ───────────────────────────────────────────────────
# These are the covariates that affect sell-through velocity.
# price_to_market_ratio is the most important — it encodes the J-curve.
COX_FEATURES = [
    "price_to_market_ratio",   # PRIMARY: how aggressively the item is priced
    "condition_score",          # worse condition -> slower sell
    "saturation_index",         # more supply -> slower sell
    "accessories_flag",         # accessories -> faster sell (5-12% premium signal)
    "archive_score",            # archive items have niche but loyal demand
    "trend_signal_encoded",     # rising trend -> faster sell
]

DURATION_COL = "days_to_sell"
EVENT_COL = "event_observed"


def prepare_cox_data(df):
    """Prepare features for Cox model. Fill missing with neutral values."""
    cox_df = df[COX_FEATURES + [DURATION_COL, EVENT_COL]].copy()

    # Fill missing with neutral/median values
    defaults = {
        "price_to_market_ratio": 1.0,
        "condition_score": 6,
        "saturation_index": 1.0,
        "accessories_flag": 0,
        "archive_score": 0,
        "trend_signal_encoded": 0,
    }
    for col, val in defaults.items():
        if col in cox_df.columns:
            cox_df[col] = pd.to_numeric(cox_df[col], errors="coerce").fillna(val)

    # Clip duration to reasonable range
    cox_df[DURATION_COL] = cox_df[DURATION_COL].clip(1, 180)
    cox_df[EVENT_COL] = cox_df[EVENT_COL].fillna(1).astype(int)

    return cox_df


def predict_survival_at_checkpoints(cox_model, feature_row, checkpoints=[7, 14, 30, 60, 90]):
    """
    Given a single feature row, return sell probability at each checkpoint.
    Returns dict: {days: probability_of_selling_by_that_day}
    """
    survival = cox_model.predict_survival_function(feature_row)
    result = {}
    for day in checkpoints:
        # Survival function = P(not sold by day). Sell prob = 1 - survival.
        closest_idx = survival.index[
            np.argmin(np.abs(survival.index - day))
        ]
        survival_prob = float(survival.loc[closest_idx].iloc[0])
        result[day] = round(1 - survival_prob, 3)
    return result


def compute_price_sensitivity(cox_model, base_row, price_ratios=[0.9, 0.95, 1.05, 1.1]):
    """
    Show how sell probability at 30 days changes with price adjustments.
    Returns list of {price_change_pct, probability_30d} dicts.
    """
    sensitivities = []
    base_prob = predict_survival_at_checkpoints(cox_model, base_row, [30])[30]

    for ratio in price_ratios:
        adjusted = base_row.copy()
        adjusted["price_to_market_ratio"] = base_row["price_to_market_ratio"].iloc[0] * ratio
        prob = predict_survival_at_checkpoints(cox_model, adjusted, [30])[30]
        change_pct = int((ratio - 1) * 100)
        sensitivities.append({
            "price_change_pct": change_pct,
            "probability_30d": prob,
            "delta_vs_base": round(prob - base_prob, 3),
        })
    return sensitivities


def run_cox_training():
    print("=" * 60)
    print("RESALE VELOCITY ENGINE — COX MODEL TRAINING")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Run 03_trends.py first. Expected: {INPUT_PATH}"
        )

    # Install lifelines if needed
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        print("Installing lifelines...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lifelines", "-q"])
        from lifelines import CoxPHFitter

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Loading {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"    {len(df):,} rows")

    # Check required columns exist
    missing = [c for c in COX_FEATURES + [DURATION_COL, EVENT_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            "Re-run data pipeline (01_data_pipeline.py) to generate V2 features."
        )

    print("\n[2/4] Preparing Cox training data")
    cox_df = prepare_cox_data(df)
    print(f"    Duration range: {cox_df[DURATION_COL].min():.0f} - {cox_df[DURATION_COL].max():.0f} days")
    print(f"    Event rate: {cox_df[EVENT_COL].mean()*100:.1f}% sold")
    print(f"    Median days_to_sell: {cox_df[DURATION_COL].median():.1f} days")

    print("\n[3/4] Fitting Cox Proportional Hazards model")
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.1)  # penalizer prevents overfitting on small dataset
    cph.fit(
        cox_df,
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
        show_progress=False,
    )

    # Concordance index: 0.5 = random, 1.0 = perfect ranking
    c_index = concordance_index(
        cox_df[DURATION_COL],
        -cph.predict_partial_hazard(cox_df[COX_FEATURES]),
        cox_df[EVENT_COL],
    )

    print(f"\n    Concordance index: {c_index:.3f}  (0.5=random, 1.0=perfect)")
    print(f"    (>0.6 is good for survival models on noisy data)")
    print("\n    Coefficient summary (positive = faster sell):")
    summary = cph.summary[["coef", "exp(coef)", "p"]].round(3)
    # Flip sign for display: negative hazard = slower sell. We want intuitive direction.
    for feat, row in summary.iterrows():
        direction = "faster" if row["coef"] < 0 else "slower"
        bar = "+" * min(20, abs(int(row["coef"] * 10)))
        print(f"    {feat:<30} {row['coef']:+.3f}  ({direction}) p={row['p']:.3f}")

    # Save model and baseline survival
    with open(COX_MODEL_PATH, "wb") as f:
        pickle.dump(cph, f)

    baseline_survival = cph.baseline_survival_
    with open(COX_BASELINE_PATH, "wb") as f:
        pickle.dump(baseline_survival, f)

    # Demo prediction on a sample item
    print("\n[4/4] Demo prediction")
    sample = cox_df[COX_FEATURES].iloc[[0]].copy()
    probs = predict_survival_at_checkpoints(cph, sample)
    print("    Sample item sell probabilities:")
    for day, prob in probs.items():
        bar = "█" * int(prob * 20)
        print(f"      Day {day:>3}: {bar:<20} {prob*100:.0f}%")

    metrics = {
        "concordance_index": round(c_index, 4),
        "n_train": len(cox_df),
        "event_rate": round(cox_df[EVENT_COL].mean(), 3),
        "median_days_to_sell": round(cox_df[DURATION_COL].median(), 1),
        "features": COX_FEATURES,
        "duration_col": DURATION_COL,
        "event_col": EVENT_COL,
        "note": "V1: synthetic days_to_sell. V2 SWAP: real eBay timestamps.",
    }
    with open(COX_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("Cox model training complete")
    print(f"    Concordance index: {c_index:.3f}")
    print(f"    Model saved to: {COX_MODEL_PATH}")
    print("=" * 60)
    print("\nNext step: python -m uvicorn backend.app.main:app --reload")


if __name__ == "__main__":
    run_cox_training()