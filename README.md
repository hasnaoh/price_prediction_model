# Resale Velocity Engine

**B2B pricing and sell-through optimization for luxury resale marketplaces.**

A direct outreach asset for The RealReal, Vestiaire Collective, and ThredUp. Submit a consignment item, get back a recommended listing price, a confidence range, a trend signal, and a sell-through prediction.

**Live demo:** [price-prediction-model-alpha.vercel.app](https://price-prediction-model-alpha.vercel.app/)

---

## The Problem

Pricing is the last bottleneck before a consignment item goes live at The RealReal. When it's wrong, the consequences compound. An item listed 15–20% too high doesn't sell in 30 days, triggering the markdown schedule. The markdown erodes the take rate. The consignor gets less than their floor. They churn. CAC goes up.

At The RealReal's FY2025 GMV of $2.13B, that mispricing dynamic costs an estimated $23M annually in lost revenue at their 36.5% blended take rate. A one-third reduction in mispricing recovers $7.6M.

TRR's AI system Athena handles intake and initial pricing. The markdown decision is still largely rules-based and manual. That's the gap this tool is designed to fill.

---

## What It Returns

**Recommended listing price** from an XGBoost regression model trained on 11,044 luxury items, calibrated at inference time against brand-specific retention curves. Hermès and Chanel retain 85%+ of original retail. Gucci and LV retain 50–70%. Contemporary brands like Ganni sit at 15–30%.

**Confidence range** from quantile regression at the 10th and 90th percentiles. If the recommended price is near the top of that band, the interface surfaces a markdown risk flag.

**Trend signal** from Google Trends RSV pulled for 18 brand/category pairs, with a 2-week lag applied. The lag matters because research shows roughly a 14-day delay between a search volume spike and its effect on realized resale prices.

**Sell-through velocity** from a Cox Proportional Hazards model. Probability of selling within 7, 14, 30, 60, and 90 days at the recommended price, plus a price sensitivity estimate: what happens to the 30-day sell probability if you drop 10%. Concordance index on V1 training data: 0.833.

That fourth output is what makes this a B2B tool. A consumer app doesn't need a 30-day sell probability. A VP of Operations running warehouse throughput does.

---

## Architecture

Consignment Item Input
│
▼
AI Parsing Layer
Gemini 1.5 Flash + Groq fallback + rule-based safety net
Extracts: era, creative director period, accessories score (0–4),
condition notes, collab flags, aesthetic tags
│
▼
Feature Engineering
brand_tier · condition_score · accessories_score
saturation_index · archive_score · price_to_market_ratio
days_listed (NaN in V1) · burned_asset_penalty (0.0 in V1)
│
├────────────────────────┐
▼                        ▼
XGBoost Regression        Cox Proportional Hazards

Quantile Bands          Sell-Through Velocity
16% MAPE improvement      C-index: 0.833
│                        │
└──────────┬─────────────┘
▼
Trend Adjustment Layer
Google Trends RSV · 2-week lag
│
▼
FastAPI /valuate


---

## Model Performance

| Model | Result |
|---|---|
| Baseline (median comp) | 96.24% MAPE |
| XGBoost regression | 80.84% MAPE (16% improvement) |
| Cox PH sell-through | 0.833 concordance index |

PRD target was ≥15% MAPE improvement over baseline. Met.

**One honest note on training data:** this is built on Mercari's price suggestion dataset, a general consumer marketplace. Mercari prices run lower than The RealReal or Vestiaire by design. The ORP anchor and brand retention curves in the inference layer correct for this at runtime. The V2 data swap to eBay sold listings and Grailed luxury data will close this gap at the model level without architectural changes.

---

## Feature Engineering Notes

**`accessories_score`** is a 0–4 weighted scale rather than a binary flag, based on what the literature calls the "functional alibi coefficient" — utilitarian accessories reduce buyer indulgence guilt, and the effect scales with brand tier. Dust bag, original box, authenticity card, and receipt each contribute one point. The premium per point: 8% for Hermès/Chanel, 5% for LV/Gucci/Prada, 2% for contemporary.

**`days_listed`** is the burned asset feature. Items sitting on a platform for 30+ days face accelerating price decay because buyers assume hidden defects. Mercari doesn't provide timestamps so this column is NaN in V1. XGBoost handles NaN natively — when real eBay/Grailed data arrives with timestamps, the column populates and the model incorporates the signal automatically. No pipeline rebuild needed.

**`burned_asset_penalty`** activates when `days_listed` is populated. Formula: `max(0, (days_listed - 30) × tier_decay_rate)`, where decay rates are 0.002 (ultra-high), 0.005 (high), 0.012 (contemporary). A Hermès bag decays slowly because the buyer pool is deep and patient. A Ganni coat decays fast because it's trend-sensitive.

**`price_to_market_ratio`** is the item's price divided by the median comp for its brand/category bucket. Primary predictor in the Cox model. Encodes the J-curve: price below market and it moves fast, price above market and the markdown spiral begins.

---

## Stack

| Layer | Technology |
|---|---|
| Data | Mercari Price Suggestion Dataset (Kaggle, 1.4M rows → 11K luxury) |
| AI parsing | Google Gemini 1.5 Flash (free tier) + Groq Llama 3 fallback |
| Trend signals | pytrends, Google Trends RSV, 2-week lag |
| Pricing model | XGBoost + quantile regression confidence bands |
| Sell-through model | lifelines, Cox Proportional Hazards |
| Backend | FastAPI |
| Frontend | React, Vite |
| Deploy | Railway (API) + Vercel (frontend) |

---

## Quickstart

Requires Python 3.13 and Node.js 22+.

```bash
git clone https://github.com/hasnaoh/resale-velocity-engine
cd resale-velocity-engine
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your API keys (both free):
- Gemini: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- Groq: [console.groq.com](https://console.groq.com)

Download `train.tsv` from the [Mercari Price Suggestion Challenge](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data) and place it at `data/raw/train.tsv`.

Run the pipeline in order:

```bash
python scripts/data_pipeline.py
python scripts/ai_parsing.py
python scripts/trends.py
python scripts/train_model.py
pip install lifelines
python scripts/cox_model.py
```

Start the API:

```bash
python -m uvicorn backend.app.main:app --reload
# http://127.0.0.1:8000/docs
```

Start the frontend:

```bash
cd frontend && npm install && npm run dev
# http://localhost:5173
```

---

## Roadmap

**V1.5 — Architecture documented in codebase**
- Heckman two-stage selection correction for survivorship bias (probit selection model + Inverse Mills Ratio on price regression). Requires unsold item data from eBay scrape.
- Computer vision condition assessment as a swap-in for self-reported condition grades.

**V2 — Data infrastructure**
- Replace Mercari training data with scraped eBay sold listings + Grailed luxury data. Real timestamps activate `days_listed` and allow Cox model to train on actual sell-through velocity.
- Bulk CSV upload for batch valuation (100 items → 100 valuations).
- `like_count_24h` as live engagement velocity signal when platform API access is available.

**V3**
- Digital Product Passport integration as DPPs become standard across LVMH brands.
- Geographic pricing arbitrage for Vestiaire's cross-market inventory.

---

*Built by Hana — Portfolio Project*  
*Target: The RealReal, Vestiaire Collective, ThredUp, Garde-Robe, The Mall*
