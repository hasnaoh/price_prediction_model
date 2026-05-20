# Resale Velocity Engine

A B2B pricing tool for luxury resale platforms. Submit a consignment item, get back a recommended listing price, a confidence range, a trend signal, and a sell-through prediction. Built as a direct outreach asset for The RealReal, Vestiaire Collective, and ThredUp.

---

## The Problem

Pricing is the last bottleneck before a consignment item goes live at The RealReal. When it's wrong, the consequences compound. An item listed 15-20% too high doesn't sell in 30 days, triggering the markdown schedule. The markdown erodes the take rate. The consignor gets less than their floor. They churn. CAC goes up.

At The RealReal's FY2025 GMV of $2.13B, that mispricing dynamic costs an estimated $23M annually in lost revenue at their 36.5% blended take rate. A one-third reduction in mispricing recovers $7.6M.

The RealReal's AI system Athena handles intake and initial pricing. The markdown decision is still largely rules-based and manual. That's the gap this tool is designed to fill.

---

## What It Returns

**Recommended listing price** from an XGBoost regression model trained on 11,044 luxury items, calibrated at inference time against brand-specific retention curves. Hermès and Chanel retain 85%+ of original retail. Gucci and LV retain 50-70%. Contemporary brands like Ganni sit at 15-30%.

**Confidence range** from quantile regression at the 10th and 90th percentiles. If the recommended price is near the top of that band, the interface surfaces a markdown risk flag.

**Trend signal** from Google Trends RSV pulled for 18 brand/category pairs, with a 2-week lag applied. The lag matters because research shows roughly a 14-day delay between a search volume spike and its effect on realized resale prices. The signal tells you whether to list now, hold, or price aggressively.

**Sell-through velocity** from a Cox Proportional Hazards model. It gives you the probability of selling within 7, 14, 30, 60, and 90 days at the recommended price, plus a price sensitivity estimate: what happens to the 30-day sell probability if you drop 10%. The concordance index on V1 training data is 0.833.

That fourth output is what makes this a B2B tool. A consumer app doesn't need a 30-day sell probability. A VP of Operations running warehouse throughput does.

---

## Architecture

```
Consignment Item Input
        │
        ▼
AI Parsing Layer
Gemini 1.5 Flash + Groq fallback + rule-based safety net
Extracts: era, creative director period, accessories score (0-4),
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
+ Quantile Bands          Sell-Through Velocity
16% MAPE improvement      C-index: 0.833
        │                        │
        └──────────┬─────────────┘
                   ▼
        Trend Adjustment Layer
        Google Trends RSV · 2-week lag
                   │
                   ▼
            FastAPI /valuate
```

---

## Model Performance

| Model | Result |
|---|---|
| Baseline (median comp) | 96.24% MAPE |
| XGBoost regression | 80.84% MAPE (16% improvement) |
| Cox PH sell-through | 0.833 concordance index |

The PRD target was a 15% MAPE improvement over baseline. That's met.

One honest note on the training data: this is built on Mercari's price suggestion dataset, which is a general consumer marketplace. Mercari prices run lower than The RealReal or Vestiaire by design. The ORP anchor and brand retention curves in the inference layer correct for this at runtime, so the output prices are calibrated to luxury platform norms rather than Mercari norms. The V2 data swap to eBay sold listings and Grailed luxury data will close this gap at the model level without requiring architectural changes.

---

## Feature Engineering Notes

**accessories_score** is a 0-4 weighted scale rather than a binary flag. It's based on what the academic literature calls the "functional alibi coefficient" -- utilitarian accessories reduce buyer indulgence guilt, and the effect is larger for ultra-luxury items where the purchase justification is harder. Dust bag, original box, authenticity card, and receipt each contribute one point. The premium per point scales by brand tier: 8% for Hermès/Chanel, 5% for LV/Gucci/Prada, 2% for contemporary brands.

**days_listed** is the burned asset feature. Items sitting on a platform for 30+ days face accelerating price decay because buyers assume hidden defects or seller ambivalence. Mercari doesn't provide listing timestamps so this column is NaN in V1. XGBoost handles NaN natively, so when real eBay/Grailed data arrives with timestamps, the column populates and the model incorporates the signal automatically. No pipeline rebuild needed.

**burned_asset_penalty** is a derived decay coefficient that activates when days_listed is populated. The V2 formula is `max(0, (days_listed - 30) * tier_decay_rate)`, where the decay rate is 0.002 for ultra-high tier, 0.005 for high tier, and 0.012 for contemporary. A Hermès bag decays slowly because the buyer pool is deep and patient. A Ganni coat decays fast because it's trend-sensitive.

**price_to_market_ratio** is the item's price divided by the median comp for its brand/category bucket. It's the primary predictor in the Cox model and encodes the J-curve: price below market and it moves fast, price above market and the markdown spiral begins.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data | Mercari Price Suggestion Dataset (Kaggle, 1.4M rows filtered to 11K luxury) |
| AI parsing | Google Gemini 1.5 Flash (free tier) with Groq Llama 3 as fallback |
| Trend signals | pytrends, Google Trends RSV, 2-week lag |
| Pricing model | XGBoost with quantile regression confidence bands |
| Sell-through model | lifelines, Cox Proportional Hazards |
| Backend | FastAPI |
| Frontend | React, Vite |
| Deploy (planned) | Railway for the API, Vercel for the frontend |

---

## Running Locally

Requires Python 3.13 and Node.js 22+.

```bash
git clone https://github.com/hasnaoh/resale-velocity-engine
cd resale-velocity-engine
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your API keys. Both are free:
- Gemini key: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- Groq key: [console.groq.com](https://console.groq.com)

Download `train.tsv` from the [Mercari Price Suggestion Challenge](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data) and place it at `data/raw/train.tsv`.

Then run the pipeline in order:

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
# http://127.0.0.1:8000
# Docs at http://127.0.0.1:8000/docs
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev
# http://localhost:5173
```

---

## Roadmap

**V1.5 (architecture commented in codebase)**

The current model trains on sold items only, which creates survivorship bias. Items that never sold because they were overpriced or badly conditioned are invisible to the model. The Heckman two-stage correction addresses this: a probit model first estimates what determines whether an item sells at all, then the price regression runs on sold items with the selection bias corrected via the Inverse Mills Ratio. The architecture is documented in the codebase. Building it requires unsold item data, which eBay scraping will provide.

Computer vision condition assessment is also flagged as a V2 swap point on the `condition_score` feature. Self-reported condition grades miss a lot. A photo catches scuffs, hardware wear, and fabric pilling that a seller won't mention.

**V2 (data infrastructure)**

The main unlock is replacing the Mercari training data with scraped eBay sold listings and Grailed luxury data. Real timestamps mean real `days_listed` values, which activates the burned asset feature and allows the Cox model to train on actual sell-through data rather than synthetic proxies. The pipeline is already wired for this swap.

On the product side: bulk CSV upload so a pricing analyst can run 100 valuations in one batch, and `like_count_24h` as a live engagement velocity signal when platform API access is available.

**V3**

Digital Product Passport integration as DPPs become standard across LVMH brands. Geographic pricing arbitrage for Vestiaire's cross-market inventory.

---

## Built For

The RealReal, Vestiaire Collective, ThredUp, and early-stage fashion tech companies including Garde-Robe and The Mall.

---

*Hana -- Mechanical Engineering, University of Michigan*
*V1 in progress.*
