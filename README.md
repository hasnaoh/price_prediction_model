# Resale Velocity Engine

**Automated pricing & sell-through optimization for luxury resale marketplaces.**

A B2B pricing infrastructure prototype for platforms like The RealReal, Vestiaire Collective, and ThredUp. Accepts a luxury consignment item as input and returns a recommended listing price, confidence range, trend timing signal, and rationale breakdown — the infrastructure that turns intake into yield management.

---

## The Problem

At The RealReal's FY2025 GMV of $2.13B, an estimated 15% mispricing rate on initial listings translates to ~$23M in lost annual revenue at a 36.5% take rate. Pricing is the last bottleneck before an item goes live, and when it's wrong, markdowns cascade into what we call the **Markdown Death Spiral**.

Athena (TRR's AI system) automates intake. **The Resale Velocity Engine automates the markdown decision.** Together they close the loop.

---

## Architecture

```
Input (form + free-text description)
         │
         ▼
AI Parsing Layer (Claude Haiku)
   Extracts: era, creative director period, condition notes,
   accessories, collab flags, aesthetic tags
         │
         ▼
Feature Engineering
   brand_tier · condition_score · msrp_ratio
   saturation_index · archive_score · accessories_flag
         │
         ▼
XGBoost Regression + Quantile Confidence Bands
   Beats median baseline by ≥15% MAPE
         │
         ▼
Trend Adjustment Layer (Google Trends / pytrends)
   2-week lag RSV signal: rising / flat / declining
         │
         ▼
Output
   Recommended price · Confidence range · Trend signal · Rationale
```

---

## Stack

| Layer | Technology |
|---|---|
| Data collection | Python, Apify, Playwright |
| Data processing | Pandas, NumPy |
| Trend signals | pytrends (Google Trends) |
| Baseline model | scikit-learn (median comp) |
| Primary model | XGBoost + quantile regression |
| AI parsing | Anthropic API (Claude Haiku) |
| Backend API | FastAPI |
| Frontend | React + Tailwind CSS |
| Deploy | Vercel (frontend) + Railway (API) |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourusername/resale-velocity-engine
cd resale-velocity-engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Download the dataset

1. Go to [Mercari Price Suggestion Challenge](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data)
2. Accept the competition rules and download `train.tsv`
3. Place it at `data/raw/train.tsv`

### 4. Run the pipeline

```bash
# Step 1: Filter Mercari data, engineer features (~2 min)
python scripts/01_data_pipeline.py

# Optional: fast dev run on 50K rows
python scripts/01_data_pipeline.py --sample 50000

# Step 2: AI parsing layer — extracts structured attributes (~10 min for full dataset)
python scripts/02_ai_parsing.py

# Step 3: Pull Google Trends signals (~5 min, rate-limited)
python scripts/03_trends.py

# Step 4: Train baseline + XGBoost models (~5–10 min)
python scripts/04_train_model.py
```

### 5. Run the API

```bash
uvicorn backend.app.main:app --reload
# API runs at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 6. Run the frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

---

## Model Performance

| Model | MAPE | vs. Baseline |
|---|---|---|
| Baseline (median comp) | ~35% | — |
| XGBoost | target ≤20% | ≥15% improvement |

*Results updated after training run.*

---

## V2 Roadmap

- [ ] Sell-through velocity model (Cox Proportional Hazards — architecture in codebase)
- [ ] Live eBay/Grailed data ingestion (7-day refresh)
- [ ] Bulk CSV upload (100 items → 100 valuations)
- [ ] High-value item routing to GPT-4o for parsing
- [ ] Fast fashion expansion (Depop/Poshmark tier)

---

## Data Sources

- **Mercari Price Suggestion Dataset** (Kaggle) — 1.4M items, V1 training data
- **eBay sold listings** (Apify) — V2 ground truth replacement
- **Google Trends** (pytrends) — Trend RSV signal
- **The RealReal / Grailed** (Playwright scrape) — V2 pricing comp data

---

*Built by Hana — Portfolio Project 3*
*PRD v1.0 | Target: The RealReal, Vestiaire Collective, ThredUp*
