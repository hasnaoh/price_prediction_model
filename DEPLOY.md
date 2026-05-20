DEPLOYMENT STEPS — Resale Velocity Engine
==========================================

BEFORE YOU START: confirm backend/models/ has these 4 files:
  - xgboost_model.pkl
  - encoder.pkl
  - baseline_lookup.json
  - training_metrics.json
If any are missing, run: python scripts/04_train_model.py

─── STEP 1: Fix the repo ─────────────────────────────────────────────────────

Replace your .gitignore with the updated one (model files are now un-ignored).
Then run:

  git rm -r --cached backend/models/
  git add backend/models/
  git add .gitignore railway.toml nixpacks.toml requirements-prod.txt
  git commit -m "deploy: add model artifacts + Railway config"
  git push


─── STEP 2: Deploy backend to Railway ────────────────────────────────────────

1. Go to https://railway.app → New Project → Deploy from GitHub repo
2. Select your resale-velocity-engine repo
3. Railway auto-detects nixpacks.toml and uses requirements-prod.txt
4. Under Settings → Variables, no env vars needed for V1
5. Wait for build (~2 min). Check Deploy Logs.
6. Under Settings → Networking → Generate Domain
   → Copy your Railway URL (e.g. https://resale-velocity-engine.up.railway.app)


─── STEP 3: Update frontend API URL ──────────────────────────────────────────

In your frontend, find wherever the API base URL is set.
It's either in:
  - frontend/src/config.js  (or similar)
  - an import.meta.env.VITE_API_URL reference
  - hardcoded as http://localhost:8000

Change it to your Railway URL. Then:

  cd frontend
  npm run build        ← confirm it builds cleanly before deploying


─── STEP 4: Deploy frontend to Vercel ────────────────────────────────────────

Option A — Vercel CLI (fastest):
  npm install -g vercel
  cd frontend
  vercel --prod

Option B — Vercel dashboard:
  1. Go to https://vercel.com → New Project → Import Git repo
  2. Set Root Directory to: frontend
  3. Framework: Vite (auto-detected)
  4. Add env var: VITE_API_URL = https://your-railway-url.up.railway.app
  5. Deploy


─── STEP 5: Smoke test ───────────────────────────────────────────────────────

Hit your Railway URL directly first:
  curl https://your-railway-url.up.railway.app/health

Then test the endpoint:
  curl -X POST https://your-railway-url.up.railway.app/valuate \
    -H "Content-Type: application/json" \
    -d '{"brand":"gucci","category":"Handbags","condition":"Excellent","original_retail_price":1800}'

Then open the Vercel URL and confirm the form submits and returns output.
