"""
Resale Velocity Engine — AI Parsing Layer
Uses Google Gemini Flash (FREE — 1M tokens/day) to extract structured
attributes from listing descriptions.

Fallback: Groq (Llama 3) — also free tier.

Run AFTER: python scripts/01_data_pipeline.py

Setup:
  1. Get Gemini API key (free): https://aistudio.google.com/app/apikey
  2. Add to .env:  GEMINI_API_KEY=your-key-here
  3. Optional fallback — get Groq key (free): https://console.groq.com
     Add to .env:  GROQ_API_KEY=your-key-here

Cost: $0.00
Rate limits:
  Gemini Flash free tier: 15 requests/min, 1M tokens/day
  Groq free tier:         30 requests/min, 14.4K tokens/min
"""

import os
import json
import time
import urllib.request
import pandas as pd
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "luxury_clean.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "luxury_with_ai_features.csv"
CACHE_PATH = ROOT / "data" / "processed" / "ai_parse_cache.json"

# ─── Prompt ───────────────────────────────────────────────────────────────────
PARSE_PROMPT = """You are a luxury fashion expert and pricing analyst.
Extract the following attributes from the listing description.
Return ONLY valid JSON. No preamble, no explanation, no markdown fences.

Schema:
{{
  "era": string or null,
  "creative_director_period": string or null,
  "condition_notes": [string],
  "provenance": [string],
  "is_collaboration": true or false,
  "collab_name": string or null,
  "accessories_included": [string],
  "aesthetic_tags": [string],
  "archive_score": 0 or 1,
  "accessories_flag": 0 or 1
}}

Rules:
- archive_score = 1 if from a recognized golden creative director era
  (Phoebe Philo era Celine, Hedi Slimane era YSL, Tom Ford era Gucci,
   Nicolas Ghesquiere era Balenciaga, early Jonathan Anderson Loewe)
- accessories_flag = 1 if dust bag, box, authenticity card, or receipt mentioned
- Keep arrays short (max 3 items each)

Listing: {listing_text}"""

# ─── Rule-based fallback (no API needed) ─────────────────────────────────────

ARCHIVE_SIGNALS = [
    "phoebe philo", "old celine", "old celine", "hedi slimane",
    "slimane era", "tom ford", "nicolas ghesquiere",
    "jonathan anderson", "archive", "runway", "sample",
]
ACCESSORY_SIGNALS = [
    "dust bag", "dustbag", "box", "authenticity card", "auth card",
    "certificate", "receipt", "tags attached", "never worn", "unworn",
    "original packaging", "comes with",
]

def rule_based_scores(description):
    base = {
        "era": None, "creative_director_period": None,
        "condition_notes": [], "provenance": [],
        "is_collaboration": False, "collab_name": None,
        "accessories_included": [], "aesthetic_tags": [],
        "archive_score": 0, "accessories_flag": 0,
    }
    if not description or len(description) < 20:
        return base
    lower = description.lower()
    base["archive_score"] = int(any(s in lower for s in ARCHIVE_SIGNALS))
    base["accessories_flag"] = int(any(s in lower for s in ACCESSORY_SIGNALS))
    return base


# ─── Provider: Gemini Flash (primary — free) ──────────────────────────────────

def parse_with_gemini(description, api_key):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": PARSE_PROMPT.format(listing_text=description[:800])}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 300},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
        raw = body["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)
    except Exception:
        return None


# ─── Provider: Groq / Llama 3 (fallback — free) ──────────────────────────────

def parse_with_groq(description, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a luxury fashion pricing expert. Always respond with valid JSON only."},
            {"role": "user", "content": PARSE_PROMPT.format(listing_text=description[:800])},
        ],
        "temperature": 0,
        "max_tokens": 300,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
        raw = body["choices"][0]["message"]["content"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)
    except Exception:
        return None


# ─── Main Dispatcher ──────────────────────────────────────────────────────────

def parse_description(description, cache, gemini_key, groq_key):
    cache_key = (description or "").strip()[:400]

    if cache_key in cache:
        return cache[cache_key]

    if not description or description == "No description yet" or len(description) < 30:
        result = rule_based_scores(description)
        cache[cache_key] = result
        return result

    if gemini_key:
        result = parse_with_gemini(description, gemini_key)
        if result:
            result.setdefault("archive_score", 0)
            result.setdefault("accessories_flag", 0)
            cache[cache_key] = result
            return result

    if groq_key:
        result = parse_with_groq(description, groq_key)
        if result:
            result.setdefault("archive_score", 0)
            result.setdefault("accessories_flag", 0)
            cache[cache_key] = result
            return result

    result = rule_based_scores(description)
    cache[cache_key] = result
    return result


# ─── Pipeline Runner ──────────────────────────────────────────────────────────

def run_ai_parsing(max_rows=None, save_every=500):
    print("=" * 60)
    print("RESALE VELOCITY ENGINE — AI PARSING LAYER")
    print("Model: Gemini 1.5 Flash (free) -> Groq Llama 3 (free fallback)")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Run 01_data_pipeline.py first. Expected: {INPUT_PATH}")

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    if not gemini_key and not groq_key:
        print(
            "\n  WARNING: No API keys found. Running in RULE-BASED MODE.\n"
            "  Add to .env for richer extraction (both are free):\n"
            "    GEMINI_API_KEY=...  -> https://aistudio.google.com/app/apikey\n"
            "    GROQ_API_KEY=...    -> https://console.groq.com\n"
        )
    else:
        if gemini_key:
            print("\n  Gemini Flash key found (primary)")
        if groq_key:
            print("  Groq key found (fallback)")

    print(f"\n[1/3] Loading {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, nrows=max_rows)
    print(f"    {len(df):,} items to parse")

    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"    Cache: {len(cache):,} descriptions already parsed")

    if gemini_key:
        has_desc = df["item_description"].str.len().fillna(0) > 30
        api_calls_needed = has_desc.sum() - sum(
            1 for d in df.loc[has_desc, "item_description"] if str(d).strip()[:400] in cache
        )
        if api_calls_needed > 0:
            est_minutes = api_calls_needed / 14
            print(f"\n  ~{api_calls_needed:,} API calls needed (~{est_minutes:.0f} min)")
            print(f"  Tip: Rerunning is safe — results are cached locally.\n")

    print(f"\n[2/3] Parsing descriptions")
    ai_results = []
    api_call_count = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        desc = str(row.get("item_description", ""))
        cache_key = desc.strip()[:400]
        is_new_api_call = (
            cache_key not in cache
            and len(desc) >= 30
            and desc != "No description yet"
            and (gemini_key or groq_key)
        )

        result = parse_description(desc, cache, gemini_key, groq_key)
        ai_results.append(result)

        if is_new_api_call:
            api_call_count += 1
            if api_call_count >= 1000:
             gemini_key = None  # switch to rule-based after 1000 calls
             groq_key = None
            time.sleep(2.1)

        if (idx + 1) % 50 == 0:
            print(f"    {idx + 1:,} / {len(df):,} parsed  (API calls: {api_call_count})", end="\r")

        if (idx + 1) % save_every == 0:
            with open(CACHE_PATH, "w") as f:
                json.dump(cache, f)

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

    print(f"\n\n[3/3] Merging AI features and saving")
    ai_df = pd.DataFrame(ai_results)
    df["archive_score"] = ai_df["archive_score"].fillna(0).astype(int)
    df["accessories_flag"] = ai_df["accessories_flag"].fillna(0).astype(int)
    df["collab_flag"] = ai_df["is_collaboration"].fillna(False).astype(int)
    df["ai_parsed_json"] = [json.dumps(r) for r in ai_results]

    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("AI parsing complete")
    print(f"    API calls made:       {api_call_count}")
    print(f"    archive_score = 1:    {df['archive_score'].sum():,} items")
    print(f"    accessories_flag = 1: {df['accessories_flag'].sum():,} items")
    print(f"    collab_flag = 1:      {df['collab_flag'].sum():,} items")
    print(f"    Output: {OUTPUT_PATH}")
    print("=" * 60)
    print("\nNext step: run  python scripts/03_trends.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    run_ai_parsing(max_rows=args.max_rows)
