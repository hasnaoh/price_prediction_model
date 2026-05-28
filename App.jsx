import { useState } from "react";
import axios from "axios";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const BRANDS = [
  "hermès", "chanel", "louis vuitton", "gucci", "prada",
  "bottega veneta", "dior", "saint laurent", "valentino", "loewe",
  "givenchy", "balenciaga", "alexander mcqueen", "burberry", "fendi",
  "celine", "off-white", "jacquemus", "the row", "rick owens",
  "maison margiela", "acne studios", "ganni", "staud", "aritzia",
  "reformation", "frame", "theory",
];

const CONDITIONS = ["Pristine", "Excellent", "Very Good", "Good", "Fair"];

const CATEGORIES = [
  "Handbags", "Shoulder Bags", "Tote Bags", "Clutches", "Crossbody Bags",
  "Dresses", "Tops", "Jackets", "Coats", "Trousers", "Skirts",
  "Shoes", "Boots", "Sneakers", "Heels",
];

const TIER_LABELS = {
  ultra_high: "Ultra-High",
  high: "High",
  contemporary: "Contemporary",
};

function HourglassLoader() {
  return (
    <div className="hourglass-wrap">
      <svg className="hourglass-svg" viewBox="0 0 56 80" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M4 4 L52 4 L52 9 L30 38 L52 71 L52 76 L4 76 L4 71 L26 38 L4 9 Z"
          stroke="#1A1917" strokeWidth="1.2" fill="none" />
        <line x1="3" y1="4" x2="53" y2="4" stroke="#1A1917" strokeWidth="1.8"/>
        <line x1="3" y1="76" x2="53" y2="76" stroke="#1A1917" strokeWidth="1.8"/>
        <path className="sand-top" d="M7 7 L49 7 L28 36 Z" fill="#B87A2E" opacity="0.75"/>
        <path className="sand-bottom" d="M28 40 L20 73 L36 73 Z" fill="#B87A2E" opacity="0.45"/>
        <circle cx="28" cy="38.5" r="1.4" fill="#B87A2E"/>
      </svg>
      <p className="hourglass-label">
        Calculating optimal listing price
        <span className="dot d1">.</span>
        <span className="dot d2">.</span>
        <span className="dot d3">.</span>
      </p>
      <p className="hourglass-stack">XGBoost · Cox PH · Google Trends RSV</p>
    </div>
  );
}

function SellThroughBar({ label, value, isHero }) {
  const pct = Math.round(value * 100);
  const barColor = isHero
    ? "#B87A2E"
    : pct >= 80
    ? "#7A9E6A"
    : "#C8BFB0";
  const barHeight = isHero ? "3px" : "1.5px";
  const labelColor = isHero ? "#1A1917" : "#A09890";
  const pctColor = isHero ? "#B87A2E" : pct >= 80 ? "#7A9E6A" : "#A09890";

  return (
    <div className="st-row">
      <span className="st-label" style={{ color: labelColor, fontWeight: isHero ? "600" : "400" }}>
        {label}
      </span>
      <div className="st-bar-wrap">
        <div className="st-bar-bg" style={{ height: barHeight }}>
          <div className="st-bar-fill" style={{ width: `${pct}%`, background: barColor, height: barHeight }} />
        </div>
      </div>
      <span className="st-pct" style={{ color: pctColor, fontWeight: isHero ? "600" : "400" }}>
        {pct}%
      </span>
    </div>
  );
}

export default function App() {
  const [form, setForm] = useState({
    brand: "", category: "", condition: "",
    original_retail_price: "", color: "", material: "",
    size: "", year_season: "", is_limited_edition: false,
    free_text_description: "",
  });

  const [result, setResult] = useState(null);
  const [view, setView] = useState("form"); // 'form' | 'loading' | 'result'
  const [error, setError] = useState(null);

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const handleSubmit = async () => {
    if (!form.brand || !form.category || !form.condition || !form.original_retail_price) {
      setError("Brand, category, condition, and retail price are required.");
      return;
    }
    setError(null);
    setView("loading");
    setResult(null);
    try {
      const { data } = await axios.post(`${API}/valuate`, {
        ...form,
        original_retail_price: parseFloat(form.original_retail_price),
      });
      setResult(data);
      setView("result");
    } catch (e) {
      setError(e.response?.data?.detail || "API not responding. Is the backend running?");
      setView("form");
    }
  };

  const handleReset = () => {
    setResult(null);
    setView("form");
    setError(null);
  };

  const trendColor = result
    ? result.trend_direction === "rising" ? "#7A9E6A"
    : result.trend_direction === "declining" ? "#C0554A"
    : "#B87A2E"
    : "#B87A2E";

  const trendIcon = result
    ? result.trend_direction === "rising" ? "↑"
    : result.trend_direction === "declining" ? "↓"
    : "="
    : null;

  const itemLabel = result
    ? [form.brand, form.category, form.year_season, form.condition]
        .filter(Boolean)
        .map(s => s.charAt(0).toUpperCase() + s.slice(1))
        .join(" · ")
    : "";

  const confidencePct = result
    ? Math.min(95, Math.max(5,
        ((result.recommended_price - result.confidence_low) /
        (result.confidence_high - result.confidence_low)) * 100
      ))
    : 50;

  return (
    <div className="shell">
      <header className="header">
        <div className="header-inner">
          <div className="wordmark">
            <span className="wordmark-main">KAIROS</span>
            <span className="wordmark-divider">|</span>
            <span className="wordmark-sub">PRICING INTELLIGENCE FOR LUXURY RESALE</span>
          </div>
          {view === "result" && (
            <button className="new-valuation-btn" onClick={handleReset}>← New Valuation</button>
          )}
          {view === "form" && (
            <span className="header-meta">v2.0</span>
          )}
        </div>
      </header>

      <main className="workspace">

        {/* ── FORM STATE ── */}
        {view === "form" && (
          <div className="form-surface">
            <p className="section-eyebrow">Consignment Item</p>

            <div className="field-group">
              <div className="field field-required">
                <label>Brand</label>
                <select value={form.brand} onChange={(e) => set("brand", e.target.value)}>
                  <option value="">Select brand</option>
                  {BRANDS.map((b) => (
                    <option key={b} value={b}>{b.charAt(0).toUpperCase() + b.slice(1)}</option>
                  ))}
                </select>
              </div>
              <div className="field field-required">
                <label>Category</label>
                <select value={form.category} onChange={(e) => set("category", e.target.value)}>
                  <option value="">Select category</option>
                  {CATEGORIES.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="field">
              <label>Condition</label>
              <div className="condition-pills">
                {CONDITIONS.map((c) => (
                  <button
                    key={c}
                    className={`pill ${form.condition === c ? "pill-active" : ""}`}
                    onClick={() => set("condition", c)}
                  >{c}</button>
                ))}
              </div>
            </div>

            <div className="field-group">
              <div className="field field-required">
                <label>Original Retail Price (USD)</label>
                <div className="input-prefix-wrap">
                  <span className="input-prefix">$</span>
                  <input type="number" placeholder="0"
                    value={form.original_retail_price}
                    onChange={(e) => set("original_retail_price", e.target.value)}
                  />
                </div>
              </div>
              <div className="field">
                <label>Year / Season</label>
                <input type="text" placeholder="e.g. FW22, SS19"
                  value={form.year_season}
                  onChange={(e) => set("year_season", e.target.value)}
                />
              </div>
            </div>

            <div className="field-group">
              <div className="field">
                <label>Color</label>
                <input type="text" placeholder="e.g. Noir, Camel"
                  value={form.color} onChange={(e) => set("color", e.target.value)} />
              </div>
              <div className="field">
                <label>Material</label>
                <input type="text" placeholder="e.g. Leather, Cashmere"
                  value={form.material} onChange={(e) => set("material", e.target.value)} />
              </div>
            </div>

            <div className="field">
              <label>Listing Description</label>
              <textarea rows={4}
                placeholder="Mention accessories, provenance, condition details, archive era..."
                value={form.free_text_description}
                onChange={(e) => set("free_text_description", e.target.value)}
              />
            </div>

            <div className="field field-toggle">
              <label>
                <input type="checkbox" checked={form.is_limited_edition}
                  onChange={(e) => set("is_limited_edition", e.target.checked)} />
                Limited edition / collaboration
              </label>
            </div>

            {error && <div className="error-msg">{error}</div>}

            <button className="submit-btn" onClick={handleSubmit}>
              Run Valuation
            </button>
          </div>
        )}

        {/* ── LOADING STATE ── */}
        {view === "loading" && <HourglassLoader />}

        {/* ── RESULT STATE ── */}
        {view === "result" && result && (
          <div className="result-surface">
            <div className="result-header">
              <p className="result-eyebrow">{itemLabel}</p>
              <h1 className="result-title">Valuation Output</h1>
            </div>

            <div className="result-divider" />

            {/* Price */}
            <div className="price-block">
              <p className="block-label">Recommended Listing Price</p>
              <div className="price-value">
                ${result.recommended_price.toLocaleString("en-US", { maximumFractionDigits: 0 })}
              </div>
              <div className="price-band-row">
                <span className="price-band-label">90% confidence band</span>
                <span className="price-band-values">
                  ${Math.round(result.confidence_low).toLocaleString()} — ${Math.round(result.confidence_high).toLocaleString()}
                </span>
              </div>
              <div className="confidence-track">
                <div className="confidence-fill" style={{ width: `${confidencePct}%` }} />
                <div className="confidence-dot" style={{ left: `${confidencePct}%` }} />
              </div>
              <div className="confidence-ends">
                <span>${Math.round(result.confidence_low).toLocaleString()}</span>
                <span>${Math.round(result.confidence_high).toLocaleString()}</span>
              </div>
              {result.rationale.markdown_risk && (
                <div className="markdown-warning">
                  High markdown risk — listing price near upper confidence bound
                </div>
              )}
            </div>

            <div className="result-divider" />

            {/* Trend */}
            <div className="trend-block">
              <div className="trend-accent" style={{ background: trendColor }} />
              <div className="trend-body">
                <div className="trend-header-row">
                  <span className="trend-direction" style={{ color: trendColor }}>
                    {trendIcon} {result.trend_direction.toUpperCase()} TREND
                  </span>
                  <span className="trend-rsv">RSV {result.rationale.trend_rsv}</span>
                </div>
                <p className="trend-text">{result.trend_signal}</p>
              </div>
            </div>

            <div className="result-divider" />

            {/* Sell-Through */}
            {result.sell_through && result.sell_through.probability_30_day > 0 && (
              <div className="st-block">
                <p className="block-label">Sell-Through Velocity</p>
                <div className="st-headline">
                  <span className="st-big">
                    {Math.round(result.sell_through.probability_30_day * 100)}%
                  </span>
                  <span className="st-caption">probability of selling<br/>within 30 days at this price</span>
                </div>
                <div className="st-bars">
                  <SellThroughBar label="7 days"  value={result.sell_through.probability_7_day}  isHero={false} />
                  <SellThroughBar label="14 days" value={result.sell_through.probability_14_day} isHero={false} />
                  <SellThroughBar label="30 days" value={result.sell_through.probability_30_day} isHero={true} />
                  <SellThroughBar label="60 days" value={result.sell_through.probability_60_day} isHero={false} />
                  <SellThroughBar label="90 days" value={result.sell_through.probability_90_day} isHero={false} />
                </div>
                <p className="st-sensitivity">{result.sell_through.price_sensitivity}</p>
              </div>
            )}

            <div className="result-divider" />

            {/* Rationale */}
            <div className="rationale-block">
              <p className="block-label">Rationale</p>
              <div className="rationale-grid">
                {[
                  ["Brand tier", TIER_LABELS[result.rationale.brand_tier] || result.rationale.brand_tier],
                  ["Condition score", `${result.rationale.condition_score} / 10`],
                  ["Accessories included", result.rationale.accessories_included ? "Yes — premium applied" : "No"],
                  ["Archive item", result.rationale.archive_item ? "Yes — era premium applied" : "No"],
                  ["Collaboration / LE", result.rationale.collaboration_item ? "Yes" : "No"],
                  ["Confidence band", result.rationale.confidence_band],
                ].map(([k, v]) => (
                  <div className="rationale-row" key={k}>
                    <span className="r-key">{k}</span>
                    <span className={`r-val ${
                      (k === "Accessories included" || k === "Archive item") && v.startsWith("Yes")
                        ? "r-flag-yes" : ""
                    }`}>{v}</span>
                  </div>
                ))}
              </div>
            </div>

            <p className="output-footer">
              Model v{result.model_version} · XGBoost + Cox PH + Google Trends RSV
            </p>
          </div>
        )}

      </main>
    </div>
  );
}
