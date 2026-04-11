import { useState } from "react";
import axios from "axios";
import "./App.css";

const API = "http://127.0.0.1:8000";

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

function SellThroughBar({ label, value }) {
  const pct = Math.round(value * 100);
  const color = pct >= 70 ? "#7fb685" : pct >= 45 ? "#c9a060" : "#c0554a";
  return (
    <div className="st-row">
      <span className="st-label">{label}</span>
      <div className="st-bar-wrap">
        <div className="st-bar-bg">
          <div className="st-bar-fill" style={{ width: `${pct}%`, background: color }} />
        </div>
      </div>
      <span className="st-pct" style={{ color }}>{pct}%</span>
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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const handleSubmit = async () => {
    if (!form.brand || !form.category || !form.condition || !form.original_retail_price) {
      setError("Brand, category, condition, and retail price are required.");
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const { data } = await axios.post(`${API}/valuate`, {
        ...form,
        original_retail_price: parseFloat(form.original_retail_price),
      });
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || "API not responding. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const trendColor = result
    ? result.trend_direction === "rising" ? "#7fb685"
    : result.trend_direction === "declining" ? "#c0554a"
    : "#c9a060"
    : "#c9a060";

  const trendIcon = result
    ? result.trend_direction === "rising" ? "+" 
    : result.trend_direction === "declining" ? "-" 
    : "="
    : null;

  return (
    <div className="shell">
      <header className="header">
        <div className="header-inner">
          <div className="wordmark">
            <span className="wordmark-main">RESALE VELOCITY ENGINE</span>
            <span className="wordmark-sub">Pricing & Sell-Through Intelligence · B2B</span>
          </div>
          <div className="header-meta">v2.0 · Luxury RTW & Bags</div>
        </div>
      </header>

      <main className="workspace">
        {/* Input Panel */}
        <section className="panel panel-input">
          <div className="panel-label">CONSIGNMENT ITEM</div>

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
                <input
                  type="number" placeholder="0"
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
            <label>Listing Description <span className="label-hint">— feeds AI parsing layer</span></label>
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

          <button
            className={`submit-btn ${loading ? "submit-loading" : ""}`}
            onClick={handleSubmit} disabled={loading}
          >
            {loading
              ? <span className="loader-wrap"><span className="loader" /> Running valuation...</span>
              : "Run Valuation"}
          </button>
        </section>

        {/* Output Panel */}
        <section className={`panel panel-output ${result ? "panel-output-active" : ""}`}>
          <div className="panel-label">VALUATION OUTPUT</div>

          {!result && !loading && (
            <div className="empty-state">
              <div className="empty-icon">◈</div>
              <p>Submit a consignment item to generate<br />a recommended listing price.</p>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <div className="empty-icon spinning">◈</div>
              <p>Calculating...</p>
            </div>
          )}

          {result && (
            <div className="output-content">

              {/* 1. Price */}
              <div className="price-block">
                <div className="panel-label">RECOMMENDED LISTING PRICE</div>
                <div className="price-value">
                  ${result.recommended_price.toLocaleString("en-US", { maximumFractionDigits: 0 })}
                </div>
                <div className="price-band">
                  90% confidence band &nbsp;·&nbsp;
                  <strong>${Math.round(result.confidence_low).toLocaleString()} - ${Math.round(result.confidence_high).toLocaleString()}</strong>
                </div>
                <div className="confidence-bar-wrap">
                  <div className="confidence-bar">
                    <div className="confidence-fill" />
                    <div className="confidence-marker" style={{
                      left: `${Math.min(95, Math.max(5,
                        ((result.recommended_price - result.confidence_low) /
                        (result.confidence_high - result.confidence_low)) * 100
                      ))}%`
                    }} />
                  </div>
                  <div className="confidence-bar-labels">
                    <span>${Math.round(result.confidence_low).toLocaleString()}</span>
                    <span>${Math.round(result.confidence_high).toLocaleString()}</span>
                  </div>
                </div>
                {result.rationale.markdown_risk && (
                  <div className="markdown-warning">
                    High markdown risk - listing price near upper confidence bound
                  </div>
                )}
              </div>

              {/* 2. Trend Signal */}
              <div className="trend-block" style={{ borderLeftColor: trendColor }}>
                <div className="trend-header">
                  <span className="trend-icon" style={{ color: trendColor }}>{trendIcon}</span>
                  <span className="trend-direction" style={{ color: trendColor }}>
                    {result.trend_direction.toUpperCase()} TREND
                  </span>
                  <span className="trend-rsv">RSV {result.rationale.trend_rsv}</span>
                </div>
                <p className="trend-signal-text">{result.trend_signal}</p>
              </div>

              {/* 3. Sell-Through Prediction */}
              {result.sell_through && result.sell_through.probability_30_day > 0 && (
                <div className="sell-through-block">
                  <div className="panel-label">SELL-THROUGH VELOCITY</div>
                  <div className="st-headline">
                    <span className="st-big">{Math.round(result.sell_through.probability_30_day * 100)}%</span>
                    <span className="st-caption">probability of selling within 30 days at this price</span>
                  </div>
                  <div className="st-bars">
                    <SellThroughBar label="7 days"  value={result.sell_through.probability_7_day} />
                    <SellThroughBar label="14 days" value={result.sell_through.probability_14_day} />
                    <SellThroughBar label="30 days" value={result.sell_through.probability_30_day} />
                    <SellThroughBar label="60 days" value={result.sell_through.probability_60_day} />
                    <SellThroughBar label="90 days" value={result.sell_through.probability_90_day} />
                  </div>
                  <div className="st-sensitivity">{result.sell_through.price_sensitivity}</div>
                </div>
              )}

              {/* 4. Rationale */}
              <div className="rationale-block">
                <div className="panel-label">RATIONALE</div>
                <div className="rationale-grid">
                  {[
                    ["Brand tier", TIER_LABELS[result.rationale.brand_tier] || result.rationale.brand_tier],
                    ["Condition score", `${result.rationale.condition_score} / 10`],
                    ["Accessories included", result.rationale.accessories_included ? "Yes - premium applied" : "No"],
                    ["Archive item", result.rationale.archive_item ? "Yes - era premium applied" : "No"],
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

              <div className="output-footer">
                Model v{result.model_version} · XGBoost + Cox PH + Google Trends RSV
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}