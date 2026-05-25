import { useState, useCallback } from "react";

const CITIES = [
  { name: "New York, NY", lat: 40.7128, lon: -74.006 },
  { name: "Los Angeles, CA", lat: 34.0522, lon: -118.2437 },
  { name: "Chicago, IL", lat: 41.8781, lon: -87.6298 },
  { name: "Houston, TX", lat: 29.7604, lon: -95.3698 },
  { name: "Phoenix, AZ", lat: 33.4484, lon: -112.074 },
  { name: "Philadelphia, PA", lat: 39.9526, lon: -75.1652 },
  { name: "San Antonio, TX", lat: 29.4241, lon: -98.4936 },
  { name: "San Diego, CA", lat: 32.7157, lon: -117.1611 },
  { name: "Dallas, TX", lat: 32.7767, lon: -96.797 },
  { name: "Denver, CO", lat: 39.7392, lon: -104.9903 },
  { name: "Seattle, WA", lat: 47.6062, lon: -122.3321 },
  { name: "Miami, FL", lat: 25.7617, lon: -80.1918 },
  { name: "Atlanta, GA", lat: 33.749, lon: -84.388 },
  { name: "Boston, MA", lat: 42.3601, lon: -71.0589 },
  { name: "Minneapolis, MN", lat: 44.9778, lon: -93.265 },
];

async function fetchHistoricalData(city, startDate, endDate) {
  const url =
    `https://archive-api.open-meteo.com/v1/archive?` +
    `latitude=${city.lat}&longitude=${city.lon}` +
    `&start_date=${startDate}&end_date=${endDate}` +
    `&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max` +
    `&temperature_unit=fahrenheit&timezone=auto`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const data = await res.json();
  return data.daily;
}

function buildFeatures(daily, idx) {
  const date = new Date(daily.time[idx]);
  const dayOfYear = Math.floor((date - new Date(date.getFullYear(), 0, 0)) / 86400000);
  const sinDay = Math.sin(2 * Math.PI * dayOfYear / 365);
  const cosDay = Math.cos(2 * Math.PI * dayOfYear / 365);
  const prevMax = idx > 0 ? daily.temperature_2m_max[idx - 1] : daily.temperature_2m_max[idx];
  const prevMin = idx > 0 ? daily.temperature_2m_min[idx - 1] : daily.temperature_2m_min[idx];
  const wind = daily.windspeed_10m_max[idx] || 0;
  const precip = daily.precipitation_sum[idx] || 0;
  const rolling3 = [idx, idx - 1, idx - 2]
    .filter((i) => i >= 0)
    .reduce((s, i) => s + daily.temperature_2m_max[i], 0) / Math.min(3, idx + 1);
  const rolling7 = [idx, idx-1, idx-2, idx-3, idx-4, idx-5, idx-6]
    .filter((i) => i >= 0)
    .reduce((s, i) => s + daily.temperature_2m_max[i], 0) / Math.min(7, idx + 1);
  return [sinDay, cosDay, prevMax, prevMin, wind, precip, rolling3, rolling7];
}

class SimpleGBM {
  constructor(nTrees = 80, lr = 0.08, maxDepth = 4) {
    this.nTrees = nTrees;
    this.lr = lr;
    this.maxDepth = maxDepth;
    this.trees = [];
    this.basePred = 0;
  }

  buildTree(X, residuals, depth) {
    if (depth === 0 || X.length <= 2) {
      return { leaf: true, value: residuals.reduce((a, b) => a + b, 0) / residuals.length };
    }
    let bestFeat = 0, bestThresh = 0, bestGain = -Infinity;
    for (let f = 0; f < X[0].length; f++) {
      const vals = [...new Set(X.map((x) => x[f]))].sort((a, b) => a - b);
      for (let t = 0; t < vals.length - 1; t++) {
        const thresh = (vals[t] + vals[t + 1]) / 2;
        const lr2 = [], rr = [];
        X.forEach((x, i) => {
          (x[f] <= thresh ? lr2 : rr).push(residuals[i]);
        });
        if (!lr2.length || !rr.length) continue;
        const lm = lr2.reduce((a, b) => a + b, 0) / lr2.length;
        const rm = rr.reduce((a, b) => a + b, 0) / rr.length;
        const gain =
          lr2.reduce((s, r) => s - (r - lm) ** 2, 0) +
          rr.reduce((s, r) => s - (r - rm) ** 2, 0);
        if (gain > bestGain) { bestGain = gain; bestFeat = f; bestThresh = thresh; }
      }
    }
    const leftIdx = [], rightIdx = [];
    X.forEach((x, i) => (x[bestFeat] <= bestThresh ? leftIdx : rightIdx).push(i));
    if (!leftIdx.length || !rightIdx.length) {
      return { leaf: true, value: residuals.reduce((a, b) => a + b, 0) / residuals.length };
    }
    return {
      leaf: false, feat: bestFeat, thresh: bestThresh,
      left: this.buildTree(leftIdx.map((i) => X[i]), leftIdx.map((i) => residuals[i]), depth - 1),
      right: this.buildTree(rightIdx.map((i) => X[i]), rightIdx.map((i) => residuals[i]), depth - 1),
    };
  }

  predictTree(tree, x) {
    if (tree.leaf) return tree.value;
    return x[tree.feat] <= tree.thresh
      ? this.predictTree(tree.left, x)
      : this.predictTree(tree.right, x);
  }

  fit(X, y) {
    this.basePred = y.reduce((a, b) => a + b, 0) / y.length;
    let preds = new Array(X.length).fill(this.basePred);
    for (let t = 0; t < this.nTrees; t++) {
      const residuals = y.map((yi, i) => yi - preds[i]);
      const tree = this.buildTree(X, residuals, this.maxDepth);
      this.trees.push(tree);
      preds = preds.map((p, i) => p + this.lr * this.predictTree(tree, X[i]));
    }
  }

  predict(X) {
    return X.map((x) =>
      this.trees.reduce((p, t) => p + this.lr * this.predictTree(t, x), this.basePred)
    );
  }
}

function computeMetrics(actual, predicted) {
  const n = actual.length;
  const mae = actual.reduce((s, a, i) => s + Math.abs(a - predicted[i]), 0) / n;
  const rmse = Math.sqrt(actual.reduce((s, a, i) => s + (a - predicted[i]) ** 2, 0) / n);
  const bias = actual.reduce((s, a, i) => s + (predicted[i] - a), 0) / n;
  return { mae: mae.toFixed(2), rmse: rmse.toFixed(2), bias: bias.toFixed(2), n };
}

// Naive baseline: yesterday's high = tomorrow's high
function naivePreds(daily, startIdx, count) {
  return Array.from({ length: count }, (_, i) => daily.temperature_2m_max[startIdx + i - 1]);
}

const ACCENT = "#00e5c3";
const BG = "#0a0f1e";
const PANEL = "#111827";
const BORDER = "#1e2d45";
const TEXT = "#e2e8f0";
const MUTED = "#64748b";

function MiniChart({ data }) {
  const W = 900, H = 220, PAD = { top: 12, right: 12, bottom: 32, left: 40 };
  const allVals = data.flatMap((d) => [d.actual, d.model, d.naive]);
  const minV = Math.min(...allVals) - 3;
  const maxV = Math.max(...allVals) + 3;
  const xScale = (i) => PAD.left + (i / (data.length - 1)) * (W - PAD.left - PAD.right);
  const yScale = (v) => PAD.top + (1 - (v - minV) / (maxV - minV)) * (H - PAD.top - PAD.bottom);

  const line = (key, color, dash = "") => {
    const pts = data.map((d, i) => `${xScale(i)},${yScale(d[key])}`).join(" ");
    return <polyline points={pts} fill="none" stroke={color} strokeWidth={1.8} strokeDasharray={dash} strokeLinejoin="round" />;
  };

  const tickEvery = Math.ceil(data.length / 7);
  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div style={{ overflowX: "auto" }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
        {yTicks.map((t) => {
          const y = PAD.top + t * (H - PAD.top - PAD.bottom);
          const val = Math.round(maxV - t * (maxV - minV));
          return (
            <g key={t}>
              <line x1={PAD.left} x2={W - PAD.right} y1={y} y2={y} stroke="#1e2d45" strokeWidth={1} />
              <text x={PAD.left - 5} y={y + 4} textAnchor="end" fontSize={10} fill="#475569">{val}°</text>
            </g>
          );
        })}
        {line("naive", "#f97316", "4 2")}
        {line("model", ACCENT)}
        {line("actual", "#94a3b8")}
        {data.filter((_, i) => i % tickEvery === 0).map((d, i) => {
          const idx = data.indexOf(d);
          return (
            <text key={i} x={xScale(idx)} y={H - 6} textAnchor="middle" fontSize={9} fill="#475569">
              {d.date.slice(5)}
            </text>
          );
        })}
      </svg>
    </div>
  );
}

export default function App() {
  const [selectedCity, setSelectedCity] = useState(CITIES[0]);
  const [months, setMonths] = useState(12);
  const [status, setStatus] = useState("idle");
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [log, setLog] = useState([]);

  const addLog = (msg) => setLog((prev) => [...prev, msg]);

  const run = useCallback(async () => {
    setStatus("loading");
    setResults(null);
    setError("");
    setLog([]);

    try {
      const endDate = new Date();
      endDate.setDate(endDate.getDate() - 7);
      const startDate = new Date(endDate);
      startDate.setMonth(startDate.getMonth() - months);
      const fmt = (d) => d.toISOString().split("T")[0];

      addLog(`📡 Fetching ${months} months of data for ${selectedCity.name}...`);
      const daily = await fetchHistoricalData(selectedCity, fmt(startDate), fmt(endDate));
      addLog(`✅ Retrieved ${daily.time.length} days of historical data`);

      const n = daily.time.length;
      const splitIdx = Math.floor(n * 0.75);

      addLog("🔧 Engineering features (seasonality, lag temps, rolling averages, wind, precip)...");
      const X = [], y = [];
      for (let i = 1; i < n; i++) {
        X.push(buildFeatures(daily, i));
        y.push(daily.temperature_2m_max[i]);
      }

      const Xtrain = X.slice(0, splitIdx - 1);
      const ytrain = y.slice(0, splitIdx - 1);
      const Xtest = X.slice(splitIdx - 1);
      const ytest = y.slice(splitIdx - 1);
      const testDates = daily.time.slice(splitIdx);

      addLog(`🧠 Training Gradient Boosting model on ${Xtrain.length} days...`);
      const model = new SimpleGBM(80, 0.08, 4);
      model.fit(Xtrain, ytrain);

      addLog(`📊 Backtesting on ${ytest.length} held-out days...`);
      const modelPreds = model.predict(Xtest);
      const naiveP = naivePreds(daily, splitIdx, ytest.length);

      const modelMetrics = computeMetrics(ytest, modelPreds);
      const naiveMetrics = computeMetrics(ytest, naiveP);

      const chartN = Math.min(90, ytest.length);
      const chartData = testDates.slice(-chartN).map((d, i) => ({
        date: d,
        actual: Math.round(ytest[ytest.length - chartN + i]),
        model: Math.round(modelPreds[modelPreds.length - chartN + i]),
        naive: Math.round(naiveP[naiveP.length - chartN + i]),
      }));

      addLog("✅ Complete!");
      setResults({ modelMetrics, naiveMetrics, chartData, testN: ytest.length, city: selectedCity.name });
      setStatus("done");
    } catch (e) {
      setError(e.message);
      setStatus("error");
    }
  }, [selectedCity, months]);

  const improvement = results
    ? ((results.naiveMetrics.mae - results.modelMetrics.mae) / results.naiveMetrics.mae * 100).toFixed(1)
    : null;

  return (
    <div style={{ background: BG, minHeight: "100vh", color: TEXT, fontFamily: "'IBM Plex Mono', monospace", padding: "28px 20px" }}>
      <div style={{ maxWidth: 920, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 36 }}>
          <div style={{ fontSize: 10, color: ACCENT, letterSpacing: "0.25em", marginBottom: 10 }}>WEATHER · ML · BACKTESTING</div>
          <h1 style={{ margin: 0, fontSize: 32, fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, lineHeight: 1.15 }}>
            Next-Day High Temp<br />
            <span style={{ color: ACCENT }}>Prediction Engine</span>
          </h1>
          <p style={{ color: MUTED, fontSize: 12, marginTop: 10, maxWidth: 520, lineHeight: 1.7 }}>
            Gradient Boosting model (MOS technique) trained on real Open-Meteo archive data.
            Backtested on a chronological holdout set vs. a persistence baseline.
          </p>
        </div>

        {/* Config */}
        <div style={{ background: PANEL, border: `1px solid ${BORDER}`, borderRadius: 12, padding: 22, marginBottom: 20 }}>
          <div style={{ fontSize: 10, color: ACCENT, letterSpacing: "0.2em", marginBottom: 18 }}>CONFIGURATION</div>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-end" }}>
            <div style={{ flex: "1 1 240px" }}>
              <label style={{ fontSize: 10, color: MUTED, display: "block", marginBottom: 7, letterSpacing: "0.1em" }}>CITY</label>
              <select
                value={selectedCity.name}
                onChange={(e) => setSelectedCity(CITIES.find((c) => c.name === e.target.value))}
                style={{ width: "100%", background: BG, border: `1px solid ${BORDER}`, color: TEXT, padding: "10px 12px", borderRadius: 8, fontSize: 13, fontFamily: "inherit" }}
              >
                {CITIES.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
            </div>
            <div style={{ flex: "1 1 160px" }}>
              <label style={{ fontSize: 10, color: MUTED, display: "block", marginBottom: 7, letterSpacing: "0.1em" }}>TRAINING WINDOW</label>
              <select
                value={months}
                onChange={(e) => setMonths(Number(e.target.value))}
                style={{ width: "100%", background: BG, border: `1px solid ${BORDER}`, color: TEXT, padding: "10px 12px", borderRadius: 8, fontSize: 13, fontFamily: "inherit" }}
              >
                <option value={3}>3 months</option>
                <option value={6}>6 months</option>
                <option value={12}>12 months</option>
                <option value={24}>24 months</option>
              </select>
            </div>
            <button
              onClick={run}
              disabled={status === "loading"}
              style={{
                flex: "0 0 auto", background: status === "loading" ? BORDER : ACCENT,
                color: BG, border: "none", padding: "10px 28px", borderRadius: 8,
                fontWeight: 700, fontSize: 13, cursor: status === "loading" ? "not-allowed" : "pointer",
                letterSpacing: "0.08em", fontFamily: "inherit", transition: "all 0.2s",
              }}
            >
              {status === "loading" ? "RUNNING..." : "▶  RUN BACKTEST"}
            </button>
          </div>
        </div>

        {/* Log */}
        {log.length > 0 && (
          <div style={{ background: "#050a14", border: `1px solid ${BORDER}`, borderRadius: 8, padding: 16, marginBottom: 20, fontSize: 12 }}>
            {log.map((l, i) => (
              <div key={i} style={{ color: i === log.length - 1 && status === "loading" ? ACCENT : MUTED, marginBottom: 4 }}>
                {l}
              </div>
            ))}
          </div>
        )}

        {error && (
          <div style={{ background: "#1a0808", border: "1px solid #e53e3e", borderRadius: 8, padding: 14, marginBottom: 20, color: "#fc8181", fontSize: 12 }}>
            ⚠ {error}
          </div>
        )}

        {/* Results */}
        {results && (
          <div>
            <div style={{ fontSize: 10, color: ACCENT, letterSpacing: "0.2em", marginBottom: 14 }}>
              BACKTEST RESULTS — {results.city} — {results.testN} TEST DAYS
            </div>

            {/* Metric cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, marginBottom: 20 }}>
              {[
                { label: "GBM MAE", value: `${results.modelMetrics.mae}°F`, sub: "Model mean abs error", hi: true },
                { label: "Baseline MAE", value: `${results.naiveMetrics.mae}°F`, sub: "Persistence baseline", hi: false },
                { label: "Improvement", value: `${improvement}%`, sub: "vs baseline", hi: Number(improvement) > 0 },
                { label: "GBM RMSE", value: `${results.modelMetrics.rmse}°F`, sub: "Root mean sq error", hi: false },
                { label: "Model Bias", value: `${results.modelMetrics.bias}°F`, sub: "Systematic offset", hi: false },
              ].map((m) => (
                <div key={m.label} style={{
                  background: PANEL, border: `1px solid ${m.hi ? ACCENT + "66" : BORDER}`,
                  borderRadius: 10, padding: 18, transition: "all 0.2s",
                }}>
                  <div style={{ fontSize: 9, color: MUTED, letterSpacing: "0.12em", marginBottom: 8 }}>{m.label}</div>
                  <div style={{ fontSize: 28, fontWeight: 700, color: m.hi ? ACCENT : TEXT, lineHeight: 1 }}>{m.value}</div>
                  <div style={{ fontSize: 10, color: MUTED, marginTop: 5 }}>{m.sub}</div>
                </div>
              ))}
            </div>

            {/* Chart */}
            <div style={{ background: PANEL, border: `1px solid ${BORDER}`, borderRadius: 12, padding: 22, marginBottom: 20 }}>
              <div style={{ fontSize: 10, color: ACCENT, letterSpacing: "0.2em", marginBottom: 16 }}>
                FORECAST vs ACTUAL — LAST {results.chartData.length} TEST DAYS
              </div>
              <MiniChart data={results.chartData} />
              <div style={{ display: "flex", gap: 24, marginTop: 14, fontSize: 11, flexWrap: "wrap" }}>
                {[
                  { color: ACCENT, label: "GBM Model", dash: false },
                  { color: "#f97316", label: "Persistence Baseline", dash: true },
                  { color: "#94a3b8", label: "Actual High", dash: false },
                ].map((l) => (
                  <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <svg width="24" height="8">
                      <line x1="0" y1="4" x2="24" y2="4" stroke={l.color} strokeWidth="2" strokeDasharray={l.dash ? "4 2" : ""} />
                    </svg>
                    <span style={{ color: MUTED }}>{l.label}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Notes */}
            <div style={{ background: PANEL, border: `1px solid ${BORDER}`, borderRadius: 12, padding: 22, fontSize: 12, lineHeight: 1.8 }}>
              <div style={{ fontSize: 10, color: ACCENT, letterSpacing: "0.2em", marginBottom: 14 }}>MODEL NOTES</div>
              <div style={{ color: MUTED }}>
                <p style={{ margin: "0 0 8px" }}>• <strong style={{ color: TEXT }}>Data source:</strong> Open-Meteo historical archive (ERA5 reanalysis) — free, no API key required.</p>
                <p style={{ margin: "0 0 8px" }}>• <strong style={{ color: TEXT }}>Features:</strong> Day-of-year (sin/cos), yesterday's high/low, 3-day and 7-day rolling averages, wind speed, precipitation.</p>
                <p style={{ margin: "0 0 8px" }}>• <strong style={{ color: TEXT }}>Baseline:</strong> Persistence model — "tomorrow's high = today's high." Simple but hard to beat.</p>
                <p style={{ margin: 0 }}>• <strong style={{ color: TEXT }}>Split:</strong> 75% training / 25% test, chronological order — no data leakage.</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
