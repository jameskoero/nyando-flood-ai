import { useState } from "react";

const API_URL = "https://nyando-flood-api.onrender.com/predict";

const WARDS = ["Ahero", "Awasi/Onjiko", "East Kano/Wawidhi", "Kabonyo/Kanyagwal", "Kobura"];

const LAND_COVER_LABELS = {
  0: "Open Water",
  1: "Cropland",
  2: "Grassland",
  3: "Shrubland",
  4: "Wetland",
  5: "Built-up",
};

const DEFAULTS = {
  elevation: 1185,
  slope: 2.4,
  rainfall_3day: 48,
  distance_river: 320,
  clay_percent: 38,
  land_cover: 1,
  ward: "Ahero",
};

const FEATURE_META = {
  elevation:      { label: "Elevation", unit: "m", min: 1100, max: 1400, step: 5, desc: "Terrain height above sea level" },
  slope:          { label: "Slope", unit: "°", min: 0, max: 25, step: 0.1, desc: "Ground gradient angle" },
  rainfall_3day:  { label: "3-Day Rainfall", unit: "mm", min: 0, max: 200, step: 1, desc: "Cumulative rainfall last 3 days" },
  distance_river: { label: "Distance to River", unit: "m", min: 0, max: 2000, step: 10, desc: "Proximity to nearest waterway" },
  clay_percent:   { label: "Clay Content", unit: "%", min: 0, max: 80, step: 1, desc: "Soil clay percentage (affects drainage)" },
};

function getRiskLevel(prediction) {
  if (!prediction) return null;
  const p = typeof prediction === "object" ? prediction.flood_probability ?? prediction.probability ?? null : null;
  const label = typeof prediction === "object" ? prediction.flood_risk ?? prediction.prediction ?? prediction.label ?? null : prediction;

  if (p !== null) {
    if (p >= 0.65) return { level: "HIGH", color: "#E84040", bg: "#2D0A0A", label: "High Flood Risk", emoji: "🔴", prob: p };
    if (p >= 0.35) return { level: "MEDIUM", color: "#F5A623", bg: "#2D1A00", label: "Moderate Flood Risk", emoji: "🟡", prob: p };
    return { level: "LOW", color: "#27AE60", bg: "#0A2D0F", label: "Low Flood Risk", emoji: "🟢", prob: p };
  }

  const str = String(label).toLowerCase();
  if (str.includes("high") || str === "1" || str === "true") return { level: "HIGH", color: "#E84040", bg: "#2D0A0A", label: "High Flood Risk", emoji: "🔴", prob: null };
  if (str.includes("med")) return { level: "MEDIUM", color: "#F5A623", bg: "#2D1A00", label: "Moderate Flood Risk", emoji: "🟡", prob: null };
  return { level: "LOW", color: "#27AE60", bg: "#0A2D0F", label: "Low Flood Risk", emoji: "🟢", prob: null };
}

function NyandoMap({ ward }) {
  const wardCoords = {
    "Ahero":               { lat: -0.165, lng: 34.918 },
    "Awasi/Onjiko":        { lat: -0.198, lng: 34.886 },
    "East Kano/Wawidhi":   { lat: -0.151, lng: 34.945 },
    "Kabonyo/Kanyagwal":   { lat: -0.221, lng: 34.875 },
    "Kobura":              { lat: -0.236, lng: 34.862 },
  };
  const active = wardCoords[ward] || wardCoords["Ahero"];

  const wardDots = Object.entries(wardCoords).map(([name, c]) => ({
    name, ...c,
    x: 50 + (c.lng - 34.9) * 600,
    y: 50 + (c.lat + 0.19) * 600,
  }));

  return (
    <div style={{ position: "relative", width: "100%", height: "220px", background: "linear-gradient(160deg, #0D2137 0%, #0A3020 100%)", borderRadius: "12px", overflow: "hidden", border: "1px solid #1E3A5F" }}>
      {/* Grid lines */}
      <svg width="100%" height="100%" style={{ position: "absolute", inset: 0, opacity: 0.15 }}>
        {[0,25,50,75,100].map(p => (
          <g key={p}>
            <line x1={`${p}%`} y1="0" x2={`${p}%`} y2="100%" stroke="#C9A84C" strokeWidth="0.5" />
            <line x1="0" y1={`${p}%`} x2="100%" y2={`${p}%`} stroke="#C9A84C" strokeWidth="0.5" />
          </g>
        ))}
      </svg>

      {/* Lake Victoria hint */}
      <div style={{ position: "absolute", left: "-20px", top: "30%", width: "90px", height: "80px", background: "rgba(30,100,180,0.35)", borderRadius: "0 50% 50% 0", border: "1px solid rgba(100,180,255,0.3)" }} />
      <span style={{ position: "absolute", left: "4px", top: "50%", transform: "translateY(-50%)", fontSize: "9px", color: "#7EB8F7", fontFamily: "serif", letterSpacing: "0.05em", writingMode: "vertical-rl" }}>LAKE VICTORIA</span>

      {/* River line */}
      <svg width="100%" height="100%" style={{ position: "absolute", inset: 0 }}>
        <path d="M 70 10 Q 130 60 150 110 Q 170 160 200 200 Q 230 240 260 270" stroke="#5BA8F5" strokeWidth="2" fill="none" opacity="0.5" strokeDasharray="4 2" />
        <text x="160" y="75" fill="#5BA8F5" fontSize="8" opacity="0.7">Nyando R.</text>
      </svg>

      {/* Ward dots */}
      <svg width="100%" height="100%" style={{ position: "absolute", inset: 0 }}>
        {wardDots.map(w => {
          const isActive = w.name === ward;
          return (
            <g key={w.name}>
              {isActive && <circle cx={`${w.x}%`} cy={`${w.y}%`} r="18" fill="#C9A84C" opacity="0.15" />}
              <circle cx={`${w.x}%`} cy={`${w.y}%`} r={isActive ? 7 : 4} fill={isActive ? "#C9A84C" : "#5BA8F5"} stroke={isActive ? "#FFE08A" : "#0A1628"} strokeWidth="1.5" />
              <text x={`${w.x + 2}%`} y={`${w.y - 2}%`} fill={isActive ? "#FFE08A" : "#aaa"} fontSize={isActive ? "9" : "7.5"} fontWeight={isActive ? "700" : "400"}>{w.name}</text>
            </g>
          );
        })}
      </svg>

      {/* Labels */}
      <div style={{ position: "absolute", bottom: 8, right: 10, fontSize: "9px", color: "#C9A84C", fontFamily: "monospace", opacity: 0.8 }}>
        0°S 34.9°E · Nyando Sub-County, Kisumu
      </div>
      <div style={{ position: "absolute", top: 8, left: 10, fontSize: "9px", color: "#C9A84C", fontFamily: "monospace", background: "rgba(10,22,40,0.7)", padding: "2px 6px", borderRadius: "4px" }}>
        ▲ NYANDO BASIN
      </div>
    </div>
  );
}

function Slider({ name, value, onChange, meta }) {
  const pct = ((value - meta.min) / (meta.max - meta.min)) * 100;
  return (
    <div style={{ marginBottom: "18px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
        <span style={{ fontSize: "12px", color: "#C9A84C", fontWeight: "600", letterSpacing: "0.04em" }}>{meta.label}</span>
        <span style={{ fontSize: "13px", color: "#fff", fontFamily: "monospace", background: "rgba(201,168,76,0.15)", padding: "1px 8px", borderRadius: "4px" }}>
          {typeof value === "number" ? value.toFixed(meta.step < 1 ? 1 : 0) : value} {meta.unit}
        </span>
      </div>
      <div style={{ position: "relative", height: "6px", background: "#1E3A5F", borderRadius: "3px" }}>
        <div style={{ position: "absolute", left: 0, width: `${pct}%`, height: "100%", background: "linear-gradient(90deg, #0A4A8A, #C9A84C)", borderRadius: "3px", transition: "width 0.1s" }} />
        <input
          type="range" min={meta.min} max={meta.max} step={meta.step}
          value={value}
          onChange={e => onChange(name, parseFloat(e.target.value))}
          style={{ position: "absolute", inset: 0, width: "100%", opacity: 0, cursor: "pointer", height: "100%", margin: 0 }}
        />
      </div>
      <div style={{ fontSize: "10px", color: "#5B7A9A", marginTop: "3px" }}>{meta.desc}</div>
    </div>
  );
}

function RiskGauge({ risk, loading }) {
  const levels = ["LOW", "MEDIUM", "HIGH"];
  const idx = risk ? levels.indexOf(risk.level) : -1;

  return (
    <div style={{ textAlign: "center", padding: "24px 16px" }}>
      {loading ? (
        <div style={{ color: "#5B7A9A", fontSize: "14px" }}>
          <div style={{ display: "inline-block", width: "32px", height: "32px", border: "3px solid #1E3A5F", borderTopColor: "#C9A84C", borderRadius: "50%", animation: "spin 0.8s linear infinite", marginBottom: "12px" }} />
          <div>Querying model…</div>
        </div>
      ) : risk ? (
        <>
          <div style={{ fontSize: "52px", lineHeight: 1, marginBottom: "8px" }}>{risk.emoji}</div>
          <div style={{ fontSize: "22px", fontWeight: "800", color: risk.color, letterSpacing: "0.06em", marginBottom: "4px" }}>
            {risk.label.toUpperCase()}
          </div>
          {risk.prob !== null && (
            <div style={{ fontSize: "13px", color: "#aaa", marginBottom: "16px" }}>
              Flood probability: <span style={{ color: risk.color, fontWeight: "700" }}>{(risk.prob * 100).toFixed(1)}%</span>
            </div>
          )}
          <div style={{ display: "flex", gap: "6px", justifyContent: "center", margin: "12px 0" }}>
            {levels.map((l, i) => (
              <div key={l} style={{
                flex: 1, maxWidth: "80px", height: "8px", borderRadius: "4px",
                background: i <= idx ? (l === "HIGH" ? "#E84040" : l === "MEDIUM" ? "#F5A623" : "#27AE60") : "#1E3A5F",
                transition: "background 0.4s",
              }} />
            ))}
          </div>
          <div style={{ fontSize: "10px", color: "#5B7A9A", letterSpacing: "0.08em" }}>LOW · MEDIUM · HIGH</div>
          <div style={{ marginTop: "16px", padding: "10px 14px", background: "rgba(201,168,76,0.08)", borderRadius: "8px", border: `1px solid ${risk.color}33`, fontSize: "12px", color: "#C0C0C0", lineHeight: 1.5 }}>
            {risk.level === "HIGH" && "⚠️ Immediate alert recommended. Coordinate with Nyando Sub-County emergency office."}
            {risk.level === "MEDIUM" && "⚡ Monitor conditions closely. Consider early warning broadcast to at-risk households."}
            {risk.level === "LOW" && "✅ Conditions stable. Continue routine monitoring via GEE satellite feed."}
          </div>
        </>
      ) : (
        <div style={{ color: "#5B7A9A", fontSize: "13px", lineHeight: 1.7 }}>
          <div style={{ fontSize: "32px", marginBottom: "8px", opacity: 0.4 }}>◎</div>
          Adjust parameters and click<br /><strong style={{ color: "#C9A84C" }}>Run Prediction</strong> to assess flood risk
        </div>
      )}
    </div>
  );
}

export default function NyandoFloodDashboard() {
  const [inputs, setInputs] = useState(DEFAULTS);
  const [risk, setRisk] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [rawResp, setRawResp] = useState(null);
  const [lastRun, setLastRun] = useState(null);

  const handleChange = (name, value) => setInputs(p => ({ ...p, [name]: value }));

  const runPrediction = async () => {
    setLoading(true);
    setError(null);
    setRisk(null);
    try {
      const body = { ...inputs };
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      setRawResp(data);
      if (!res.ok) throw new Error(data?.detail?.[0]?.msg || JSON.stringify(data));
      setRisk(getRiskLevel(data));
      setLastRun(new Date().toLocaleTimeString());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "#060E1A",
      fontFamily: "'Georgia', 'Times New Roman', serif",
      color: "#E8E8E8",
      padding: "0",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Code+Pro:wght@400;600&family=Lato:wght@300;400;700&display=swap');
        * { box-sizing: border-box; }
        input[type=range] { -webkit-appearance: none; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        .card { animation: fadeIn 0.5s ease both; }
        .run-btn:hover { background: linear-gradient(135deg, #C9A84C, #A07830) !important; transform: translateY(-1px); box-shadow: 0 6px 24px rgba(201,168,76,0.4) !important; }
        .run-btn:active { transform: translateY(0); }
        select { color-scheme: dark; }
        ::-webkit-scrollbar { width: 4px; background: #0A1628; }
        ::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 2px; }
      `}</style>

      {/* Header */}
      <div style={{ background: "linear-gradient(135deg, #0A1628 0%, #0D2137 60%, #0A2010 100%)", borderBottom: "1px solid #1E3A5F", padding: "20px 24px 16px" }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: "14px" }}>
          <div style={{ width: "44px", height: "44px", borderRadius: "10px", background: "linear-gradient(135deg, #0A3A8A, #C9A84C)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "20px", flexShrink: 0 }}>🌊</div>
          <div>
            <div style={{ fontFamily: "'Playfair Display', serif", fontSize: "20px", fontWeight: "900", color: "#C9A84C", letterSpacing: "0.02em", lineHeight: 1.1 }}>
              Nyando Flood AI
            </div>
            <div style={{ fontSize: "11px", color: "#5B7A9A", letterSpacing: "0.06em", marginTop: "2px", fontFamily: "'Lato', sans-serif" }}>
              EARLY WARNING SYSTEM · KISUMU COUNTY, KENYA
            </div>
          </div>
        </div>

        {/* Donor context bar */}
        <div style={{ marginTop: "14px", display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "8px" }}>
          {[
            { val: "~50,000", label: "Residents Protected" },
            { val: "AUC 0.97", label: "Model Accuracy" },
            { val: "5 Wards", label: "Coverage Area" },
          ].map(s => (
            <div key={s.label} style={{ background: "rgba(201,168,76,0.07)", border: "1px solid rgba(201,168,76,0.2)", borderRadius: "8px", padding: "8px 10px", textAlign: "center" }}>
              <div style={{ fontFamily: "'Playfair Display', serif", fontSize: "15px", fontWeight: "700", color: "#C9A84C" }}>{s.val}</div>
              <div style={{ fontSize: "9px", color: "#5B7A9A", letterSpacing: "0.06em", fontFamily: "'Lato', sans-serif" }}>{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ padding: "16px" }}>

        {/* Map */}
        <div className="card" style={{ marginBottom: "16px" }}>
          <NyandoMap ward={inputs.ward} />
        </div>

        {/* Ward + Land Cover */}
        <div className="card" style={{ background: "#0A1628", border: "1px solid #1E3A5F", borderRadius: "12px", padding: "16px", marginBottom: "16px" }}>
          <div style={{ fontSize: "11px", color: "#C9A84C", letterSpacing: "0.1em", marginBottom: "12px", fontFamily: "'Lato', sans-serif", fontWeight: "700" }}>📍 LOCATION & LAND USE</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
            <div>
              <label style={{ fontSize: "11px", color: "#5B7A9A", display: "block", marginBottom: "6px", fontFamily: "'Lato', sans-serif" }}>Ward</label>
              <select value={inputs.ward} onChange={e => handleChange("ward", e.target.value)}
                style={{ width: "100%", background: "#0D2137", border: "1px solid #1E3A5F", color: "#E8E8E8", borderRadius: "6px", padding: "8px 10px", fontSize: "13px", fontFamily: "'Lato', sans-serif" }}>
                {WARDS.map(w => <option key={w}>{w}</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: "11px", color: "#5B7A9A", display: "block", marginBottom: "6px", fontFamily: "'Lato', sans-serif" }}>Land Cover</label>
              <select value={inputs.land_cover} onChange={e => handleChange("land_cover", parseInt(e.target.value))}
                style={{ width: "100%", background: "#0D2137", border: "1px solid #1E3A5F", color: "#E8E8E8", borderRadius: "6px", padding: "8px 10px", fontSize: "13px", fontFamily: "'Lato', sans-serif" }}>
                {Object.entries(LAND_COVER_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </div>
          </div>
        </div>

        {/* Sliders */}
        <div className="card" style={{ background: "#0A1628", border: "1px solid #1E3A5F", borderRadius: "12px", padding: "16px", marginBottom: "16px" }}>
          <div style={{ fontSize: "11px", color: "#C9A84C", letterSpacing: "0.1em", marginBottom: "14px", fontFamily: "'Lato', sans-serif", fontWeight: "700" }}>⚙️ ENVIRONMENTAL PARAMETERS</div>
          {Object.entries(FEATURE_META).map(([name, meta]) => (
            <Slider key={name} name={name} value={inputs[name]} onChange={handleChange} meta={meta} />
          ))}
        </div>

        {/* Run button */}
        <button className="run-btn" onClick={runPrediction} disabled={loading}
          style={{ width: "100%", padding: "16px", background: loading ? "#1E3A5F" : "linear-gradient(135deg, #0A4A8A, #C9A84C)", border: "none", borderRadius: "10px", color: "#fff", fontSize: "15px", fontWeight: "700", letterSpacing: "0.1em", cursor: loading ? "not-allowed" : "pointer", transition: "all 0.2s", fontFamily: "'Lato', sans-serif", marginBottom: "16px" }}>
          {loading ? "⏳  RUNNING MODEL…" : "🚀  RUN FLOOD PREDICTION"}
        </button>

        {/* Risk result */}
        <div className="card" style={{ background: risk ? risk.bg : "#0A1628", border: `1px solid ${risk ? risk.color + "44" : "#1E3A5F"}`, borderRadius: "12px", marginBottom: "16px", transition: "all 0.4s" }}>
          <RiskGauge risk={risk} loading={loading} />
          {lastRun && !loading && <div style={{ textAlign: "center", fontSize: "10px", color: "#5B7A9A", paddingBottom: "12px", fontFamily: "monospace" }}>Last run: {lastRun}</div>}
        </div>

        {/* Error */}
        {error && (
          <div style={{ background: "#2D0A0A", border: "1px solid #E84040", borderRadius: "10px", padding: "12px 14px", marginBottom: "16px", fontSize: "12px", color: "#FF8080", fontFamily: "monospace" }}>
            ⚠️ API Error: {error}
            <div style={{ marginTop: "6px", color: "#5B7A9A", fontSize: "11px" }}>Note: Render free tier may take 30–60s to cold-start. Try again.</div>
          </div>
        )}

        {/* Raw response debug */}
        {rawResp && (
          <details style={{ marginBottom: "16px" }}>
            <summary style={{ fontSize: "11px", color: "#5B7A9A", cursor: "pointer", padding: "8px 0", fontFamily: "monospace" }}>🔍 Raw API Response</summary>
            <pre style={{ background: "#0D1F30", border: "1px solid #1E3A5F", borderRadius: "8px", padding: "10px", fontSize: "11px", color: "#7EB8F7", overflow: "auto", marginTop: "8px" }}>
              {JSON.stringify(rawResp, null, 2)}
            </pre>
          </details>
        )}

        {/* Donor context */}
        <div className="card" style={{ background: "linear-gradient(135deg, #0A1628, #0D2010)", border: "1px solid rgba(201,168,76,0.25)", borderRadius: "12px", padding: "16px", marginBottom: "16px" }}>
          <div style={{ fontSize: "11px", color: "#C9A84C", letterSpacing: "0.1em", marginBottom: "10px", fontFamily: "'Lato', sans-serif", fontWeight: "700" }}>🌍 ABOUT THIS PROJECT</div>
          <p style={{ fontSize: "12px", color: "#A0B0C0", lineHeight: 1.7, margin: "0 0 10px", fontFamily: "'Lato', sans-serif" }}>
            The Nyando River basin floods annually, displacing thousands of farming families. This AI system was built to provide <strong style={{ color: "#C9A84C" }}>ward-level flood risk predictions</strong> using satellite-derived terrain, soil, and rainfall data from Google Earth Engine.
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
            {[
              { icon: "🛰️", text: "2,308 GEE satellite points" },
              { icon: "🏘️", text: "161,000+ residents covered" },
              { icon: "📊", text: "GradientBoosting, AUC 0.97" },
              { icon: "🏛️", text: "UNDP/USAID/GCF eligible" },
            ].map(d => (
              <div key={d.text} style={{ display: "flex", gap: "6px", alignItems: "center", fontSize: "11px", color: "#7A9AB0", fontFamily: "'Lato', sans-serif" }}>
                <span>{d.icon}</span><span>{d.text}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div style={{ textAlign: "center", fontSize: "10px", color: "#2A4A6A", fontFamily: "monospace", paddingBottom: "24px", lineHeight: 1.8 }}>
          Built by James Koero · github.com/jameskoero/nyando-flood-ai<br />
          Model: GradientBoostingClassifier · F1=0.90 · CV=0.97±0.004<br />
          API: nyando-flood-api.onrender.com
        </div>
      </div>
    </div>
  );
}
