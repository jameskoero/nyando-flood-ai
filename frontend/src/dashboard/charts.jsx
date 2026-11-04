import { f2, f3, f4, int, pct1, signed4, spansZero } from "./fmt.js";

const RISK_FILL = { LOW: "var(--low)", MEDIUM: "var(--med)", HIGH: "var(--high)" };

// All three charts are drawn on a 360-unit canvas so their text stays near 11 px on a phone.
export function AucBars({ order, items, selected, onSelect }) {
  const W = 360;
  const left = 112;
  const right = 50;
  const top = 12;
  const rowH = 60;
  const H = top + order.length * rowH + 26;
  const x = (v) => left + v * (W - left - right);
  const desc = order
    .map((n) => (items[n].estimable ? n + ": " + f3(items[n].auc.mean) + " over " + items[n].auc.events + " events" : n + ": not estimable"))
    .join("; ");
  return (
    <svg className="chart" viewBox={"0 0 " + W + " " + H} role="img" aria-label={"Per-event mean AUC by ward. " + desc}>
      {[0, 0.5, 1].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top - 4} y2={H - 22} className="grid" />
          <text x={x(t)} y={H - 6} textAnchor="middle" className="tick">{f2(t)}</text>
        </g>
      ))}
      {order.map((name, i) => {
        const w = items[name];
        const y = top + i * rowH;
        const on = selected === name;
        return (
          <g
            key={name}
            role="button"
            tabIndex={0}
            aria-pressed={on}
            aria-label={"Select " + name}
            className={"rowbtn" + (on ? " on" : "")}
            onClick={() => onSelect(name)}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(name)}
          >
            <rect x="0" y={y - 4} width={W} height={rowH - 6} rx="6" className="rowbg" />
            <text x={left - 8} y={y + (name.includes("/") ? 10 : 15)} textAnchor="end" className="lab">
              {(name.includes("/") ? [name.split("/")[0] + "/", name.split("/")[1]] : [name]).map((part, k) => (
                <tspan key={part} x={left - 8} dy={k ? 13 : 0}>{part}</tspan>
              ))}
            </text>
            {w.estimable ? (
              <g>
                <rect x={left} y={y + 4} width={Math.max(0, x(w.auc.mean) - left)} height="16" rx="3" className="bar" />
                <line x1={x(w.auc.ci_low)} x2={x(w.auc.ci_high)} y1={y + 12} y2={y + 12} className="whisker" />
                <text x={W - 6} y={y + 16} textAnchor="end" className="val">{f3(w.auc.mean)}</text>
                <text x="8" y={y + 40} className="sub">{w.auc.events} events, {w.flood_rows} flood and {w.control_rows} control rows</text>
              </g>
            ) : (
              <g>
                <text x={W - 6} y={y + 16} textAnchor="end" className="val">n/a</text>
                <text x={left} y={y + 16} className="val">Not estimable</text>
                <text x="8" y={y + 40} className="sub">{w.reason || "too few rows"} ({w.events_both} event with both classes)</text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}

// Paired differences with two-sided intervals. A row is "supported" only when its interval excludes zero.
export function IntervalPlot({ rows }) {
  const W = 360;
  const pad = 16;
  const rowH = 88;
  const top = 12;
  const H = top + rows.length * rowH + 22;
  const vals = rows.flatMap((r) => [r.lo, r.hi, 0]);
  const span = Math.max(...vals) - Math.min(...vals);
  const lo = Math.min(...vals) - span * 0.1;
  const hi = Math.max(...vals) + span * 0.1;
  const x = (v) => pad + ((v - lo) / (hi - lo)) * (W - 2 * pad);
  const clamp = (v) => Math.max(96, Math.min(W - 96, v));
  const desc = rows
    .map((r) => r.label + ": " + signed4(r.mean) + ", interval " + signed4(r.lo) + " to " + signed4(r.hi) + (spansZero(r.lo, r.hi) ? ", advantage not shown" : ", advantage shown"))
    .join("; ");
  return (
    <svg className="chart" viewBox={"0 0 " + W + " " + H} role="img" aria-label={"Difference in per-event AUC against elevation alone. " + desc}>
      <line x1={x(0)} x2={x(0)} y1={top - 4} y2={H - 20} className="zero" />
      <text x={x(0)} y={H - 6} textAnchor="middle" className="tick">no difference</text>
      {rows.map((r, i) => {
        const y = top + i * rowH;
        const ok = !spansZero(r.lo, r.hi);
        return (
          <g key={r.label}>
            <text x={pad} y={y + 12} className="lab">{r.label} ({r.n} events)</text>
            <line x1={x(r.lo)} x2={x(r.hi)} y1={y + 34} y2={y + 34} className={ok ? "whisker ok" : "whisker no"} />
            <circle cx={x(r.mean)} cy={y + 34} r="5.5" className={ok ? "dot ok" : "dot no"} />
            <text x={clamp(x(r.mean))} y={y + 58} textAnchor="middle" className="val">
              {signed4(r.mean)} [{signed4(r.lo)}, {signed4(r.hi)}]
            </text>
            <text x={clamp(x(r.mean))} y={y + 74} textAnchor="middle" className="sub">{ok ? "Advantage shown" : "Advantage not shown"}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function RiskBars({ rates, cuts }) {
  const W = 360;
  const pad = 8;
  const rowH = 64;
  const top = 8;
  const bands = ["LOW", "MEDIUM", "HIGH"];
  const H = top + bands.length * rowH;
  const range = {
    LOW: "score below " + f4(cuts.low),
    MEDIUM: "score " + f4(cuts.low) + " to " + f4(cuts.high),
    HIGH: "score above " + f4(cuts.high),
  };
  const desc = bands.map((b) => b + ": " + pct1(rates[b].share) + " of " + int(rates[b].rows) + " rows flooded").join("; ");
  return (
    <svg className="chart" viewBox={"0 0 " + W + " " + H} role="img" aria-label={"Observed flooded share by risk class in the evaluation sample. " + desc}>
      {bands.map((b, i) => {
        const y = top + i * rowH;
        const full = W - 2 * pad;
        return (
          <g key={b}>
            <text x={pad} y={y + 14} className="lab">{b}</text>
            <text x={W - pad} y={y + 14} textAnchor="end" className="val">{pct1(rates[b].share)}</text>
            <rect x={pad} y={y + 22} width={full} height="14" rx="7" className="track" />
            <rect x={pad} y={y + 22} width={Math.max(4, rates[b].share * full)} height="14" rx="7" style={{ fill: RISK_FILL[b] }} />
            <text x={pad} y={y + 54} className="sub">{range[b]}, {int(rates[b].flood)} of {int(rates[b].rows)} rows flooded</text>
          </g>
        );
      })}
    </svg>
  );
}
