import { f2, f3, f4, int, pct1, signed4, spansZero } from "./fmt.js";

const RISK_FILL = { LOW: "var(--low)", MEDIUM: "var(--med)", HIGH: "var(--high)" };

export function AucBars({ order, items, selected, onSelect }) {
  const W = 600;
  const left = 176;
  const right = 24;
  const top = 26;
  const rowH = 48;
  const H = top + order.length * rowH + 28;
  const x = (v) => left + v * (W - left - right);
  const desc = order
    .map((n) => (items[n].estimable ? n + ": " + f3(items[n].auc.mean) + " over " + items[n].auc.events + " events" : n + ": not estimable"))
    .join("; ");
  return (
    <svg className="chart" viewBox={"0 0 " + W + " " + H} role="img" aria-label={"Per-event mean AUC by ward. " + desc}>
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top - 8} y2={H - 26} className="grid" />
          <text x={x(t)} y={H - 8} textAnchor="middle" className="tick">{f2(t)}</text>
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
            <rect x="0" y={y - 6} width={W} height={rowH - 6} className="rowbg" />
            <text x={left - 10} y={y + 16} textAnchor="end" className="lab">{name}</text>
            {w.estimable ? (
              <g>
                <rect x={left} y={y + 4} width={Math.max(0, x(w.auc.mean) - left)} height="18" className="bar" />
                <line x1={x(w.auc.ci_low)} x2={x(w.auc.ci_high)} y1={y + 13} y2={y + 13} className="whisker" />
                <text x={x(w.auc.mean) + 8} y={y + 17} className="val">{f3(w.auc.mean)}</text>
                <text x={left} y={y + 36} className="sub">{w.auc.events} events, {w.flood_rows} flood and {w.control_rows} control rows</text>
              </g>
            ) : (
              <g>
                <text x={left} y={y + 17} className="val">Not estimable</text>
                <text x={left} y={y + 36} className="sub">{w.reason || "too few rows"} ({w.events_both} event with both classes)</text>
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
  const W = 600;
  const left = 150;
  const right = 24;
  const rowH = 64;
  const top = 22;
  const H = top + rows.length * rowH + 20;
  const vals = rows.flatMap((r) => [r.lo, r.hi, 0]);
  const pad = (Math.max(...vals) - Math.min(...vals)) * 0.1;
  const lo = Math.min(...vals) - pad;
  const hi = Math.max(...vals) + pad;
  const x = (v) => left + ((v - lo) / (hi - lo)) * (W - left - right);
  const desc = rows
    .map((r) => r.label + ": " + signed4(r.mean) + ", interval " + signed4(r.lo) + " to " + signed4(r.hi) + (spansZero(r.lo, r.hi) ? ", advantage not shown" : ", advantage shown"))
    .join("; ");
  return (
    <svg className="chart" viewBox={"0 0 " + W + " " + H} role="img" aria-label={"Difference in per-event AUC against elevation alone. " + desc}>
      <line x1={x(0)} x2={x(0)} y1={top - 10} y2={H - 14} className="zero" />
      <text x={x(0)} y={H - 2} textAnchor="middle" className="tick">no difference</text>
      {rows.map((r, i) => {
        const y = top + i * rowH;
        const ok = !spansZero(r.lo, r.hi);
        return (
          <g key={r.label}>
            <text x={left - 10} y={y + 5} textAnchor="end" className="lab">{r.label}</text>
            <text x={left - 10} y={y + 22} textAnchor="end" className="sub">{r.n} events</text>
            <line x1={x(r.lo)} x2={x(r.hi)} y1={y} y2={y} className={ok ? "whisker ok" : "whisker no"} />
            <circle cx={x(r.mean)} cy={y} r="5" className={ok ? "dot ok" : "dot no"} />
            <text x={x(r.mean)} y={y + 26} textAnchor="middle" className="val">
              {signed4(r.mean)} [{signed4(r.lo)}, {signed4(r.hi)}]
            </text>
            <text x={x(r.mean)} y={y + 41} textAnchor="middle" className="sub">{ok ? "Advantage shown" : "Advantage not shown"}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function RiskBars({ rates, cuts }) {
  const W = 600;
  const left = 96;
  const right = 150;
  const rowH = 54;
  const top = 16;
  const bands = ["LOW", "MEDIUM", "HIGH"];
  const H = top + bands.length * rowH + 22;
  const x = (v) => left + v * (W - left - right);
  const range = {
    LOW: "score below " + f4(cuts.low),
    MEDIUM: f4(cuts.low) + " to " + f4(cuts.high),
    HIGH: "score above " + f4(cuts.high),
  };
  const desc = bands.map((b) => b + ": " + pct1(rates[b].share) + " of " + int(rates[b].rows) + " rows flooded").join("; ");
  return (
    <svg className="chart" viewBox={"0 0 " + W + " " + H} role="img" aria-label={"Observed flooded share by risk class in the evaluation sample. " + desc}>
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top - 6} y2={H - 20} className="grid" />
          <text x={x(t)} y={H - 4} textAnchor="middle" className="tick">{Math.round(t * 100)}%</text>
        </g>
      ))}
      {bands.map((b, i) => {
        const y = top + i * rowH;
        return (
          <g key={b}>
            <text x={left - 10} y={y + 17} textAnchor="end" className="lab">{b}</text>
            <text x={left - 10} y={y + 34} textAnchor="end" className="sub">{range[b]}</text>
            <rect x={left} y={y} width={Math.max(1, x(rates[b].share) - left)} height="24" style={{ fill: RISK_FILL[b] }} />
            <text x={x(rates[b].share) + 8} y={y + 17} className="val">{pct1(rates[b].share)}</text>
            <text x={x(rates[b].share) + 8} y={y + 36} className="sub">{int(rates[b].flood)} of {int(rates[b].rows)}</text>
          </g>
        );
      })}
    </svg>
  );
}
