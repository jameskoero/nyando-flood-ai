import { aucColour, f3 } from "./fmt.js";

const K = 0.6;

// Real ward polygons from data/external/nyando_wards.geojson, projected in scripts/build_evidence.py.
// Nothing here is drawn by hand: layers with no file in the repository are listed as unavailable.
export function WardMap({ geo, order, items, selected, onSelect, labels }) {
  const w = geo.viewBox[0] * K;
  const h = geo.viewBox[1] * K;
  const barUnits = (5 / geo.km_per_unit) * K;
  return (
    <svg className="map" viewBox={"0 0 " + w + " " + (h + 54)} role="group" aria-label="Map of the five study wards">
      <rect x="0" y="0" width={w} height={h + 54} className="mapbg" />
      <g transform={"scale(" + K + ")"}>
        {order.map((name) => {
          const g = geo.wards[name];
          const on = selected === name;
          const it = items[name];
          const label = name + (it.estimable ? ", mean AUC " + f3(it.auc.mean) : ", not estimable");
          return (
            <path
              key={name}
              d={g.d}
              role="button"
              tabIndex={0}
              aria-pressed={on}
              aria-label={label}
              className={"ward" + (on ? " on" : "")}
              style={{ fill: it.estimable ? aucColour(it.auc.mean) : "url(#nd-hatch)" }}
              onClick={() => onSelect(name)}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(name)}
            >
              <title>{label}</title>
            </path>
          );
        })}
      </g>
      {labels &&
        order.map((name) => {
          const g = geo.wards[name];
          const cx = g.cx * K;
          const cy = g.cy * K;
          const pw = name.length * 7.3 + 14;
          return (
            <g key={name} pointerEvents="none">
              <rect x={cx - pw / 2} y={cy - 10} width={pw} height="16" rx="8" className="pill" />
              <text x={cx} y={cy + 2} textAnchor="middle" className="wardlabel">
                {name}
              </text>
            </g>
          );
        })}
      <g transform={"translate(" + (w - 22) + ",26)"} aria-label="North arrow" role="img">
        <rect x="-11" y="-21" width="22" height="46" rx="6" className="pill" />
        <path d="M0,-16 L6,5 L0,0 L-6,5 Z" className="north" />
        <text x="0" y="19" textAnchor="middle" className="tick">N</text>
      </g>
      <g transform={"translate(12," + (h + 18) + ")"} role="img" aria-label="Scale bar, 5 kilometres">
        <line x1="0" x2={barUnits} y1="0" y2="0" className="scalebar" />
        <line x1="0" x2="0" y1="-4" y2="4" className="scalebar" />
        <line x1={barUnits} x2={barUnits} y1="-4" y2="4" className="scalebar" />
        <text x={barUnits + 6} y="4" className="tick">5 km</text>
      </g>
      <text x={w - 8} y={h + 42} textAnchor="end" className="tick">
        {geo.bbox.lat[0]} to {geo.bbox.lat[1]} N, {geo.bbox.lon[0]} to {geo.bbox.lon[1]} E
      </text>
    </svg>
  );
}

export function MapLegend() {
  return (
    <div className="legend">
      <div className="legend-bar" aria-hidden="true" />
      <div className="legend-ends" aria-hidden="true">
        <span>Chance-level ranking</span>
        <span>Strong ranking</span>
      </div>
      <p className="small">
        Ward fill: per-event mean AUC from the evaluation. Hatched: not estimable. A bright ring marks the selected ward. Colour shows
        evaluation ranking quality, not flood risk.
      </p>
    </div>
  );
}
