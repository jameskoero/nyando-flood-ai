import { f3 } from "./fmt.js";

// Real ward polygons from data/external/nyando_wards.geojson, projected in scripts/build_evidence.py.
// Nothing here is drawn by hand: layers with no file in the repository are listed as unavailable.
export function WardMap({ geo, order, items, selected, onSelect, labels }) {
  const [w, h] = geo.viewBox;
  const barUnits = 5 / geo.km_per_unit;
  return (
    <svg className="map" viewBox={"0 0 " + w + " " + (h + 40)} role="group" aria-label="Map of the five study wards">
      <rect x="0" y="0" width={w} height={h + 40} className="mapbg" />
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
            onClick={() => onSelect(name)}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(name)}
          >
            <title>{label}</title>
          </path>
        );
      })}
      {labels &&
        order.map((name) => (
          <text key={name} x={geo.wards[name].cx} y={geo.wards[name].cy} textAnchor="middle" className="wardlabel">
            {name}
          </text>
        ))}
      <g transform={"translate(" + (w - 34) + ",34)"} aria-label="North arrow" role="img">
        <path d="M0,-22 L8,6 L0,0 L-8,6 Z" className="north" />
        <text x="0" y="22" textAnchor="middle" className="tick">N</text>
      </g>
      <g transform={"translate(16," + (h + 20) + ")"} role="img" aria-label="Scale bar, 5 kilometres">
        <line x1="0" x2={barUnits} y1="0" y2="0" className="scalebar" />
        <line x1="0" x2="0" y1="-5" y2="5" className="scalebar" />
        <line x1={barUnits} x2={barUnits} y1="-5" y2="5" className="scalebar" />
        <text x={barUnits + 8} y="4" className="tick">5 km</text>
      </g>
      <text x={w - 12} y={h + 24} textAnchor="end" className="tick">
        {geo.bbox.lat[0]} to {geo.bbox.lat[1]} N, {geo.bbox.lon[0]} to {geo.bbox.lon[1]} E
      </text>
    </svg>
  );
}
