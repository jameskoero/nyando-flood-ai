import { WardMap } from "./WardMap.jsx";
import { WardSummary } from "./Wards.jsx";
import { UNAVAILABLE_LAYERS, f3, int, signed4 } from "./fmt.js";
import { useState } from "react";

function Kpi({ value, label, note }) {
  return (
    <div className="kpi">
      <div className="kpi-v">{value}</div>
      <div className="kpi-l">{label}</div>
      <div className="kpi-n">{note}</div>
    </div>
  );
}

export function Overview({ ev, ward, setWard }) {
  const [labels, setLabels] = useState(true);
  const c = ev.counts;
  const perf = ev.performance.per_event_mean_auc.block_a_mappable;
  const pooled = ev.comparison.pooled;
  return (
    <div>
      <h2 className="h2">Overview</h2>
      <p className="lead">
        Research and validation indicators for a flood susceptibility ranking of five wards in the Nyando basin.
        They describe the evaluation of the model, not live flood conditions. The output is a ranking score, not a flood
        probability and not an official warning.
      </p>
      <div className="kpis">
        <Kpi value={int(c.train_rows)} label="Training observations" note={"Case-control sample, " + c.train_dates + " scene dates"} />
        <Kpi value={int(c.flood)} label="Flood cases" note="Labelled by Copernicus GFM" />
        <Kpi value={int(c.controls)} label="Non-flood controls" note="Sampled in the same wards" />
        <Kpi value={int(c.events_pooled)} label="Events evaluated" note={c.events_training_era + " training-era and " + c.block_a_dates + " confirmatory"} />
        <Kpi value={f3(perf)} label="Confirmatory per-event mean AUC" note={c.block_a_dates + " sealed events, scored once"} />
        <Kpi
          value={signed4(pooled.mean_diff)}
          label="Gain over elevation alone"
          note={"95% interval " + signed4(pooled.ci_low) + " to " + signed4(pooled.ci_high)}
        />
      </div>
      <div className="split">
        <section className="panel" aria-labelledby="map-h">
          <h3 id="map-h" className="h3">Study wards</h3>
          <WardMap geo={ev.geo} order={ev.wards.order} items={ev.wards.items} selected={ward} onSelect={setWard} labels={labels} />
          <p className="small">
            Boundaries: data/external/nyando_wards.geojson. Select a ward on the map. The map shows no model scores, no
            flood extent and no live readings.
          </p>
          <fieldset className="layers">
            <legend>Layers</legend>
            <label>
              <input type="checkbox" checked={labels} onChange={(e) => setLabels(e.target.checked)} /> Ward names
            </label>
            {UNAVAILABLE_LAYERS.map((l) => (
              <label key={l} className="off">
                <input type="checkbox" disabled /> {l} <span>(data not in the repository)</span>
              </label>
            ))}
          </fieldset>
        </section>
        <WardSummary ev={ev} ward={ward} />
      </div>
    </div>
  );
}
