import { AucBars } from "./charts.jsx";
import { MapLegend, WardMap } from "./WardMap.jsx";
import { WardSummary } from "./WardSummary.jsx";
import { f3, int } from "./fmt.js";

export function WardExplorer({ ev, ward, setWard }) {
  const W = ev.wards;
  return (
    <div>
      <h2 className="h2">Ward Explorer</h2>
      <p className="lead">
        Per-event mean AUC by ward on the primary evaluation frame ({int(W.primary_rows)} rows). AUC measures how well a ward
        ranks flooded points above non-flooded points. It is not the share of land predicted correctly. Sample sizes are shown
        because some wards rest on few events.
      </p>
      <div className="split">
        <section className="panel" aria-labelledby="auc-h">
          <h3 id="auc-h" className="h3">Mean AUC by ward, with interval</h3>
          <AucBars order={W.order} items={W.items} selected={ward} onSelect={setWard} />
        </section>
        <div>
          <section className="panel">
            <h3 className="h3">Location</h3>
            <WardMap geo={ev.geo} order={W.order} items={W.items} selected={ward} onSelect={setWard} labels={false} />
            <MapLegend />
          </section>
          <WardSummary ev={ev} ward={ward} />
        </div>
      </div>
      <section className="panel" aria-labelledby="tab-h">
        <h3 id="tab-h" className="h3">Comparison table</h3>
        <div className="scroll">
          <table>
            <caption className="sr">Ward comparison</caption>
            <thead>
              <tr><th scope="col">Ward</th><th scope="col">Events</th><th scope="col">Mean AUC</th><th scope="col">Interval</th><th scope="col">Flood rows</th><th scope="col">Control rows</th></tr>
            </thead>
            <tbody>
              {W.order.map((n) => {
                const it = W.items[n];
                return (
                  <tr key={n}>
                    <th scope="row">{n}</th>
                    <td>{it.estimable ? it.auc.events : it.events_both}</td>
                    <td>{it.estimable ? f3(it.auc.mean) : "Not estimable"}</td>
                    <td>{it.estimable ? f3(it.auc.ci_low) + " to " + f3(it.auc.ci_high) : "n/a"}</td>
                    <td>{int(it.flood_rows)}</td>
                    <td>{int(it.control_rows)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="small">
          A ward with too few flood rows or events is reported as not estimable. It is never shown as zero. Wards with few
          events have wide intervals and should be read with caution.
        </p>
      </section>
    </div>
  );
}
