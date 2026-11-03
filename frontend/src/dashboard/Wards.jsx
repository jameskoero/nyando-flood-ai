import { AucBars } from "./charts.jsx";
import { WardMap } from "./WardMap.jsx";
import { WARD_FLAGS, f3, int, pct1, signed4 } from "./fmt.js";

export function WardSummary({ ev, ward }) {
  const it = ev.wards.items[ward];
  if (!it) {
    return (
      <section className="panel" aria-live="polite">
        <h3 className="h3">Selected ward</h3>
        <p className="small">Select a ward on the map to see its evaluation summary.</p>
      </section>
    );
  }
  return (
    <section className="panel" aria-live="polite">
      <h3 className="h3">{ward}</h3>
      <dl className="dl">
        <dt>Flood rows</dt><dd>{int(it.flood_rows)}</dd>
        <dt>Control rows</dt><dd>{int(it.control_rows)}</dd>
        <dt>Events with both classes</dt><dd>{it.events_both}</dd>
        <dt>Per-event mean AUC</dt>
        <dd>
          {it.estimable
            ? f3(it.auc.mean) + " (" + f3(it.auc.ci_low) + " to " + f3(it.auc.ci_high) + ")"
            : "Not estimable"}
        </dd>
        {it.gap && (
          <>
            <dt>Gap to the pooled result</dt>
            <dd>{signed4(it.gap.gap)} ({signed4(it.gap.ci_low)} to {signed4(it.gap.ci_high)})</dd>
          </>
        )}
      </dl>
      {it.rates && (
        <p className="small">
          Observed flooded share by risk class in this ward (evaluation sample): LOW {pct1(it.rates.LOW.share)}, MEDIUM{" "}
          {pct1(it.rates.MEDIUM.share)}, HIGH {pct1(it.rates.HIGH.share)}.
        </p>
      )}
      {it.flags.length > 0 && <p className="flag">{it.flags.map((f) => WARD_FLAGS[f] || f).join(". ")}.</p>}
      {!it.estimable && it.reason && <p className="small">Reason: {it.reason}.</p>}
    </section>
  );
}

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
