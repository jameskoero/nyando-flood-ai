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
