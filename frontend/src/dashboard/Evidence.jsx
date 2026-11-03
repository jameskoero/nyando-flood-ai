import { IntervalPlot, RiskBars } from "./charts.jsx";
import { f3, f4, int, pct1, signed4, spansZero } from "./fmt.js";

const PERF_LABELS = {
  block_a_mappable: "Confirmatory block, mappable frame (headline)",
  block_a_full: "Confirmatory block, full frame",
  existing_events_mappable: "Training-era events, mappable frame",
  existing_events_full: "Training-era events, full frame",
};
const PERF_ORDER = Object.keys(PERF_LABELS);

export function Evidence({ ev }) {
  const c = ev.counts;
  const perf = ev.performance.per_event_mean_auc;
  const cmp = ev.comparison;
  const lv = ev.levels;
  const keys = PERF_ORDER.filter((k) => k in perf).concat(Object.keys(perf).filter((k) => !PERF_ORDER.includes(k)));
  const nEvents = (k) => (k.startsWith("block_a") ? c.block_a_dates : k.startsWith("existing") ? c.events_training_era : "n/a");
  const rows = [
    { label: "All events pooled", ...cmp.pooled },
    { label: "1 km spatial buffer", ...cmp.buffered },
    { label: "Out-of-time split", ...cmp.temporal },
  ].map((r) => ({ label: r.label, n: r.n_events, mean: r.mean_diff, lo: r.ci_low, hi: r.ci_high, wins: r.wins, losses: r.losses, ties: r.ties }));
  return (
    <div>
      <h2 className="h2">Model Evidence</h2>
      <p className="lead">
        The registered model is a monotonic-constrained gradient-boosting classifier. This page separates confirmatory
        results, training-era results, baseline comparisons and robustness tests, and it reports the tests the model does not pass.
      </p>

      <section className="panel" aria-labelledby="perf-h">
        <h3 id="perf-h" className="h3">Per-event mean AUC</h3>
        <div className="scroll">
          <table>
            <caption className="sr">Per-event mean AUC of the registered model</caption>
            <thead><tr><th scope="col">Frame</th><th scope="col">Events</th><th scope="col">Mean AUC</th></tr></thead>
            <tbody>
              {keys.map((k) => (
                <tr key={k}><th scope="row">{PERF_LABELS[k] || k}</th><td>{nEvents(k)}</td><td>{f3(perf[k])}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small">
          The confirmatory block was scored once, after the model and the rules were fixed. Training-era values come from
          leave-one-event-out validation with shared locations removed. The mappable frame excludes points inside the GFM
          exclusion mask.
        </p>
      </section>

      <section className="panel" aria-labelledby="cmp-h">
        <h3 id="cmp-h" className="h3">Comparison with elevation alone</h3>
        <p className="small">
          Paired per-event AUC difference, constrained gradient boosting minus a baseline that ranks points by minus their
          elevation. Intervals are two-sided 95% bootstrap intervals over events.
        </p>
        <IntervalPlot rows={rows} />
        <div className="scroll">
          <table>
            <caption className="sr">Robustness of the advantage over elevation alone</caption>
            <thead><tr><th scope="col">Test</th><th scope="col">Events</th><th scope="col">Difference</th><th scope="col">Interval</th><th scope="col">Wins</th><th scope="col">Losses</th><th scope="col">Ties</th><th scope="col">Reading</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.label}>
                  <th scope="row">{r.label}</th><td>{r.n}</td><td>{signed4(r.mean)}</td>
                  <td>{signed4(r.lo)} to {signed4(r.hi)}</td><td>{r.wins}</td><td>{r.losses}</td><td>{r.ties}</td>
                  <td>{spansZero(r.lo, r.hi) ? "Not demonstrated" : "Demonstrated"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small">
          The pooled advantage is small. The buffered and out-of-time tests are the stricter ones, and their intervals
          include zero. The model is therefore not shown to rank places better than elevation does for a future event or a
          neighbouring place.
        </p>
      </section>

      <section className="panel" aria-labelledby="risk-h">
        <h3 id="risk-h" className="h3">Risk-class evidence</h3>
        <p className="small">
          Class boundaries were fixed from training scores alone: LOW below {f4(lv.cutpoints.low)}, MEDIUM from{" "}
          {f4(lv.cutpoints.low)} to {f4(lv.cutpoints.high)}, HIGH above {f4(lv.cutpoints.high)}. Outcome of the validation:{" "}
          {lv.outcome}.
        </p>
        <RiskBars rates={lv.rates} cuts={lv.cutpoints} />
        <p className="notice">{ev.disclaimer}</p>
        <p className="small">
          Evaluation sample: {int(c.block_a_rows_primary)} primary-frame rows. HIGH minus LOW flooded share: {pct1(lv.bootstrap.diff)}{" "}
          (interval {pct1(lv.bootstrap.ci_low)} to {pct1(lv.bootstrap.ci_high)}, {int(lv.bootstrap.draws)} event-clustered resamples).
        </p>
      </section>

      <section className="panel" aria-labelledby="lim-h">
        <h3 id="lim-h" className="h3">What the evidence does not show</h3>
        <ul className="list">
          {ev.not_done.concat(ev.limits).map((t) => (
            <li key={t}>{t}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
