import { int, pct1, year } from "./fmt.js";

export function DataInfo({ ev }) {
  const c = ev.counts;
  const q = ev.quality;
  return (
    <div>
      <h2 className="h2">Data Intelligence</h2>
      <p className="lead">
        Source datasets, how they enter the model, and the data-quality findings recorded in the repository. Terms, credits
        and versions of every source are in data/DATA_SOURCES.md.
      </p>
      <section className="panel" aria-labelledby="ds-h">
        <h3 id="ds-h" className="h3">Source datasets</h3>
        <div className="scroll">
          <table>
            <caption className="sr">Source datasets</caption>
            <thead><tr><th scope="col">Dataset</th><th scope="col">Provider</th><th scope="col">Role</th></tr></thead>
            <tbody>
              {ev.datasets.map((d) => (
                <tr key={d.name}><th scope="row">{d.name}</th><td>{d.org}</td><td>{d.role}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="panel" aria-labelledby="sum-h">
        <h3 id="sum-h" className="h3">Study data</h3>
        <dl className="dl">
          <dt>Training file</dt>
          <dd>{int(c.train_rows)} rows over {c.train_dates} scene dates ({year(c.date_range.train[0])} to {year(c.date_range.train[1])})</dd>
          <dt>Confirmatory block</dt>
          <dd>{int(c.block_a_rows)} rows over {c.block_a_dates} scene dates ({year(c.date_range.block_a[0])} to {year(c.date_range.block_a[1])})</dd>
          <dt>Sampling design</dt>
          <dd>Case-control: flooded and non-flooded points in fixed numbers per date, so a flooded share in a file is a design choice, not a prevalence.</dd>
          <dt>Frames</dt>
          <dd>
            Confirmatory block: {int(c.block_a_rows)} rows, {int(c.block_a_rows_mappable)} outside the GFM exclusion mask, {int(c.block_a_rows_primary)} after
            removing {int(c.block_a_rows_excluded)} rows at locations used in training.
          </dd>
        </dl>
      </section>
      <section className="panel" aria-labelledby="dq-h">
        <h3 id="dq-h" className="h3">Data-quality findings</h3>
        <ul className="list">
          <li>
            Clay content is blank on {int(q.clay_blank_rows)} training rows: {pct1(q.clay_blank_share_of_flood)} of flood rows
            against {pct1(q.clay_blank_share_of_controls)} of control rows. It is never imputed. Missingness tracks the label.
          </li>
          <li>
            {int(q.elevation_floor_rows)} training rows sit at the floor of the elevation layer, and {pct1(q.elevation_floor_flooded_share)} of
            them are flooded. They are kept and disclosed.
          </li>
          <li>GFM publishes an exclusion mask. A point inside it cannot be labelled flooded, so scoring is reported outside the mask.</li>
          <li>Land cover is one 2020 map applied to every scene date.</li>
        </ul>
        <p className="small">
          These figures are computed from the committed training file ({int(q.rows)} rows) when the evidence file is built.
        </p>
      </section>
    </div>
  );
}
