const CHECKS = [
  "Leakage-controlled validation: events are the unit of independence, never random rows.",
  "Shared-location removal: the model never scores a place it saw in training.",
  "One-time confirmatory scoring: the sealed block was scored once, in a logged run.",
  "Hash-verified registration: the served model is identified by its SHA-256, not its file name.",
  "ONNX parity: the served artifact reproduces the reference model within a declared tolerance.",
  "Provenance and missing-data reporting: sources, blank values and known data limits are published.",
  "Written limitations: negative and inconclusive results stay in the public record.",
];

export function About({ ev, check }) {
  return (
    <div>
      <h2 className="h2">Methodology &amp; About</h2>
      <p className="lead">
        Nyando Flood AI V2 is a research product. It ranks locations in five Nyando wards by relative flood susceptibility. It
        does not forecast floods, estimate probabilities or issue warnings.
      </p>

      <section className="panel" aria-labelledby="int-h">
        <h3 id="int-h" className="h3">Scientific integrity</h3>
        <ul className="list">{CHECKS.map((t) => <li key={t}>{t}</li>)}</ul>
        <h4 className="h4">Live service check</h4>
        {check ? (
          <ul className="list">
            <li>Registered model loaded in the service: {check.loaded ? "yes" : "no"}.</li>
            <li>Service model hash matches the repository manifest: {check.shaMatches && check.metricsMatch ? "yes" : "no"}.</li>
            <li>Service evidence matches the repository record: {check.gainMatches ? "yes" : "no"}.</li>
            {check.legacyFile && (
              <li>
                A legacy v1 artifact ({check.legacyFile}) is still present in the service. Its endpoint answers 410 Gone by
                default and its removal is tracked as register row D51.
              </li>
            )}
          </ul>
        ) : (
          <p className="small">The live service has not answered, so no live verification is shown.</p>
        )}
      </section>

      <section className="panel" aria-labelledby="feat-h">
        <h3 id="feat-h" className="h3">Model inputs and constraints</h3>
        <p className="small">
          The registered model takes six inputs. Four carry a monotonic constraint that fixes the direction of the
          relationship. A constraint is a modelling choice that follows physical expectation. It does not prove causation.
        </p>
        <div className="scroll">
          <table>
            <caption className="sr">Model inputs</caption>
            <thead><tr><th scope="col">Input</th><th scope="col">Effect on the score</th></tr></thead>
            <tbody>
              {ev.features.map((f) => (
                <tr key={f.key}>
                  <th scope="row">{f.label}</th>
                  <td>
                    {f.direction === "increases" && "\u2191 Score increases as the input increases"}
                    {f.direction === "decreases" && "\u2193 Score decreases as the input increases"}
                    {f.direction === "unconstrained" && "No constraint"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel" aria-labelledby="adv-h">
        <h3 id="adv-h" className="h3">Advisory text</h3>
        <p className="small">
          The Score a location tab uses the advisory service. The ranking score and the risk class come from the registered
          model. Sentences about drivers come from a fixed template. A language model writes only a short summary and a
          statement of meaning, and its answer is checked before display. The score never depends on a language model. The
          advisory is explanatory output, not an independent hydrological assessment.
        </p>
      </section>

      <section className="panel profile" aria-labelledby="dev-h">
        <img src="/koero.jpg" width="96" height="96" alt="James Koero" loading="lazy" onError={(e) => { e.currentTarget.style.display = "none"; }} />
        <div>
          <h3 id="dev-h" className="h3">James Koero</h3>
          <p className="small">Machine Learning Engineer, Kisumu, Kenya</p>
          <p className="small">
            <a href="https://github.com/jameskoero/nyando-flood-ai" rel="noopener noreferrer">github.com/jameskoero/nyando-flood-ai</a>
          </p>
          <p className="small">Designed and developed by James Koero, Machine Learning Engineer.</p>
        </div>
      </section>
    </div>
  );
}
