import { lazy, Suspense, useState } from "react";
import "./dashboard.css";
import { useEvidence, useService, verify } from "./useLive.js";
import { Overview } from "./Overview.jsx";
import { WardExplorer } from "./Wards.jsx";
import { Evidence } from "./Evidence.jsx";
import { DataInfo } from "./DataInfo.jsx";
import { About } from "./About.jsx";

// The existing, tested scoring page loads only when its tab is opened, so the first screen stays light.
const ScoreTool = lazy(() => import("../App.jsx"));

const TABS = [
  ["overview", "Overview"],
  ["wards", "Ward Explorer"],
  ["evidence", "Model Evidence"],
  ["data", "Data Intelligence"],
  ["about", "Methodology & About"],
  ["score", "Score a location"],
];

function Logo() {
  return (
    <svg className="logo" viewBox="0 0 40 40" width="40" height="40" aria-hidden="true" focusable="false">
      <rect width="40" height="40" rx="6" className="logo-bg" />
      <path d="M5 16c5-6 8 6 15 0s10 6 15 0" className="logo-wave" />
      <path d="M5 25c5-6 8 6 15 0s10 6 15 0" className="logo-wave two" />
    </svg>
  );
}

function StatusBar({ svc, check }) {
  const reg = svc.health && svc.health.registered_model;
  let service = "Checking the service";
  if (svc.waking) service = "Waking the service (free tier, up to a minute)";
  if (svc.status === "ok") service = "Service reachable";
  if (svc.status === "unavailable") service = "Service unavailable";
  return (
    <ul className="status" aria-label="Service status">
      <li><span role="status">{service}</span></li>
      {svc.status === "ok" && reg && reg.loaded && (
        <li>Model {reg.arm}: {reg.file.replace(".onnx", "")}</li>
      )}
      {check && <li>{check.shaMatches ? "Checksum matches the repository manifest" : "Checksum does not match the manifest"}</li>}
      {svc.at && <li>Verified at {svc.at.toLocaleTimeString()} on this device</li>}
    </ul>
  );
}

export default function Dashboard() {
  const [tab, setTab] = useState("overview");
  const [ward, setWard] = useState(null);
  const ev = useEvidence();
  const svc = useService();
  const check = verify(svc, ev.data);

  let body;
  if (tab === "score") {
    body = (
      <Suspense fallback={<p className="state" role="status">Loading the scoring page.</p>}>
        <ScoreTool />
      </Suspense>
    );
  } else {
    let inner;
    if (ev.status === "loading") inner = <p className="state" role="status">Loading the research evidence.</p>;
    else if (ev.status === "error")
      inner = (
        <div className="state" role="alert">
          <p>The research evidence file could not be loaded, so no figures are shown.</p>
          <button type="button" className="btn" onClick={ev.retry}>Try again</button>
        </div>
      );
    else if (tab === "overview") inner = <Overview ev={ev.data} ward={ward} setWard={setWard} />;
    else if (tab === "wards") inner = <WardExplorer ev={ev.data} ward={ward} setWard={setWard} />;
    else if (tab === "evidence") inner = <Evidence ev={ev.data} />;
    else if (tab === "data") inner = <DataInfo ev={ev.data} />;
    else inner = <About ev={ev.data} check={check} />;
    body = <main className="nd-main" id="content">{inner}</main>;
  }

  return (
    <div className="nd">
      <header className="nd-head">
        <div className="brand">
          <Logo />
          <div>
            <h1 className="title">Nyando Flood AI V2</h1>
            <p className="sub">Flood susceptibility ranking, Nyando Basin, Kisumu County, Kenya</p>
            <p className="tag">Rank. Verify. Prepare.</p>
          </div>
        </div>
        <StatusBar svc={svc} check={check} />
      </header>
      <nav className="nd-nav" aria-label="Dashboard sections">
        {TABS.map(([id, label]) => (
          <button key={id} type="button" className={"tab" + (tab === id ? " on" : "")} aria-current={tab === id ? "page" : undefined} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </nav>
      {body}
      <footer className="nd-foot">
        <p>
          A ranking score, not a flood probability and not an official warning. Research code, data and the evidence behind every
          figure: github.com/jameskoero/nyando-flood-ai
        </p>
      </footer>
    </div>
  );
}
