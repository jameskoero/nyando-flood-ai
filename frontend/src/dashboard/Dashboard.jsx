import { lazy, Suspense, useEffect, useState } from "react";
import "./dashboard.css";
import { useEvidence, useService, verify } from "./useLive.js";
import { Overview } from "./Overview.jsx";
import { Hero } from "./Hero.jsx";

// The existing, tested scoring page loads only when its tab is opened, so the first screen stays light.
const ScoreTool = lazy(() => import("../App.jsx"));
// Only the Overview is needed at load. The other sections are fetched when their tab is opened, which keeps the first load small.
const WardExplorer = lazy(() => import("./Wards.jsx").then((m) => ({ default: m.WardExplorer })));
const Evidence = lazy(() => import("./Evidence.jsx").then((m) => ({ default: m.Evidence })));
const DataInfo = lazy(() => import("./DataInfo.jsx").then((m) => ({ default: m.DataInfo })));
const About = lazy(() => import("./About.jsx").then((m) => ({ default: m.About })));

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

function GradientDefs() {
  return (
    <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden="true" focusable="false">
      <defs>
        <linearGradient id="nd-bar" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#22d3ee" />
          <stop offset="1" stopColor="#8b5cf6" />
        </linearGradient>
        <linearGradient id="nd-ward" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#12408a" />
          <stop offset="1" stopColor="#0b2a63" />
        </linearGradient>
        <linearGradient id="nd-ward-on" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#14e3c5" stopOpacity="0.85" />
          <stop offset="1" stopColor="#0e7490" stopOpacity="0.9" />
        </linearGradient>
        <pattern id="nd-hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="8" height="8" fill="#1b2f66" />
          <line x1="0" y1="0" x2="0" y2="8" stroke="#a78bfa" strokeWidth="3" />
        </pattern>
        <radialGradient id="nd-mapbg" cx="0.5" cy="0.45" r="0.8">
          <stop offset="0" stopColor="#0d2f6b" />
          <stop offset="1" stopColor="#050b1e" />
        </radialGradient>
      </defs>
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
      <li>{reg && reg.loaded ? "Model " + reg.arm + ": " + reg.file.replace(".onnx", "") : "Model: not confirmed yet"}</li>
      <li>
        {check
          ? check.shaMatches
            ? "Checksum matches the repository manifest"
            : "Checksum does not match the manifest"
          : "Checksum: not checked yet"}
      </li>
      <li>{svc.at ? "Verified at " + svc.at.toLocaleTimeString() + " on this device" : "Last verified: not yet"}</li>
    </ul>
  );
}

export default function Dashboard() {
  const [tab, setTab] = useState("overview");
  const [ward, setWard] = useState(null);
  // Decorative motion is not in the page at load. It is added on the first touch, scroll or key press, never on a timer.
  const [alive, setAlive] = useState(false);
  useEffect(() => {
    const wake = () => setAlive(true);
    const events = ["pointerdown", "keydown", "scroll"];
    events.forEach((e) => window.addEventListener(e, wake, { once: true, passive: true }));
    return () => events.forEach((e) => window.removeEventListener(e, wake));
  }, []);
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
    else {
      inner = (
        <Suspense fallback={<p className="state" role="status">Loading this section.</p>}>
          {tab === "wards" && <WardExplorer ev={ev.data} ward={ward} setWard={setWard} />}
          {tab === "evidence" && <Evidence ev={ev.data} />}
          {tab === "data" && <DataInfo ev={ev.data} />}
          {tab === "about" && <About ev={ev.data} check={check} />}
        </Suspense>
      );
    }
    body = (
      <main className="nd-main" id="content">
        {tab === "overview" && <Hero alive={alive} />}
        {inner}
      </main>
    );
  }

  return (
    <div className={"nd" + (alive ? " alive" : "")}>
      <GradientDefs />
      <header className="nd-head">
        <div className="brand">
          <Logo />
          <div>
            <h1 className="title">Flood-susceptibility ranking for the Nyando Basin, Kenya</h1>
            <p className="sub">Nyando Flood AI V2</p>
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
