import { useEffect, useState } from "react";

const BASE = "https://nyando-flood-v2.onrender.com";

// Research figures: one generated file, read at runtime, never typed into the code.
export function useEvidence() {
  const [state, setState] = useState({ status: "loading", data: null });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const ctl = new AbortController();
    fetch("/evidence.json", { signal: ctl.signal })
      .then((r) => {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then((data) => setState({ status: "ready", data }))
      .catch((e) => {
        if (e.name !== "AbortError") setState({ status: "error", data: null });
      });
    return () => ctl.abort();
  }, [attempt]);
  const retry = () => {
    setState({ status: "loading", data: null });
    setAttempt((n) => n + 1);
  };
  return { ...state, retry };
}

// Service status: every field comes from the live /health and /v2/metrics responses.
export function useService() {
  const [svc, setSvc] = useState({ status: "checking", waking: false, health: null, metrics: null, at: null });
  useEffect(() => {
    const ctl = new AbortController();
    const wake = setTimeout(() => setSvc((s) => (s.status === "checking" ? { ...s, waking: true } : s)), 4000);
    const stop = setTimeout(() => ctl.abort(), 120000);
    const get = (path) =>
      fetch(BASE + path, { signal: ctl.signal }).then((r) => {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
    Promise.all([get("/health"), get("/v2/metrics")])
      .then(([health, metrics]) => setSvc({ status: "ok", waking: false, health, metrics, at: new Date() }))
      .catch(() => setSvc({ status: "unavailable", waking: false, health: null, metrics: null, at: null }))
      .finally(() => {
        clearTimeout(wake);
        clearTimeout(stop);
      });
    return () => {
      ctl.abort();
      clearTimeout(wake);
      clearTimeout(stop);
    };
  }, []);
  return svc;
}

// Compare the live service with the repository evidence. Returns null until the service has answered.
export function verify(svc, ev) {
  if (!svc || svc.status !== "ok" || !ev) return null;
  const reg = svc.health && svc.health.registered_model;
  const live = svc.metrics && svc.metrics.versus_elevation_only && svc.metrics.versus_elevation_only.pooled_mappable;
  return {
    loaded: !!reg && reg.loaded === true,
    shaMatches: !!reg && reg.sha256 === ev.model.sha256,
    metricsMatch: !!svc.metrics && svc.metrics.model_sha256 === ev.model.sha256,
    gainMatches: !!live && Math.abs(live.mean_diff - ev.comparison.pooled.mean_diff) < 1e-9,
    legacyFile:
      svc.health && svc.health.model_file && reg && svc.health.model_file !== reg.file ? svc.health.model_file : null,
  };
}
