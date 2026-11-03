// Number formatting only. No research value is written in this folder; all of them come from /evidence.json.
const MINUS = "\u2212";

export const f2 = (x) => (typeof x === "number" ? x.toFixed(2) : "n/a");
export const f3 = (x) => (typeof x === "number" ? x.toFixed(3) : "n/a");
export const f4 = (x) => (typeof x === "number" ? x.toFixed(4) : "n/a");
export const signed4 = (x) =>
  typeof x === "number" ? (x < 0 ? MINUS : "+") + Math.abs(x).toFixed(4) : "n/a";
export const pct1 = (x) => (typeof x === "number" ? (100 * x).toFixed(1) + "%" : "n/a");
export const int = (x) => (typeof x === "number" ? x.toLocaleString("en-US") : "n/a");
export const year = (iso) => (typeof iso === "string" ? iso.slice(0, 4) : "n/a");
export const spansZero = (lo, hi) => lo <= 0 && hi >= 0;

export const WARD_FLAGS = {
  differs_from_pooled: "Differs from the pooled result",
  bands_not_separated: "Risk bands are not separated",
  not_estimable: "Not estimable",
};

export const UNAVAILABLE_LAYERS = [
  "Lake Victoria shoreline",
  "Nyando River network",
  "Nyando basin boundary",
  "Terrain and elevation surface",
  "Spatial flood susceptibility surface",
];
