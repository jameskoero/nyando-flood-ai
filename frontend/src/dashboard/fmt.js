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

const lerp = (a, b, t) => Math.round(a + (b - a) * t);
// Sequential violet to blue to turquoise scale for the ward fill. It avoids red, amber and green, which belong to the risk classes.
export function aucColour(mean) {
  const t = Math.max(0, Math.min(1, (mean - 0.5) / 0.5));
  const lo = [91, 33, 182];
  const mid = [59, 130, 246];
  const hi = [20, 227, 197];
  const [a, b, u] = t < 0.5 ? [lo, mid, t * 2] : [mid, hi, (t - 0.5) * 2];
  return "rgb(" + lerp(a[0], b[0], u) + "," + lerp(a[1], b[1], u) + "," + lerp(a[2], b[2], u) + ")";
}
