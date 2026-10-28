"""Phase E advisory service (docs/PHASE_E_ADVISORY.md Section 8; register row D49): the score, the level, the three drivers and the advisory text for one location. The level is used only if the live model's SHA-256 appears in docs/PHASE_E_RESULTS.json, the file that validated the cutpoints; the ward is used only if it is one of the five in data/external/nyando_wards.geojson, so free text never reaches a provider."""
import json
from pathlib import Path

from src.advisory import rules
from src.advisory.attribution import substitution_attribution
from src.advisory.context import to_context
from src.advisory.orchestrator import Advisor
from src.advisory.providers import providers_from_env
from src.advisory.reference import load as load_reference

SCORE_MEANING = "ranking score from a case-control model; not a flood probability"
NOTICE = "Generated from a ranking score; not an official warning."
LEVEL_BASIS = "third of the reference score distribution, validated in docs/PHASE_E_RESULTS.json"


class AdvisoryService:
    def __init__(self, registered, root, environ, advisor=None):
        root = Path(root)
        self.registered = registered
        self.reference = load_reference(root / "docs" / "PHASE_E_REFERENCE.json")
        raw = (root / "docs" / "PHASE_E_RESULTS.json").read_text(encoding="utf-8")
        cuts = json.loads(raw)["cutpoints"]
        self.cuts = (float(cuts["low"]), float(cuts["high"]))
        self.levels_bound = bool(registered.sha256) and registered.sha256 in raw
        geo = json.loads((root / "data" / "external" / "nyando_wards.geojson").read_text(encoding="utf-8"))
        self.wards = frozenset(f["properties"]["ward"] for f in geo["features"])
        self.advisor = advisor if advisor is not None else Advisor(providers_from_env(environ))

    def respond(self, values, warnings=()):
        score = self.registered.score(values)
        level = rules.level_of(score, self.cuts) if self.levels_bound else None
        ward = values.get("ward") if values.get("ward") in self.wards else None
        context = to_context(ward, level, substitution_attribution(self.registered.score, values, self.reference))
        result = self.advisor.advise(context)
        return {
            "score": round(score, 4),
            "score_meaning": SCORE_MEANING,
            "level": level,
            "level_basis": LEVEL_BASIS if level else None,
            "ward": ward,
            "advisory": result["advisory"],
            "notice": NOTICE,
            "advisory_source": result["source"],
            "advisory_model": result["model"],
            "cached": result["cached"],
            "prompt_version": result["prompt_version"],
            "attempts": result["attempts"],
            "model": self.registered.describe(),
            "claim_limit": self.registered.meta.get("claim_limit"),
            "warnings": list(warnings),
        }
