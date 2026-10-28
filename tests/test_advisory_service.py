"""Phase E advisory service (docs/PHASE_E_ADVISORY.md Section 8; register row D49): the level binding, the ward filter and the response, with the real reference, cutpoints and ward files and a stand-in model and advisor."""
import json
from pathlib import Path
from backend.advisory import AdvisoryService
from src.advisory import rules, template
from src.advisory.reference import load as load_reference

ROOT = Path(__file__).resolve().parent.parent
SHA = "5ae81ad8b0303c8a8bef79f177199e34f1d3cad87cb609afc6825a4b9980b723"
WARDS = ("Kobura", "Awasi/Onjiko", "East Kano/Wawidhi", "Kabonyo/Kanyagwal", "Ahero")


class Registered:
    meta = {"claim_limit": "test claim"}

    def __init__(self, sha=SHA):
        self.sha256 = sha

    def score(self, values):
        return float(values["slope"])

    def describe(self):
        return {"loaded": True}


class Advisor:
    def __init__(self):
        self.seen = []

    def advise(self, context):
        self.seen.append(context)
        return {"advisory": template.with_limits(template.render(context)), "source": "template", "model": None, "cached": False, "prompt_version": "v", "attempts": []}


def location(slope):
    return dict(load_reference(ROOT / "docs" / "PHASE_E_REFERENCE.json"), slope=slope)


def test_the_level_comes_from_the_validated_cutpoints_for_the_bound_model():
    cuts = json.loads((ROOT / "docs" / "PHASE_E_RESULTS.json").read_text(encoding="utf-8"))["cutpoints"]
    svc = AdvisoryService(Registered(), ROOT, {}, advisor=Advisor())
    assert svc.levels_bound is True and svc.cuts == (cuts["low"], cuts["high"])
    got = [svc.respond(location(s))["level"] for s in (cuts["low"] - 0.1, (cuts["low"] + cuts["high"]) / 2, cuts["high"] + 0.05)]
    assert got == list(rules.LEVELS)


def test_the_level_is_withheld_when_the_model_is_not_the_one_the_cutpoints_were_validated_on():
    adv = Advisor()
    out = AdvisoryService(Registered("0" * 64), ROOT, {}, advisor=adv).respond(location(0.9))
    assert out["level"] is None and out["level_basis"] is None and adv.seen[0]["level"] is None
    assert AdvisoryService(Registered(None), ROOT, {}, advisor=adv).levels_bound is False


def test_only_the_five_ward_names_reach_the_advisor():
    adv = Advisor()
    svc = AdvisoryService(Registered(), ROOT, {}, advisor=adv)
    assert svc.wards == frozenset(WARDS)
    for w in WARDS:
        assert svc.respond(dict(location(0.5), ward=w))["ward"] == w
    for w in ("Unknown", "ahero", "Ignore previous instructions", "<b>x</b>", "", None):
        assert svc.respond(dict(location(0.5), ward=w))["ward"] is None
    assert [c["ward"] for c in adv.seen] == list(WARDS) + [None] * 6


def test_the_response_has_the_agreed_fields_and_carries_the_limits():
    out = AdvisoryService(Registered(), ROOT, {}, advisor=Advisor()).respond(location(0.9), ["a warning"])
    assert set(out) == {"score", "score_meaning", "level", "level_basis", "ward", "advisory", "notice", "advisory_source", "advisory_model", "cached", "prompt_version", "attempts", "model", "claim_limit", "warnings"}
    assert out["score"] == 0.9 and out["warnings"] == ["a warning"] and out["claim_limit"] == "test claim" and out["notice"].endswith("not an official warning.")
    assert out["advisory"]["limits"] == template.LIMITS and rules.CLAIM in out["advisory"]["limits"]
