"""Phase E deterministic advisory (docs/PHASE_E_ADVISORY.md): the fallback when no provider answers, written from the categorical context only. The fixed limits text is added by code to every advisory, whoever wrote the rest."""
from src.advisory.context import LABELS

LIMITS = "This is a ranking score, not a flood probability: it comes from a model trained on a case-control sample, gives no lead time and is not an official warning. Follow county and national disaster authorities for guidance."
BAND = {"LOW": "in the lowest third of the reference scores", "MEDIUM": "in the middle third of the reference scores", "HIGH": "in the highest third of the reference scores"}
MEANING = "In the held-out validation, locations in higher bands were flooded more often than locations in lower bands. This ranks locations; it does not forecast an event."
POSITION = {"above": "above", "below": "below", "same": "about the same as"}


def driver_sentence(d):
    label = LABELS[d["feature"]]
    if d["feature"] == "land_cover":
        if d.get("land_cover") is None:
            return "The land cover is not available for this location, which %s the score compared with a typical location." % d["effect"]
        return "The land cover is %s, which %s the score compared with the most common class in the training data." % (d["land_cover"], d["effect"])
    if d["position"] == "missing":
        return "The %s is not available for this location, which %s the score compared with a typical location." % (label, d["effect"])
    return "The %s here is %s the typical training location, which %s the score." % (label, POSITION[d["position"]], d["effect"])


def render(context):
    band = "this location's score is %s." % BAND[context["level"]] if context["level"] else "this location's score is shown without a level."
    summary = ("For %s: %s" % (context["ward"], band)) if context["ward"] else (band[0].upper() + band[1:])
    return {"summary": summary, "drivers": [driver_sentence(d) for d in context["drivers"]], "meaning": MEANING}


def with_limits(text):
    return dict(text, limits=LIMITS)
