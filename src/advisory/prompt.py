"""Phase E prompt (docs/PHASE_E_ADVISORY.md): the system text is fixed and versioned by its hash; the user content is the categorical context as sorted JSON, so it holds no number and is the cache key."""
import hashlib
import json

from src.advisory.context import LABELS

SYSTEM = (
    "You write short plain-language advisories about flood susceptibility for community readers in the Nyando basin, Kenya. "
    "You receive JSON with a ward name (or null), a score band (or null) and up to three drivers. "
    "Return only a JSON object with exactly these keys: summary (one sentence), drivers (a list with one short sentence per driver, in the order given) and meaning (one or two sentences). "
    "Rules: never write a digit or a percent sign; never state or imply a probability, a chance, a likelihood, a timing, a lead time or a forecast; "
    "never tell people to evacuate or to take any action; never use words such as extreme, severe, critical or dangerous; never name or describe a risk level; "
    "describe only what the drivers say, as higher or lower than a typical training location; say that the result ranks locations and is not a warning. "
    "Do not add information that is not in the JSON."
)
PROMPT_VERSION = hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()[:12]


def _named(driver):
    """Feature keys can hold a digit (rainfall_3day), so the provider is shown the plain label instead."""
    return dict({k: v for k, v in driver.items() if k != "feature"}, feature=LABELS[driver["feature"]])


def user_content(context):
    return json.dumps({"ward": context["ward"], "score_band": context["level"], "drivers": [_named(d) for d in context["drivers"]]}, sort_keys=True)


def cache_key(context):
    return hashlib.sha256((PROMPT_VERSION + user_content(context)).encode("utf-8")).hexdigest()
