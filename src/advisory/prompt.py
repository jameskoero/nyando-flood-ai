"""Phase E prompt (docs/PHASE_E_ADVISORY.md Sections 8 and 11): the system text is fixed and versioned by its hash. The user content is the ward, the band and the names of the factors only, as sorted JSON: it holds no number and no direction, so a provider cannot state a direction it was never given, and it is the cache key."""
import hashlib
import json

from src.advisory.context import LABELS

SYSTEM = (
    "You write short plain-language notes about flood susceptibility for community readers in the Nyando basin, Kenya. "
    "You receive JSON with a ward name (or null), a score band (or null) and a list of factors that were used. "
    "Return only a JSON object with exactly these keys: summary (one sentence) and meaning (one or two sentences). "
    "Rules: never write a digit or a percent sign; never state or imply a probability, a chance, a likelihood, a timing, a lead time or a forecast; "
    "never tell people to evacuate or to take any action; never use words such as extreme, severe, critical or dangerous; never name or describe a risk level; "
    "you are not told whether any factor is higher or lower than usual, so never say that a factor raises, lowers, increases, decreases or is above or below anything; "
    "you may name the factors in general terms; say that the result ranks locations and is not a warning. "
    "Do not add information that is not in the JSON."
)
PROMPT_VERSION = hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()[:12]


def user_content(context):
    return json.dumps({"ward": context["ward"], "score_band": context["level"], "factors": [LABELS[d["feature"]] for d in context["drivers"]]}, sort_keys=True)


def cache_key(context):
    return hashlib.sha256((PROMPT_VERSION + user_content(context)).encode("utf-8")).hexdigest()
