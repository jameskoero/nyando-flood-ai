"""Phase E output guard (docs/PHASE_E_ADVISORY.md Sections 4 and 11): text written by a provider is accepted only if it has the agreed shape and none of these rules fires. A rejected text is replaced by the deterministic template. With provider=True only summary and meaning are read (a drivers key is ignored, because the driver sentences always come from the template) and a word that states a direction is refused, because the provider is told no direction."""
import re

KEYS = ("summary", "drivers", "meaning")
PROVIDER_KEYS = ("summary", "meaning")
MAX_CHARS = {"summary": 240, "driver": 160, "meaning": 320}
MAX_DRIVERS = 3
RULES = {
    "digit": r"\d",
    "percent": r"(?i:%|\bper ?cent\b)",
    "probability_word": r"(?i:\b(probabilit\w*|probable|chance\w*|odds|likel\w*|unlikel\w*)\b)",
    "action_instruction": r"(?i:\b(evacuat\w*|flee\w*|leave (the area|your home|your homes|immediately|now)|move to higher ground|shelter\w*)\b)",
    "certainty": r"(?i:\b(will (flood|be flooded|overflow|happen)|going to flood|certain\w*|guarantee\w*|definite\w*|inevitabl\w*|sure to)\b)",
    "timing": r"(?i:\b(tomorrow|tonight|today|this (week|weekend|evening|morning|month)|next (hour|hours|day|days|week|weeks)|within|in the (next|coming)|soon|imminent\w*|hours?|minutes?|lead time)\b)",
    "severity": r"(?i:\b(extreme\w*|severe\w*|critical\w*|catastroph\w*|dangerous\w*|life[- ]threatening|emergenc\w*|alert\w*|alarm\w*|disaster\w*)\b)",
    "level_word": r"\b(LOW|MEDIUM|HIGH)\b|(?i:\b(high|low|medium|moderate)[- ]risk\b|\brisk level\b)",
    "markup": r"[<>`]|(?i:https?://|www\.)|\]\(",
    "injection": r"(?i:\b(ignore (all |any )?(previous|prior)|system prompt|as an ai)\b)",
}
PROVIDER_RULES = {
    "direction_word": r"(?i:\b(above|below|higher|lower|raises?|lowers?|increases?|decreases?|boosts?|reduces?)\b)",
}
COMPILED = {name: re.compile(pattern) for name, pattern in RULES.items()}
PROVIDER_COMPILED = {name: re.compile(pattern) for name, pattern in PROVIDER_RULES.items()}


def check(advisory, provider=False):
    """The reasons a text is rejected; an empty list means it is accepted."""
    keys = PROVIDER_KEYS if provider else KEYS
    got = (set(advisory) - ({"drivers"} if provider else set())) if isinstance(advisory, dict) else None
    if got != set(keys):
        return ["structure: the keys must be exactly %s" % ", ".join(keys)]
    s, m = advisory["summary"], advisory["meaning"]
    dr = [] if provider else advisory["drivers"]
    if not isinstance(s, str) or not isinstance(m, str) or not isinstance(dr, list) or not all(isinstance(x, str) for x in dr):
        return ["structure: summary and meaning must be text%s" % ("" if provider else " and drivers a list of text")]
    reasons = []
    if len(dr) > MAX_DRIVERS:
        reasons.append("length: more than %d drivers" % MAX_DRIVERS)
    if len(s) > MAX_CHARS["summary"] or len(m) > MAX_CHARS["meaning"] or any(len(x) > MAX_CHARS["driver"] for x in dr):
        reasons.append("length: a part is too long")
    if not s.strip() or not m.strip():
        reasons.append("length: summary and meaning must not be empty")
    rules = dict(COMPILED, **PROVIDER_COMPILED) if provider else COMPILED
    for name, rx in rules.items():
        if any(rx.search(t) for t in [s, m] + dr):
            reasons.append(name)
    return reasons
