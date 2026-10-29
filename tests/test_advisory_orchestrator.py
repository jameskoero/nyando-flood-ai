"""Phase E orchestrator (docs/PHASE_E_ADVISORY.md Section 6; register row D49): provider order, the guard, the template fallback, the time limits and the cache, with fake providers and a fake clock."""
import json
from pathlib import Path
from src.advisory import orchestrator, prompt, rules, template
from src.advisory.orchestrator import Advisor, Cache
from src.advisory.providers import Provider, ProviderError, providers_from_env

ROOT = Path(__file__).resolve().parent.parent
CTX = {"ward": "Ahero", "level": "HIGH", "drivers": [{"feature": "rainfall_3day", "effect": "raises", "position": "above"}]}
GOOD = {"summary": "For Ahero: this location's score is in the highest third of the reference scores.", "drivers": ["The rainfall over the previous three days here is above the typical training location, which raises the score."], "meaning": "This ranks locations; it does not forecast an event."}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def fake(name, reply, clock=None, cost=0.0):
    calls = []

    def call(system, user, timeout_s):
        calls.append(timeout_s)
        if clock:
            clock.t += cost
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)
    p = Provider(name, name + "-model", call)
    p.calls = calls
    return p


def test_the_first_accepted_provider_answers_and_the_result_is_cached():
    a, b, clock = fake("gemini", GOOD), fake("groq", GOOD), Clock()
    adv = Advisor([a, b], now=clock)
    r = adv.advise(CTX)
    assert r["source"] == "gemini" and r["model"] == "gemini-model" and r["cached"] is False and r["attempts"] == [{"provider": "gemini", "status": "accepted"}]
    assert r["advisory"]["limits"] == template.LIMITS and r["advisory"]["summary"] == GOOD["summary"] and r["prompt_version"] == prompt.PROMPT_VERSION
    r2 = adv.advise(CTX)
    assert r2["cached"] is True and r2["source"] == "gemini" and len(a.calls) == 1 and not b.calls
    assert a.calls == [orchestrator.PER_PROVIDER_S]


def test_a_rejected_or_broken_answer_falls_to_the_next_provider_then_the_template():
    bad = dict(GOOD, meaning="There is a high chance of flooding tomorrow.")
    adv = Advisor([fake("gemini", bad), fake("groq", GOOD)], now=Clock())
    r = adv.advise(CTX)
    assert r["source"] == "groq" and r["attempts"][0]["status"] == "rejected" and "probability_word" in r["attempts"][0]["reasons"] and r["attempts"][1]["status"] == "accepted"
    for reply in ("not json", "[1, 2]", dict(GOOD, extra="x")):
        r = Advisor([fake("gemini", reply)], now=Clock()).advise(CTX)
        assert r["source"] == "template" and r["model"] is None and r["advisory"]["meaning"] == template.MEANING


def test_every_failure_ends_in_the_template_and_no_provider_message_escapes():
    secret = "SECRET-TOKEN-123"
    a, b = fake("gemini", ProviderError("timeout", secret)), fake("groq", RuntimeError(secret))
    adv = Advisor([a, b], now=Clock())
    r = adv.advise(CTX)
    assert r["source"] == "template" and [x["status"] for x in r["attempts"]] == ["timeout", "error: RuntimeError"]
    assert secret not in json.dumps(r) and r["advisory"] == template.with_limits(template.render(CTX))
    adv.advise(CTX)
    assert len(a.calls) == 2, "a template fallback must not be cached"
    assert Advisor([], now=Clock()).advise(CTX)["source"] == "template"


def test_the_whole_request_is_limited_to_twelve_seconds():
    clock = Clock()
    a, b = fake("gemini", ProviderError("timeout"), clock, cost=10.0), fake("groq", GOOD, clock)
    r = Advisor([a, b], now=clock).advise(CTX)
    assert a.calls == [8] and b.calls == [2] and r["source"] == "groq"
    clock2 = Clock()
    a2, b2 = fake("gemini", ProviderError("timeout"), clock2, cost=12.5), fake("groq", GOOD, clock2)
    r2 = Advisor([a2, b2], now=clock2).advise(CTX)
    assert not b2.calls and r2["source"] == "template" and r2["attempts"][1] == {"provider": "groq", "status": "skipped"}


def test_the_cache_expires_after_one_hour_and_is_bounded():
    clock = Clock()
    a = fake("gemini", GOOD)
    adv = Advisor([a], now=clock)
    adv.advise(CTX)
    clock.t += 3599
    assert adv.advise(CTX)["cached"] is True
    clock.t += 2
    assert adv.advise(CTX)["cached"] is False and len(a.calls) == 2
    small = Cache(max_items=2)
    for i in range(5):
        small.put("k%d" % i, i, clock.t)
    assert len(small._items) == 2 and small.get("k4", clock.t) == 4 and small.get("k0", clock.t) is None


def test_the_limits_come_from_the_protocol_and_the_environment_builds_the_providers():
    assert (orchestrator.PER_PROVIDER_S, orchestrator.TOTAL_S, orchestrator.CACHE_TTL_S) == (8, 12, 3600) and rules.PROVIDERS == ("gemini", "groq")
    env = {"GROQ_API_KEY": "kq", "GROQ_MODEL": "mq", "GEMINI_API_KEY": "kg", "GEMINI_MODEL": "mg"}
    assert [p.name for p in providers_from_env(env)] == ["gemini", "groq"]
    assert [p.name for p in providers_from_env(dict(env, GEMINI_MODEL=" "))] == ["groq"] and providers_from_env({}) == []
    assert "kg" not in repr(providers_from_env(env)) and "kq" not in repr(providers_from_env(env))


def test_the_advisory_document_names_the_settings_and_the_cache_rule():
    doc = (ROOT / "docs" / "PHASE_E_ADVISORY.md").read_text(encoding="utf-8")
    for must in ("GEMINI_API_KEY", "GEMINI_MODEL", "GROQ_API_KEY", "GROQ_MODEL", "is not cached", "8 s", "12 s", "one hour"):
        assert must in doc, must


CTX_FLIPPED = {"ward": "Ahero", "level": "HIGH", "drivers": [{"feature": "rainfall_3day", "effect": "lowers", "position": "below"}]}


def test_the_drivers_always_come_from_the_template_never_from_the_provider():
    wrong = dict(GOOD, drivers=["The rainfall here is below the typical location, which lowers the score."])
    r = Advisor([fake("gemini", wrong)], now=Clock()).advise(CTX)
    assert r["source"] == "gemini" and r["advisory"]["drivers"] == template.render(CTX)["drivers"] and r["advisory"]["summary"] == GOOD["summary"]


def test_a_cached_answer_keeps_its_prose_but_gets_the_drivers_of_the_new_request():
    a = fake("gemini", GOOD)
    adv = Advisor([a], now=Clock())
    first, second = adv.advise(CTX), adv.advise(CTX_FLIPPED)
    assert second["cached"] is True and len(a.calls) == 1 and second["advisory"]["summary"] == first["advisory"]["summary"]
    assert first["advisory"]["drivers"] == template.render(CTX)["drivers"] and second["advisory"]["drivers"] == template.render(CTX_FLIPPED)["drivers"]
    assert first["advisory"]["drivers"] != second["advisory"]["drivers"]


def test_provider_prose_that_states_a_direction_is_refused():
    r = Advisor([fake("gemini", dict(GOOD, meaning="Elevation is above typical, which lowers the score."))], now=Clock()).advise(CTX)
    assert r["source"] == "template" and r["attempts"][0]["status"] == "rejected" and "direction_word" in r["attempts"][0]["reasons"]
