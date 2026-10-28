"""Phase E orchestrator (docs/PHASE_E_PROTOCOL.md Section 5, docs/PHASE_E_ADVISORY.md Section 6): the providers in order, each answer through the guard, then the deterministic template. It never raises and never returns a provider message. Only a provider answer that the guard accepted is cached."""
import json
import threading
import time

from src.advisory import guard, prompt, template
from src.advisory.providers import ProviderError

PER_PROVIDER_S = 8
TOTAL_S = 12
CACHE_TTL_S = 3600
CACHE_MAX = 512


class Cache:
    def __init__(self, ttl_s=CACHE_TTL_S, max_items=CACHE_MAX):
        self.ttl_s, self.max_items, self._items, self._lock = ttl_s, max_items, {}, threading.Lock()

    def get(self, key, now):
        with self._lock:
            hit = self._items.get(key)
            if hit is None:
                return None
            if now >= hit[0]:
                del self._items[key]
                return None
            return hit[1]

    def put(self, key, value, now):
        with self._lock:
            if len(self._items) >= self.max_items:
                for k in [k for k, (exp, _) in self._items.items() if now >= exp]:
                    del self._items[k]
            while len(self._items) >= self.max_items:
                del self._items[next(iter(self._items))]
            self._items[key] = (now + self.ttl_s, value)


def _parse(text):
    try:
        data = json.loads(text)
    except ValueError:
        raise ProviderError("shape") from None
    if not isinstance(data, dict):
        raise ProviderError("shape")
    return data


class Advisor:
    def __init__(self, providers, cache=None, now=time.monotonic):
        self.providers, self.cache, self.now = list(providers), cache if cache is not None else Cache(), now

    def advise(self, context):
        key = prompt.cache_key(context)
        hit = self.cache.get(key, self.now())
        if hit is not None:
            return dict(hit, cached=True)
        started, attempts, chosen = self.now(), [], None
        for p in self.providers:
            remaining = TOTAL_S - (self.now() - started)
            if remaining <= 0:
                attempts.append({"provider": p.name, "status": "skipped"})
                continue
            try:
                text = _parse(p.call(prompt.SYSTEM, prompt.user_content(context), min(PER_PROVIDER_S, remaining)))
            except ProviderError as e:
                attempts.append({"provider": p.name, "status": e.kind})
                continue
            except Exception as e:
                attempts.append({"provider": p.name, "status": "error: " + type(e).__name__})
                continue
            reasons = guard.check(text)
            if reasons:
                attempts.append({"provider": p.name, "status": "rejected", "reasons": reasons})
                continue
            attempts.append({"provider": p.name, "status": "accepted"})
            chosen = (p, text)
            break
        if chosen:
            body, source, model = {k: chosen[1][k] for k in guard.KEYS}, chosen[0].name, chosen[0].model
        else:
            body, source, model = template.render(context), "template", None
        result = {"advisory": template.with_limits(body), "source": source, "model": model, "prompt_version": prompt.PROMPT_VERSION, "attempts": attempts, "cached": False}
        if chosen:
            self.cache.put(key, result, self.now())
        return dict(result)
