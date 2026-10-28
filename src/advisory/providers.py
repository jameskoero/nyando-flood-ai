"""Phase E providers (docs/PHASE_E_PROTOCOL.md Section 5, docs/PHASE_E_ADVISORY.md Section 6): Gemini and Groq over the standard library. A key travels only in a request header; an error carries only its kind and an HTTP status, never a key, a URL or a response body."""
import http.client
import json
import socket
import urllib.error
import urllib.request

from src.advisory.rules import PROVIDERS

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_RESPONSE_BYTES = 1000000
MAX_OUTPUT_TOKENS = 400
TEMPERATURE = 0.2


class ProviderError(Exception):
    """kind is one of: timeout, http, network, shape."""

    def __init__(self, kind, detail=""):
        super().__init__(kind)
        self.kind, self.detail = kind, detail


class Provider:
    def __init__(self, name, model, call):
        self.name, self.model, self.call = name, model, call

    def __repr__(self):
        return "Provider(%s, %s)" % (self.name, self.model)


def post_json(url, headers, payload, timeout_s):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=dict(headers, **{"Content-Type": "application/json"}), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read(MAX_RESPONSE_BYTES).decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise ProviderError("http", "status %d" % e.code) from None
    except TimeoutError:
        raise ProviderError("timeout") from None
    except urllib.error.URLError as e:
        raise ProviderError("timeout" if isinstance(e.reason, TimeoutError) else "network") from None
    except (OSError, http.client.HTTPException):
        raise ProviderError("network") from None
    except ValueError:
        raise ProviderError("shape") from None


def gemini(api_key, model, base=GEMINI_BASE):
    def call(system, user, timeout_s):
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": TEMPERATURE, "maxOutputTokens": MAX_OUTPUT_TOKENS},
        }
        data = post_json(base + model + ":generateContent", {"x-goog-api-key": api_key}, payload, timeout_s)
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("shape") from None
        if not isinstance(text, str):
            raise ProviderError("shape")
        return text
    return Provider("gemini", model, call)


def groq(api_key, model, url=GROQ_URL):
    def call(system, user, timeout_s):
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "temperature": TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        data = post_json(url, {"Authorization": "Bearer " + api_key}, payload, timeout_s)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("shape") from None
        if not isinstance(text, str):
            raise ProviderError("shape")
        return text
    return Provider("groq", model, call)


MAKERS = {"gemini": gemini, "groq": groq}


def providers_from_env(environ):
    """The providers whose key and model are both set, in the protocol's order."""
    out = []
    for name in PROVIDERS:
        key, model = environ.get(name.upper() + "_API_KEY", "").strip(), environ.get(name.upper() + "_MODEL", "").strip()
        if key and model:
            out.append(MAKERS[name](key, model))
    return out
