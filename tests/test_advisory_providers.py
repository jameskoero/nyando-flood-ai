"""Phase E providers (docs/PHASE_E_ADVISORY.md Section 6; register row D49): the Gemini and Groq callers against local stand-in servers: the request each one sends, the answer it reads, and how every failure is reported without leaking a key."""
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time
from src.advisory import providers
from src.advisory.orchestrator import Advisor
from src.advisory.providers import ProviderError

GOOD = {"summary": "For Ahero: this location's score is in the highest third of the reference scores.", "drivers": [], "meaning": "This ranks locations; it does not forecast an event."}
CTX = {"ward": "Ahero", "level": "HIGH", "drivers": []}


@contextmanager
def stand_in(status=200, body=None, delay=0.0):
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.update(path=self.path, headers={k.lower(): v for k, v in self.headers.items()}, body=json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            time.sleep(delay)
            raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        daemon_threads, block_on_close = True, False

        def handle_error(self, request, client_address):
            pass

    srv = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield "http://127.0.0.1:%d" % srv.server_address[1], seen
    finally:
        srv.shutdown()
        srv.server_close()


def gemini_reply(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def groq_reply(text):
    return {"choices": [{"message": {"content": text}}]}


def test_gemini_sends_the_documented_request_and_reads_the_documented_answer():
    with stand_in(body=gemini_reply("{\"a\": 1}")) as (base, seen):
        p = providers.gemini("KEY-G", "test-model", base=base + "/v1beta/models/")
        assert p.call("SYSTEM TEXT", "USER TEXT", 5) == "{\"a\": 1}"
    assert seen["path"] == "/v1beta/models/test-model:generateContent" and "KEY-G" not in seen["path"]
    assert seen["headers"]["x-goog-api-key"] == "KEY-G" and seen["headers"]["content-type"] == "application/json"
    b = seen["body"]
    assert b["systemInstruction"]["parts"][0]["text"] == "SYSTEM TEXT" and b["contents"][0]["parts"][0]["text"] == "USER TEXT"
    assert b["generationConfig"]["responseMimeType"] == "application/json" and "KEY-G" not in json.dumps(b)


def test_groq_sends_the_documented_request_and_reads_the_documented_answer():
    with stand_in(body=groq_reply("{\"a\": 1}")) as (base, seen):
        p = providers.groq("KEY-Q", "test-model", url=base + "/openai/v1/chat/completions")
        assert p.call("SYSTEM TEXT", "USER TEXT", 5) == "{\"a\": 1}"
    assert seen["path"] == "/openai/v1/chat/completions" and seen["headers"]["authorization"] == "Bearer KEY-Q"
    b = seen["body"]
    assert b["model"] == "test-model" and b["messages"] == [{"role": "system", "content": "SYSTEM TEXT"}, {"role": "user", "content": "USER TEXT"}]
    assert b["response_format"] == {"type": "json_object"} and "KEY-Q" not in json.dumps(b)


def failure(make, status=200, body=None, delay=0.0, timeout=5):
    with stand_in(status=status, body=body, delay=delay) as (base, _):
        try:
            make(base).call("s", "u", timeout)
        except ProviderError as e:
            return e
    raise AssertionError("no ProviderError")


def test_every_failure_is_reported_by_kind_without_a_key_or_a_body():
    for name, make, ok in (("gemini", lambda b: providers.gemini("KEY-G", "m", base=b + "/m/"), gemini_reply), ("groq", lambda b: providers.groq("KEY-Q", "m", url=b + "/c"), groq_reply)):
        for status in (401, 429, 500):
            e = failure(make, status=status, body={"error": "KEY-G KEY-Q body text"})
            assert (e.kind, e.detail) == ("http", "status %d" % status), (name, status)
            assert "KEY" not in e.kind + e.detail + str(e) + repr(e)
        assert failure(make, body=b"<html>not json</html>").kind == "shape"
        assert failure(make, body={"unexpected": True}).kind == "shape"
        assert failure(make, body=ok(12345)).kind == "shape"
        assert failure(make, body=ok("x"), delay=1.5, timeout=0.3).kind == "timeout"
    with stand_in() as (base, _):
        pass
    try:
        providers.groq("KEY-Q", "m", url=base + "/c").call("s", "u", 2)
    except ProviderError as e:
        assert e.kind == "network" and "KEY-Q" not in str(e) + e.detail
    else:
        raise AssertionError("a closed port must fail")


def test_the_two_callers_work_through_the_orchestrator_end_to_end():
    with stand_in(body=gemini_reply("not json")) as (gb, _), stand_in(body=groq_reply(json.dumps(GOOD))) as (qb, qseen):
        adv = Advisor([providers.gemini("KEY-G", "m", base=gb + "/m/"), providers.groq("KEY-Q", "m", url=qb + "/c")])
        r = adv.advise(CTX)
    assert r["source"] == "groq" and [a["status"] for a in r["attempts"]] == ["shape", "accepted"] and r["advisory"]["summary"] == GOOD["summary"]
    assert "KEY" not in json.dumps(r) and "Ahero" in qseen["body"]["messages"][1]["content"]
