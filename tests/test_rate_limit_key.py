"""The per-client rate-limit key (docs/SECURITY_BASELINE.md open item): the key is the connecting client's address header when it holds a valid IP and the socket address otherwise, and every limited route still carries the per-client and the shared limit."""
from pathlib import Path
from types import SimpleNamespace

from backend.main import _client_key

ROOT = Path(__file__).resolve().parent.parent


def req(header=None, host="10.0.0.1"):
    headers = {} if header is None else {"cf-connecting-ip": header}
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=host))


def test_a_valid_address_header_is_the_key_and_ipv6_is_normalised():
    assert _client_key(req("203.0.113.7")) == "203.0.113.7"
    assert _client_key(req(" 203.0.113.7 ")) == "203.0.113.7"
    assert _client_key(req("2001:DB8::1")) == "2001:db8::1"


def test_a_missing_or_invalid_header_falls_back_to_the_socket_address():
    for bad in (None, "", "abc", "999.1.1.1", "1.2.3.4, 5.6.7.8", "a" * 500):
        assert _client_key(req(bad, host="10.0.0.9")) == "10.0.0.9", bad


def test_clients_behind_one_proxy_get_different_keys_and_one_client_keeps_its_key():
    a, b = _client_key(req("203.0.113.7")), _client_key(req("203.0.113.8"))
    assert a != b and a == _client_key(req("203.0.113.7", host="10.0.0.2"))


def test_the_limiter_uses_the_client_key_and_every_limited_route_keeps_both_limits():
    src = (ROOT / "backend" / "main.py").read_text(encoding="utf-8")
    assert "Limiter(key_func=_client_key)" in src and "Limiter(key_func=get_remote_address)" not in src
    assert src.count("key_func=_global_key") >= 3
    assert '@limiter.limit("10/minute")' in src and '@limiter.limit("5/minute")' in src
