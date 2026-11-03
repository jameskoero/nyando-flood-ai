"""Phase H wrappers: the MCP tools and the LangGraph chain over the replay core.

They need the packages in requirements-agent.txt, so they run in the agent-check workflow
(AGENT_WRAPPERS_REQUIRED=1, where a missing package is a failure) and are skipped where
those packages are not installed. The transport stub returns labelled STUB values."""
import os

from src.agent import core

REQUIRED = os.environ.get("AGENT_WRAPPERS_REQUIRED") == "1"
ROWS, N_MAPPABLE = core.load_primary()


def stub_post(calls):
    def post(url, payload):
        calls.append(url.rsplit("/v2/", 1)[1])
        if url.endswith("/v2/score"):
            return 200, {"score": 0.25, "model": {"id": "STUB"}}
        return 200, {
            "score": 0.25, "model": {"id": "STUB"}, "level": "STUB", "level_basis": "STUB BASIS",
            "advisory": "STUB TEXT", "advisory_source": "stub", "advisory_model": None,
            "notice": "STUB NOTICE", "attempts": [{"provider": "STUB"}],
            "providers_configured": ["stub"], "claim_limit": "STUB CLAIM",
            "score_meaning": "STUB MEANING",
        }
    return post


def test_the_chain_gives_the_same_report_as_the_core():
    try:
        from src.agent import graph
    except ImportError:
        if REQUIRED:
            raise
        return
    a, b = [], []
    out = graph.build_graph(post=stub_post(a)).invoke({"generated": "TEST"})
    want = core.run_replay(core.plan(ROWS, N_MAPPABLE), post=stub_post(b), generated="TEST")
    assert out["report"] == want and a == b and len(a) == 2 * len(core.plan(ROWS)["picked"])


def test_the_mcp_tools_call_the_service_through_the_core():
    try:
        import anyio
        from mcp import Client
        from src.agent import mcp_server
    except ImportError:
        if REQUIRED:
            raise
        return
    calls = []
    mcp_server.set_transport(stub_post(calls))
    args = core.to_payload(ROWS[0])

    def body(result):
        sc = result.structured_content
        return sc["result"] if list(sc) == ["result"] else sc

    async def run():
        async with Client(mcp_server.mcp, raise_exceptions=True) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert names == {"score_location", "advise_location", "replay_event_report"}
            r = await client.call_tool("score_location", args)
            assert r.is_error is False and body(r)["score"] == 0.25
            r = await client.call_tool("advise_location", args)
            assert r.is_error is False and body(r)["notice"] == "STUB NOTICE"
            r = await client.call_tool("replay_event_report", {})
            assert r.is_error is False and "Replay of recorded data" in r.content[0].text

    anyio.run(run)
    assert calls[:2] == ["score", "advisory"] and len(calls) == 2 + 2 * len(core.plan(ROWS)["picked"])
