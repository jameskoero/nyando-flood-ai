"""Phase H MCP server: three tools over the replay core, on stdio.

Needs mcp 2.x (requirements-agent.txt); the core does not. Run from the repo root:
    python -m src.agent.mcp_server
"""
from typing import Any

from mcp.server import MCPServer

from . import core

mcp = MCPServer("nyando-flood-ai")
POST = core.http_post  # tests replace it, so nothing touches the network


def set_transport(post):
    global POST
    POST = post


def _payload(elevation, slope, rainfall_3day, distance_river, land_cover, ward, clay_percent):
    p = {"elevation": elevation, "slope": slope, "rainfall_3day": rainfall_3day,
         "distance_river": distance_river, "land_cover": land_cover, "ward": ward}
    if clay_percent is not None:
        p["clay_percent"] = clay_percent
    return p


@mcp.tool()
def score_location(elevation: float, slope: float, rainfall_3day: float, distance_river: float,
                   land_cover: int, ward: str = "Unknown",
                   clay_percent: float | None = None) -> dict[str, Any]:
    """Score one location with the registered model (POST /v2/score).

    The score ranks locations; it is not a flood probability."""
    return core.call(POST, "/v2/score", _payload(
        elevation, slope, rainfall_3day, distance_river, land_cover, ward, clay_percent))


@mcp.tool()
def advise_location(elevation: float, slope: float, rainfall_3day: float, distance_river: float,
                    land_cover: int, ward: str = "Unknown",
                    clay_percent: float | None = None) -> dict[str, Any]:
    """Score one location and return its band and advisory text (POST /v2/advisory).

    It is not an official warning and gives no lead time."""
    return core.call(POST, "/v2/advisory", _payload(
        elevation, slope, rainfall_3day, distance_river, land_cover, ward, clay_percent))


@mcp.tool()
def replay_event_report() -> str:
    """Replay the recorded Block A event with the highest mean rainfall through the live
    service and return the one-page situation report. It makes 10 calls and can take minutes.
    It is an illustration, not a validation and not a forecast."""
    rows, n_mappable = core.load_primary()
    return core.run_replay(core.plan(rows, n_mappable), post=POST)


if __name__ == "__main__":
    mcp.run()
