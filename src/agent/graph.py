"""Phase H LangGraph chain: fetch -> predict -> advise -> report over the replay core.

Needs langgraph (requirements-agent.txt). The nodes are the core's own functions, so the
chain and core.run_replay give the same report for the same transport. Run live:
    python -m src.agent.graph [--out FILE]
"""
import sys
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import core


class State(TypedDict, total=False):
    plan: dict
    scores: dict
    advice: dict
    generated: str
    report: str


def build_graph(post=core.http_post, load=core.load_primary, **kw):
    def fetch(state):
        rows, n_mappable = load()
        return {"plan": core.plan(rows, n_mappable)}

    def predict(state):
        return {"scores": core.predict_step(post, state["plan"]["picked"], **kw)}

    def advise(state):
        return {"advice": core.advise_step(post, state["plan"]["picked"], **kw)}

    def report(state):
        problems = core.check_consistency(state["scores"], state["advice"])
        if problems:
            raise core.ApiError("; ".join(problems))
        stamp = state.get("generated") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return {"report": core.build_report(state["plan"], state["scores"], state["advice"], stamp)}

    g = StateGraph(State)
    for name, fn in (("fetch", fetch), ("predict", predict), ("advise", advise), ("report", report)):
        g.add_node(name, fn)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "predict")
    g.add_edge("predict", "advise")
    g.add_edge("advise", "report")
    g.add_edge("report", END)
    return g.compile()


if __name__ == "__main__":
    out = build_graph().invoke({})["report"]
    if len(sys.argv) > 2 and sys.argv[1] == "--out":
        with open(sys.argv[2], "w", encoding="utf-8") as fh:
            fh.write(out + "\n")
    else:
        print(out)
