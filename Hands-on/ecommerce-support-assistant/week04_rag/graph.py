"""
graph.py — the LangGraph wiring.

Shape (see Lecture 3, Module 8 mapping):
    Planner    -> Node        (planner.planner_node)
    Memory     -> State       (state.AgentState) + a checkpointer for
                               cross-turn persistence
    Tool       -> ToolNode    (tools.tool_node)
    Decision   -> Conditional edge (planner.should_continue)
    Workflow   -> the compiled Graph
    Final Output -> END

    planner --tool requested--> tools --always--> planner (the loop)
    planner --no tool requested--> END

Requires: pip install langgraph
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from planner import planner_node, should_continue
from tools import tool_node


def route_after_planner(state: AgentState) -> str:
    print("in route_after_planner")
    
    return "tools" if should_continue(state) else "end"


def build_graph():
    print("in build_graph")

    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {"tools": "tools", "end": END},
    )
    # after tools run, always go back to the planner to read the
    # results and decide what to do next — this is the loop
    workflow.add_edge("tools", "planner")

    # persists AgentState per thread_id across .invoke() calls — this
    # is what lets memory survive between Streamlit chat turns without
    # app.py re-sending the whole history by hand.
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


graph = build_graph()