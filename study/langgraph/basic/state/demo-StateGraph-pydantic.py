from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from pydantic import BaseModel
'''
采用：pydantic

'''

# The overall state of the graph (this is the public state shared across nodes)
class OverallState(BaseModel):
    a: str


def bad_node(state: OverallState):
    return {
        "a": 123  # Invalid
    }


def ok_node(state: OverallState):
    return {"a": "goodbye"}


# Build the state graph
builder = StateGraph(OverallState)
builder.add_node(bad_node)
builder.add_node(ok_node)
builder.add_edge(START, "bad_node")
builder.add_edge("bad_node", "ok_node")
builder.add_edge("ok_node", END)
graph = builder.compile()

# Test the graph with a valid input
try:
    graph.invoke({"a": "hello"})
except Exception as e:
    print("An exception was raised because bad_node sets `a` to an integer.")
    print(e)