from typing import TypedDict, Annotated

from langgraph.constants import START
from langgraph.graph import StateGraph, add_messages

print('------------------StateGraph定义方式1----------------------------')

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

print('------------------StateGraph定义方式2---------------------------')
# 这个state就是invoke的参数

def my_node(state):
    return {"x": state["x"] + 1, "y": state["y"] + 1}
builder = StateGraph(dict)


builder.add_node(my_node)

builder.add_edge(START,"my_node")

graph = builder.compile()
print(graph)
print(graph.invoke({"x":1,"y":2}))


graph_png = graph.get_graph().draw_mermaid_png()
with open("demo-StateGraph.png", "wb") as f:
    f.write(graph_png)
# 显示图