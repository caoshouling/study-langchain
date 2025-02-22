import os
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from typing import Literal


from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from langgraph.prebuilt import ToolNode
@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."


@tool
def get_coolest_cities():
    """Get a list of coolest cities"""
    return "nyc, sf"

tools = [get_weather, get_coolest_cities]
tool_node = ToolNode(tools)
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END




model_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

model_with_tools = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1/",
    # model="paultimothymooney/qwen2.5-7b-instruct",
    model = "triangle104/qwen2.5-7b-instruct",
    api_key="323"
).bind_tools(tools)
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

app = workflow.compile()

print("---------------------------------------------")
# example with a single tool call
for chunk in app.stream(
    {"messages": [("human", "what's the weather in sf?")]}, stream_mode="values"
):
    chunk["messages"][-1].pretty_print()
print("---------------------------------------------")


for chunk in app.stream(
    {"messages": [("human", "what's the weather in the coolest cities?")]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()


graph_png = app.get_graph().draw_mermaid_png()
with open(os.path.basename(__file__).replace("py","png"), "wb") as f:
    f.write(graph_png)