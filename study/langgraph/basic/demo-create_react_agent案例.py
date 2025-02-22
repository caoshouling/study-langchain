# First we initialize the model we want to use.
import os

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# For this tutorial we will use custom tool that returns pre-defined values for weather in two cities (NYC & SF)

from typing import Literal

from langchain_core.tools import tool


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]


# Define the graph

from langgraph.prebuilt import create_react_agent

#  也可以传送自定义提示词 prompt
# 也可以添加内存：memory = MemorySaver() ，然后在调用时设置线程ID config = {"configurable": {"thread_id": "1"}}
# 还可以添加人机交互：interrupt_before=["tools"],
graph = create_react_agent(model, tools=tools)

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
inputs = {"messages": [("user", "what is the weather in sf")]}
print_stream(graph.stream(inputs, stream_mode="values"))

inputs = {"messages": [("user", "who built you?")]}
print_stream(graph.stream(inputs, stream_mode="values"))

graph_png = graph.get_graph().draw_mermaid_png()
with open(os.path.basename(__file__).replace("py","png"), "wb") as f:
    f.write(graph_png)