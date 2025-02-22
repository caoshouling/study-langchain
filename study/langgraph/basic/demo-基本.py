import os
from typing import Annotated

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)


graph_builder.add_edge(START, "chatbot")

graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

for event in graph.stream({"messages": [{"role": "user", "content": "你好"}]}):
    for value in event.values():
        print("Assistant:", value["messages"][-1].content)
# 将生成的图片保存到文件
graph_png = graph.get_graph().draw_mermaid_png()
with open(os.path.basename(__file__).replace("py","png"), "wb") as f:
    f.write(graph_png)
