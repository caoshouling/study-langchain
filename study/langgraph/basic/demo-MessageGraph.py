import os

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessageGraph

os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langsmith-langgraph"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
# 这个state就是invoke的参数

def my_node(state):
    print(state)
    return state

builder = MessageGraph()


builder.add_node("my_node",my_node)

builder.add_node("chatbot", lambda state: [AIMessage(content="你好,我是你的聊天助手。")])
builder.add_edge("my_node","chatbot")

builder.set_entry_point("my_node")
builder.set_finish_point("chatbot")

graph = builder.compile()

print(graph.invoke([HumanMessage(content="你好")]))

graph_png = graph.get_graph().draw_mermaid_png()
with open(os.path.basename(__file__).replace("py","png"), "wb") as f:
    f.write(graph_png)
# 显示图