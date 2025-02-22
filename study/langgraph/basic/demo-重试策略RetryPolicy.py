import operator
import sqlite3
from typing import Annotated, Sequence

from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage

from langgraph.graph import END, StateGraph, START
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage

db = SQLDatabase.from_uri("sqlite:///:memory:")

model = ChatAnthropic(model_name="claude-2.1")

'''
RetryPolicy()

RetryPolicy(initial_interval=0.5, backoff_factor=2.0, max_interval=128.0, max_attempts=3, jitter=True, retry_on=<function default_retry_on at 0x78b964b89940>)
默认情况下，该retry_on参数使用该default_retry_on函数，该函数会对除以下情况之外的任何异常进行重试：

ValueError
TypeError
ArithmeticError
ImportError
LookupError
NameError
SyntaxError
RuntimeError
ReferenceError
StopIteration
StopAsyncIteration
OSError
此外，对于流行的 http 请求库中的异常（例如requests），httpx它仅对 5xx 状态代码进行重试。

'''

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def query_database(state):
    query_result = db.run("SELECT * FROM Artist LIMIT 10;")
    return {"messages": [AIMessage(content=query_result)]}


def call_model(state):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# Define a new graph
builder = StateGraph(AgentState)
builder.add_node(
    "query_database",
    query_database,
    retry=RetryPolicy(retry_on=sqlite3.OperationalError),
)
builder.add_node("model", call_model, retry=RetryPolicy(max_attempts=5))
builder.add_edge(START, "model")
builder.add_edge("model", "query_database")
builder.add_edge("query_database", END)

graph = builder.compile()