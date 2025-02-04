import json
import logging
import operator
import os
import time
from decimal import Decimal, localcontext
from typing import Annotated, List, TypedDict

from langchain.globals import set_debug, set_verbose
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, render_text_description_and_args
from langchain_core.utils.json import parse_json_markdown
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph

"""
  这个agent对于简单的能处理，对于复杂的有问题。比如需要连续调用三个工具，工具之间有依赖。
"""


logging.basicConfig(level=logging.DEBUG)
set_debug(True)
set_verbose(True)
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langsmith-langgraph"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# llm = ChatOpenAI(
#     openai_api_base="https://8f13-154-12-181-41.ngrok-free.app/v1/",
#     # model="paultimothymooney/qwen2.5-7b-instruct",
#     model = "qwen2.5-14b-instruct",
#     api_key="323",
#     temperature=0
# )
#
# llm = ChatOpenAI(openai_api_base= "https://dashscope.aliyuncs.com/compatible-mode/v1",
#                    model_name='qwen-turbo',
#                  api_key ="sk-474e1a10893e4913bbe860dc90edda42",
#                  temperature=0)

prompt_template = """尽可能帮用户回答任何问题。

您可以使用以下工具帮忙解决问题，如果已经知道答案，也可以直接回答。

{tools}

回复我时，请以下面2种格式之一进行回复：

选项一：如果您希望使用工具，请使用此JSON格式回复内容：
```json
{{
 "reason": string, \\叙述使用工具的原因
 "action": string, \\要使用的工具。工具必须是{tool_names}之一
 "action_input": string \\工具的输入
}}
```
选项二：如果您以为您已经有答案或者已经通过工具找到了答案，想直接对用户作出答复，请使用此JSON格式化的回复：
```json
{{
  "action": "Final Answer",
  "answer": string \\最终回复问题的答案放在这里
}}
```

下面是用户的输入，请记住只回复上面两种格式的其中一种，且必须以Json格式回复，不要回复其他内容。
用户的输入：{input}
"""
# 创建提示模板
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "你是一个非常强大的助手，你可以使用各种工具来完成人类交给的问题和任务。"),
    ("human", prompt_template),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])



@tool
def get_weather(city):
    """根据输入的城市获取天气数据"""
    ## 睡眠1秒，模拟网络请求
    time.sleep(1)
    return f"今天{city}的天气白天是晴天，晚上下雨"

@tool
def get_sales(city: str) -> float:
    """根据地区获取销售额"""

    time.sleep(1)
    return 129.9
@tool
def get_cost(city: str) -> float:
    """根据地区获取销售成本"""

    time.sleep(1)
    return 100.0
@tool
def get_profit(city:str,sales: float, cost: float) -> float:
    """根据地区、销售金额、销售成本计算增长值。"""

    time.sleep(1)
    return sales - cost  - 10

@tool
def subtract_float(amount1: float, amount2: float) -> Decimal:
    """
    计算两个数字的差值。
    """
    # 将 float 转换为 Decimal
    decimal_amount1 = Decimal(str(amount1))
    decimal_amount2 = Decimal(str(amount2))

    # 使用局部上下文设置精度
    with localcontext() as ctx:
        ctx.prec = 10  # 设置局部精度
        result = decimal_amount1 - decimal_amount2

    return result
@tool
def add_float(amount1: float, amount2: float) -> Decimal:
    """
    计算两个数字的和。
    """
    # 将 float 转换为 Decimal
    decimal_amount1 = Decimal(str(amount1))
    decimal_amount2 = Decimal(str(amount2))

    # 使用局部上下文设置精度
    with localcontext() as ctx:
        ctx.prec = 10  # 设置局部精度
        result = decimal_amount1 + decimal_amount2

    return result
@tool
def serp_search(query:str):
    """根据输入关键字，使用SerpAPI调用谷歌搜索引擎进行搜索"""
    search = SerpAPIWrapper()
    return search.run(query)

tools = [get_weather,get_sales,get_cost,add_float,subtract_float,get_profit]
# 部分填充提示模板
prompt = prompt1.partial(
    tools=render_text_description_and_args(tools),
    tool_names=",".join([t.name for t in tools]),
)
print(prompt.format_messages(input = "23233", agent_scratchpad = []))
# 定义状态
class GState(TypedDict):
    messages:Annotated[List, operator.add]
# 是否需要使用工具
def isUseTool(gState: GState):
    print("-------------isUseTool--gState----------")
    print(gState)
    obj = gState["messages"][-1]
    if obj["action"] == "Final Answer":
        return END
    return "toolNode"
# 使用工具
def useTool(gState: GState):
    print("-------------useTool--gState----------")
    print(gState)
    obj = gState["messages"][-1]
    for tool in tools:
        if tool.name == obj["action"]:
            return {"messages": [tool.invoke(obj["action_input"])]}
    print(f"工具{obj['action']}不存在")

# 定义图
agentGraph = StateGraph(GState)
# 格式化中间步骤，支持智能体在作出最终决策前进行多轮内部评估
def startParse(gState):
    print("-------------startParse----------------")
    print(gState)
    tool_response_prompt =  """工具响应：
    ------------------------
    {tools_response}
    
    请根据工具的响应判断，是否能够回答问题：
    
    {input}
    
    请根据工具响应的内容思考接下来的回复。回复格式严格按照前面所说的2种JSON回复格式，选择其中1种进行回复。请记住只选择单个选项格式，以JSON格式化的内容进行回复，不要回复其他内容！
    """
    if len(gState["messages"]) > 1:
        agent_scratchpad = [
            AIMessage(json.dumps(gState["messages"][-2])),
            HumanMessage(
                content = tool_response_prompt.format(
                    input=gState["messages"][0],
                    tools_response=json.dumps(gState["messages"][-1])
                )
            )
        ]
    else:
        agent_scratchpad = []
    return {
        "input": gState["messages"][0],
        "agent_scratchpad": agent_scratchpad
    }

def startMsgParse(message):
    message = message.content.replace("'", '"')
    message = parse_json_markdown(message)
    return {"messages": [message]}

strParser =  StrOutputParser()
chain = prompt | llm

agentGraph.add_node("startNode", startParse | chain | startMsgParse)
agentGraph.add_node("toolNode", useTool)
agentGraph.add_conditional_edges("startNode", isUseTool)
agentGraph.add_edge("toolNode", "startNode")
agentGraph.set_entry_point("startNode")
agent = agentGraph.compile()
# 可视化图
graph_png = agent.get_graph().draw_mermaid_png()
with open("自定义智能体.png", "wb") as f:
    f.write(graph_png)
# 显示图

result  = agent.invoke({"messages": ["南京地区的增长值是多少？"]})

print(result)

