# pip install google-search-results
import logging
import time
from langchain.globals import set_debug
from langchain_core.globals import set_verbose
from langchain_core.tools import tool, Tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
'''

该实施基于基础反应论文，但较旧，不适合生产应用。
对于更强大且功能丰富的实现，我们建议使用langgraph库中的create_react_agent函数。

在本地部署的Qwen 7B 或 14B模型，会有些问题，比如回答getSchoolMessage会出现问题

'''
logging.basicConfig(level=logging.DEBUG)
set_debug(True)
set_verbose(True)
@tool
def getSchoolMessage(name):
    """根据人名获取毕业学校和专业"""
    ## 睡眠1秒，模拟网络请求
    time.sleep(1)
    return f"{name}2008年毕业于剑桥大学计算机系，专业是软件工程专业"


@tool
def multiply(a: int, b: int) -> int:
    """计算两个数字相乘"""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """计算两个数字相加"""
    return a + b


tools = [add, multiply, getSchoolMessage]

# 支持性不好， 调用搜索是支持的，但是调用函数支持性不好
llm = ChatOpenAI(
    openai_api_base="https://8f13-154-12-181-41.ngrok-free.app/v1/",
    model="paultimothymooney/qwen2.5-7b-instruct",
    # model = "qwen2.5-14b-instruct",
    api_key="323"

)
#  支持性好
# llm = ChatOpenAI(model_name="gpt-3.5-turbo")


#  支持性好
# llm = ChatOpenAI(openai_api_base= "https://dashscope.aliyuncs.com/compatible-mode/v1",
#                    model_name='qwen-turbo',
#                  api_key ="sk-474e1a10893e4913bbe860dc90edda42")

#
llm = ChatOpenAI(openai_api_base="https://api.deepseek.com/v1",
                   model_name='deepseek-chat',
                 api_key ="sk-37b28ddd69354f6e8f813de3a0f218f2")

import os

# 导入LangChain Hub
from langchain import hub

# 从hub中获取React的Prompt
prompt = hub.pull("hwchase17/react")
print(prompt)

prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
    template='''
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
''')

# 导入ChatOpenAI
from langchain_community.llms import OpenAI

# 导入SerpAPIWrapper即工具包
from langchain_community.utilities import SerpAPIWrapper

os.environ["SERPAPI_API_KEY"] = "ce02cbc1fa127d958c347aec81e6356dbe774ba00ac600a7be2777303484e7a8"
# 实例化SerpAPIWrapper
search = SerpAPIWrapper()

# 准备工具列表
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="当大模型没有相关知识时，用于搜索知识"
    ),
    Tool(
        name="getSchoolMessage",
        func=getSchoolMessage,
        description="根据人名获取毕业学校和专业"
    ),
    Tool(
        name="multiply",
        func=multiply,
        description="计算两个数相乘"
    ),
    Tool(
        name="add",
        func=add,
        description="计算两个数相加"
    ),
]

# 导入create_react_agent功能
from langchain.agents import create_react_agent
# 导入AgentExecutor
from langchain.agents import AgentExecutor

# 构建ReAct代理
'''
create_react_agent： 对应的agent可以处理多个任务，但是调用工具时，
  输入参数只允许有1个参数，如果为多个参数则报错。


'''
agent = create_react_agent(llm, tools, prompt)

# 创建代理执行器并传入代理和工具
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 调用代理执行器，传入输入数据
# print("第一次运行的结果：")
# result = agent_executor.invoke({"input": "2025央视春晚，王菲唱的歌曲名叫什么？"})
# print(result)
print("第二次运行的结果：")
result = agent_executor.invoke({"input": "张三毕业于哪所大学，学的什么专业？"})
print(result)
# print("第三次运行的结果：")
# result = agent_executor.invoke({"input": "小明有120只鸭子，后面朋友送了他99只鸭子，他一共有多少只鸭子？"})
# print(result)
