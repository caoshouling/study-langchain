import logging
import os

from langchain import hub

from langchain.agents import (
    AgentExecutor, create_self_ask_with_search_agent
)
from langchain_community.tools import TavilyAnswer, TavilySearchResults
from langchain_community.utilities import SerpAPIWrapper
from langchain.globals import set_debug, set_verbose
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
logging.basicConfig(level=logging.DEBUG)
set_debug(True)
set_verbose(True)
prompt = hub.pull("hwchase17/self-ask-with-search")
model = ChatOpenAI(openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model_name='qwen-turbo',
                 api_key="sk-474e1a10893e4913bbe860dc90edda42")
# model = ChatOpenAI(model_name="gpt-3.5-turbo")
# model = ChatOpenAI(model_name="gpt-4o-mini")
os.environ["TAVILY_API_KEY"] = "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"

tool = TavilySearchResults(max_results=2)
result = tool.invoke("南京的天气是怎样的?")
print(f'TavilySearchResults搜索返回: {result}')


result = TavilyAnswer().invoke("南京的天气是怎样的?")
print(f'TavilyAnswer搜索返回: {result}')

# 实例化SerpAPIWrapper
os.environ["SERPAPI_API_KEY"] = "ce02cbc1fa127d958c347aec81e6356dbe774ba00ac600a7be2777303484e7a8"
search = SerpAPIWrapper()
result = search.run("南京的天气是怎样的?")
print(f'SerpAPIWrapper搜索返回: {result}')
# Should just be one tool with name `Intermediate Answer`
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
]
agent = create_self_ask_with_search_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)

result =agent_executor.invoke({"input": "上个赛季的NBA总冠军是哪支球队？"})
print(result)