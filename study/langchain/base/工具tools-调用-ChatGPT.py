import time

from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_weather(city):
    """根据城市获取天气数据"""
    ## 睡眠1秒，模拟网络请求
    time.sleep(1)
    return f"{city}的天气白天是晴天，晚上下雨"
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [multiply,get_weather]

print(get_weather.invoke("南京"))

prompt=hub.pull("hwchase17/openai-functions-agent")


llm =ChatOpenAI(model_name="gpt-3.5-turbo")
# 这个方法貌似只有ChatGPT支持，兼容OpenAI的开源模型不支持
agent= create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor =AgentExecutor(agent=agent, tools=tools,verbose=True)
result = agent_executor.invoke({"input": "南京的天气"})
print(result)