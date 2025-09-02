import time

from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_weather(city):
    """根据城市获取天气数据"""
    ## 睡眠1秒，模拟网络请求
    time.sleep(1)
    return f"{city}的天气白天是晴天，晚上下雨"
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b
tools = [multiply,add,get_weather]

print(get_weather.invoke("南京"))

print("---------------------调用方式一:create_openai_functions_agent--------------------------")

prompt=hub.pull("hwchase17/openai-functions-agent")

llm =ChatOpenAI(model_name="gpt-3.5-turbo")
llm = ChatOpenAI(openai_api_base= "https://dashscope.aliyuncs.com/compatible-mode/v1",
                       model_name='qwen-turbo',
                     api_key ="sk-d9ca67dd361c4347b582386197867c05")
# 这个方法貌似只有ChatGPT支持，兼容OpenAI的开源模型不支持
agent= create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor =AgentExecutor(agent=agent, tools=tools,verbose=True)
result = agent_executor.invoke({"input": "南京的天气"})
print(result)


print("---------------------调用方式二:create_openai_tools_agent--------------------------")

prompt = hub.pull("hwchase17/openai-tools-agent")

llm =ChatOpenAI(model_name="gpt-3.5-turbo")
llm = ChatOpenAI(openai_api_base= "https://dashscope.aliyuncs.com/compatible-mode/v1",
                       model_name='qwen-turbo',
                     api_key ="sk-d9ca67dd361c4347b582386197867c05")


agent= create_openai_tools_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor =AgentExecutor(agent=agent, tools=tools,verbose=True)
result = agent_executor.invoke({"input": "南京的天气"})
print(result)



print("---------------------调用方式三:bind_tools--------------------------")

llm_with_tools = llm.bind_tools(tools)
query = "What is 3 * 12? Also, what is 11 + 49?"
messages = [HumanMessage(query)]

ai_msg = llm_with_tools.invoke(messages)

print(ai_msg.tool_calls)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    tool_map = {tool.name: tool for tool in tools}
    selected_tool = tool_map[tool_call["name"].lower()]
    # 调用工具
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)
'''
[HumanMessage(content='What is 3 * 12? Also, what is 11 + 49?'),
 AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_loT2pliJwJe3p7nkgXYF48A1', 'function': {'arguments': '{"a": 3, "b": 12}', 'name': 'multiply'}, 'type': 'function'}, {'id': 'call_bG9tYZCXOeYDZf3W46TceoV4', 'function': {'arguments': '{"a": 11, "b": 49}', 'name': 'add'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 50, 'prompt_tokens': 87, 'total_tokens': 137}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_661538dc1f', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-e3db3c46-bf9e-478e-abc1-dc9a264f4afe-0', tool_calls=[{'name': 'multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_loT2pliJwJe3p7nkgXYF48A1', 'type': 'tool_call'}, {'name': 'add', 'args': {'a': 11, 'b': 49}, 'id': 'call_bG9tYZCXOeYDZf3W46TceoV4', 'type': 'tool_call'}], usage_metadata={'input_tokens': 87, 'output_tokens': 50, 'total_tokens': 137}),
 ToolMessage(content='36', name='multiply', tool_call_id='call_loT2pliJwJe3p7nkgXYF48A1'),
 ToolMessage(content='60', name='add', tool_call_id='call_bG9tYZCXOeYDZf3W46TceoV4')]

'''

print(f'messages：{messages}')

result  = llm.invoke(messages)
'''
AIMessage(content='The result of \\(3 \\times 12\\) is 36, and the result of \\(11 + 49\\) is 60.',
  response_metadata={'token_usage': {'completion_tokens': 31, 'prompt_tokens': 153, 'total_tokens': 184},
 'model_name': 'gpt-4o-mini-2024-07-18',
 'system_fingerprint': 'fp_661538dc1f', 'finish_reason': 'stop', 'logprobs': None},
   id='run-87d1ef0a-1223-4bb3-9310-7b591789323d-0',
   usage_metadata={'input_tokens': 153, 'output_tokens': 31, 'total_tokens': 184})

'''
print(f'result：{result}')