import time

from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor, create_react_agent, \
    create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_weather(city):
    """根据城市获取天气数据"""
    ## 睡眠1秒，模拟网络请求
    time.sleep(1)
    return f"今天{city}的天气白天是晴天，晚上下雨"
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [multiply,get_weather]

print(get_weather.invoke("南京"))
print("---------------------千问大模型--------------------------")
# 定义提示词
prompt =ChatPromptTemplate.from_template("""
 按照给定的格式回答以下问题。你可以使用下面这些工具：

    {tools}

 回答时需要遵循以下用---括起来的格式：

    ---
    Question: 我需要回答的问题
    Thought: 回答这个问题我需要做些什么
    Action: {tool_names} 中的其中一个工具名
    Action Input: 选择工具所需要的输入
    Observation: 选择工具返回的结果
    ...（这个思考/行动/行动输入/观察可以重复N次）
    Thought: 我现在知道最终答案
    Final Answer: 原始输入问题的最终答案
    ---

 现在开始回答，记得在给出最终答案前多按照指定格式进行一步一步的推理。

    Question: {input}
    {agent_scratchpad}
    """)


# llm = ChatOpenAI(openai_api_base="https://api.deepseek.com/v1",
#                    model_name='deepseek-chat',
#                  openai_api_key ="sk-37b28ddd69354f6e8f813de3a0f218f2")

# llm_base_url: str = "http://localhost:1234/v1/"
# 初始化语言模型
llm = ChatOpenAI(
    openai_api_base="https://8f13-154-12-181-41.ngrok-free.app/v1/",
    model="paultimothymooney/qwen2.5-7b-instruct",
api_key ="323")

## 提示词
system = '''尽可能帮助和准确地回应人类。您可以使用以下工具:

{tools}

使用json blob通过提供action key（工具名称）和action_input key（工具输入）来指定工具。
"action"的有效取值为: "Final Answer" or {tool_names}

每个$JSON_BLOB只提供一个action，如下所示：
```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

遵循此格式:

Question: 用户输入的问题
Thought:  回答这个问题我需要做些什么，尽可能考虑前面和后面的步骤
Action:   回答问题所选取的工具
```
$JSON_BLOB
```
Observation: 工具返回的结果
... (这个思考/行动/行动输入/观察可以重复N次)
Thought: 我现在知道最终答案
Action: 工具返回的结果信息
```
{{
  "action": "Final Answer",
  "action_input": "原始输入问题的最终答案"
}}
```
开始！提醒始终使用单个操作的有效json blob进行响应。必要时使用工具. 如果合适，直接回应。格式是Action：“$JSON_BLOB”然后是Observation'''

human = '''{input}

{agent_scratchpad}

(提醒:无论如何都要在JSON blob中响应!)'''

prompt_muti_param = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human),
    ]
)
agent= create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_muti_param
)

agent_executor =AgentExecutor(agent=agent, tools=tools,verbose=True)


result = agent_executor.invoke({"input": "今天南京的天气怎样？"})
print(result)


