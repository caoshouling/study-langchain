import logging
import time
from typing import List, Union, Optional
from venv import logger
import re
import json
from langchain.agents import AgentOutputParser, AgentExecutor
from langchain.agents.agent import MultiActionAgentOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import render_text_description, tool, render_text_description_and_args
from langchain_core.utils.json import parse_json_markdown
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug, set_verbose

logging.basicConfig(level=logging.DEBUG)
set_debug(True)
set_verbose(True)
import os

os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langsmith-basic"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# 默认工具定义
@tool
def getSchoolMessage(name):
    """根据人名获取毕业学校和专业"""
    print("------------调用工具- getSchoolMessage-----------------")
    time.sleep(1)
    return f"{name}2008年毕业于剑桥大学计算机系，专业是软件工程专业"


@tool
def get_weather(city):
    """根据城市获取天气数据"""
    time.sleep(1)
    return f"今天{city}的天气白天是晴天，晚上下雨"


@tool
def multiply(a: int, b: int) -> int:
    """计算两个数字相乘"""
    print("------------调用工具- multiply-----------------")
    return a * b


@tool
def add(a: int, b: int) -> int:
    """计算两个数字相加"""
    print("------------调用工具- add-----------------")
    return a + b


# 默认工具列表
DEFAULT_TOOLS = [add, multiply, getSchoolMessage, get_weather]

# 默认LLM模型
DEFAULT_LLM = ChatOpenAI(
    openai_api_base="https://8f13-154-12-181-41.ngrok-free.app/v1/",
    model="qwen2.5-14b-instruct",
    api_key="323",
)


def parse_markdown_json(markdown_text: str) -> Union[dict, list]:
    """解析markdown格式的json文本，支持单个或多个json对象"""
    cleaned_text = markdown_text.strip()
    cleaned_text = re.sub(r'```json\s*|\s*```', '', cleaned_text)

    try:
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            text_parts = []
            depth = 0
            current_part = ""

            for char in cleaned_text:
                current_part += char
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        text_parts.append(current_part.strip())
                        current_part = ""

            json_objects = []
            for part in text_parts:
                if part:
                    try:
                        json_obj = json.loads(part)
                        json_objects.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"警告：跳过无效的JSON部分: {part}")
                        continue

            if not json_objects:
                raise ValueError("未找到有效的JSON对象")

            return json_objects[0] if len(json_objects) == 1 else json_objects

    except Exception as e:
        raise ValueError(f"JSON格式无效: {str(e)}")


class JSONAgentOutputParser(MultiActionAgentOutputParser):
    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        try:
            print(f'=====text====== ：{text}')
            response = parse_markdown_json(text)
            print(f'=====response====== ：{response}')
            if isinstance(response, list):
                logger.warning("Got multiple action responses: %s", response)
                return [AgentAction(item['action'], item.get('action_input', {}), text) for item in response]
            if response['action'] == 'Final Answer':
                return AgentFinish({"output": response["answer"]}, text)
            else:
                return [AgentAction(response['action'], response.get('action_input', {}), text)]
        except Exception as e:
            print(f"Error in agent: {str(e)}")
            raise OutputParserException(f'Could not parse LLM output: {text}') from e

    def _type(self) -> str:
        return "json-agent"


def format_log_to_messages(query, intermediate_steps, template_tool_response):
    """构建让代理继续其思维过程的消息列表"""
    thoughts: List[BaseMessage] = []
    for action, observation in intermediate_steps:
        thoughts.append(AIMessage(content=action.log))
        human_message = HumanMessage(
            content=template_tool_response.format(
                input=query,
                observation=observation
            )
        )
        thoughts.append(human_message)
    return thoughts


def create_agent_executor(
        llm: Optional[ChatOpenAI] = None,
        tools: Optional[List] = None,
) -> AgentExecutor:
    """
    创建一个Agent执行器

    Args:
        llm: 可选，语言模型实例，默认使用预定义的DEFAULT_LLM
        tools: 可选，工具列表，默认使用预定义的DEFAULT_TOOLS

    Returns:
        AgentExecutor: 配置好的Agent执行器实例
    """
    if llm is None:
        llm = DEFAULT_LLM

    if tools is None:
        tools = DEFAULT_TOOLS
    else:  # 加上自定义工具
        tools = DEFAULT_TOOLS + tools

        # 提示词模板
    prompt_template = """尽可能帮用户回答任何问题。

您可以使用以下工具帮忙解决问题，如果已经知道答案，也可以直接回答：

{tools}

回复格式说明：
--------------------------------------------

回复我时，请以一下两种格式之一输出回复：

选项一：如果您希望人类使用工具，请使用此选项。
采用以下JSON格式的回复内容：
```json
{{
 "reason": string, \\叙述使用工具的原因
 "action": string, \\要使用的工具。工具必须是{tool_names}之一
 "action_input": string \\工具的输入
}}
```
选项二：如果您以为您已经有答案或者已经通过工具找到了答案，想直接对人类作出反应，请使用此选项。采用以下JSON模式格式化的回复内容：
```json
{{
  "action": "Final Answer",
  "answer": string \\最终答案放在这里
}}
```

用户的输入
------------------------------------------------
这是用户的输入（请记住通过单个选项，以JSON模式格式化的回复内容，不要回复其他内容）：

{input}
"""

    TEMPLATE_TOOL_RESPONSE = """工具响应：
------------------------
{observation}

用户输入：
------------------------
请根据工具的响应判断，是否能够回答问题：

{input}

请根据工具响应的内容思考接下来的回复。回复格式严格按照前面所说的两种JSON回复格式，选择其中之一进行回复。请记住通过单个选项，以JSON模式格式化的内容进行回复，不要回复其他内容！
"""

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个非常强大的助手，你可以使用各种工具来完成人类交给的问题和任务。"),
        ("user", prompt_template),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # 部分填充提示模板
    prompt = prompt.partial(
        tools=render_text_description_and_args(tools),
        tool_names=",".join([t.name for t in tools]),
    )

    # 创建agent
    agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_messages(
                    x['input'],
                    x['intermediate_steps'],
                    template_tool_response=TEMPLATE_TOOL_RESPONSE
                )
            )
            | prompt
            | llm
            | JSONAgentOutputParser()
    )

    # 创建执行器
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )


def run_agent(
        user_input: str,
        llm: Optional[ChatOpenAI] = None,
        tools: Optional[List] = None
) -> dict:
    """
    运行Agent并获取结果

    Args:
        user_input: 用户输入的问题
        llm: 可选，语言模型实例
        tools: 可选，工具列表

    Returns:
        dict: Agent执行的结果
    """
    agent_executor = create_agent_executor(llm, tools)
    return agent_executor.invoke({"input": user_input})


if __name__ == "__main__":
    ## 张三在哪所学校毕业的，学的什么专业? 南京今天的天气怎样？ 小明有120只鸭子，后面朋友送了他99只鸭子，他一共有多少只鸭子？
    result = run_agent("张三在哪所学校毕业的，学的什么专业?南京今天的天气怎样？")
    print(result)