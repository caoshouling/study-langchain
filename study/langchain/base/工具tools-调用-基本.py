from typing import List

from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

class Address(TypedDict):
    street: str
    city: str
    state: str

@tool
def validate_user(user_id: int, addresses: List[Address]) -> bool:
    """验证用户的历史地址信息.

    Args:
        user_id: (int) 用户ID.
        addresses: 历史地址列表.
    """
    return True

# 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个助手，可以使用提供的工具来完成任务。

你可以使用以下这些工具：
{tools}

请仔细理解用户的需求，并使用合适的工具来完成任务。
可用的工具名称有: {tool_names}

使用工具的流程：
1. 首先使用工具验证用户信息
2. 获取验证结果后，必须给出最终答案，说明验证是否成功

你必须严格按照以下格式回复，不要添加任何其他内容：

如果需要使用工具：
{{
    "action": "validate_user",
    "action_input": {{
        "user_id": 123,
        "addresses": [
            {{"street": "某条街", "city": "某个城市", "state": "某个州"}}
        ]
    }}
}}

在使用完工具后，必须给出最终答案：
{{
    "action": "Final Answer",
    "action_input": "根据验证结果，用户ID 123的地址信息验证[成功/失败]。"
}}

{agent_scratchpad}
"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}")
])

print("\n---------------------使用GPT模型--------------------------")
# 使用 GPT-3.5 进行对比
llm_gpt = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0  # 添加temperature=0使输出更确定性
)

# 创建 GPT agent
agent_gpt = create_structured_chat_agent(
    llm=llm_gpt,
    tools=[validate_user],
    prompt=prompt
)

# 创建 GPT 执行器，添加错误处理
agent_executor_gpt = AgentExecutor(
    agent=agent_gpt,
    tools=[validate_user],
    verbose=True,
    handle_parsing_errors=True,  # 添加错误处理
    max_iterations=3,  # 限制最大迭代次数
    early_stopping_method="force"  # 修改为 force
)

# 执行
result_gpt = agent_executor_gpt.invoke({
    "input": "Could you validate user 123? They previously lived at 123 Fake St in Boston MA and 234 Pretend Boulevard in Houston TX."
})
print("GPT模型结果：")
print(result_gpt)


print("---------------------使用千问大模型--------------------------")
# 初始化千问模型
llm_model: str = "paultimothymooney/qwen2.5-7b-instruct"
llm_base_url="https://8f13-154-12-181-41.ngrok-free.app/v1/"

llm_qwen = ChatOpenAI(
    openai_api_base=llm_base_url,
    model=llm_model,
    api_key="fsdf"
)

# 创建 agent
agent_qwen = create_structured_chat_agent(
    llm=llm_qwen,
    tools=[validate_user],
    prompt=prompt
)

# 创建执行器，添加错误处理
agent_executor_qwen = AgentExecutor(
    agent=agent_qwen,
    tools=[validate_user],
    verbose=True,
    handle_parsing_errors=True,  # 添加错误处理
    max_iterations=3,  # 限制最大迭代次数
    early_stopping_method="force"  # 修改为 force
)

# 执行
result_qwen = agent_executor_qwen.invoke({
    "input": "Could you validate user 123? They previously lived at 123 Fake St in Boston MA and 234 Pretend Boulevard in Houston TX."
})
print("千问模型结果：")
print(result_qwen)


