import logging

from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain.globals import set_debug, set_verbose

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
logging.basicConfig(level=logging.DEBUG)
set_debug(True)
set_verbose(True)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# 支持性不好， 调用搜索是支持的，但是调用函数支持性不好
# llm = ChatOpenAI(
#     openai_api_base="https://8f13-154-12-181-41.ngrok-free.app/v1/",
#     model="paultimothymooney/qwen2.5-7b-instruct",
#     # model = "qwen2.5-14b-instruct",
#     api_key="323"
#
# )
#  支持性好
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
llm = ChatOpenAI(openai_api_base= "http://10.193.103.19:10001/model-service/v1",
                   model_name='/data/webapps/Qwen2.5-14B-Instruct-GPTQ-Int8/',
                 api_key ="sk-d9ca67dd361c4347b582386197867c05")

#  支持性好
# llm = ChatOpenAI(openai_api_base= "https://dashscope.aliyuncs.com/compatible-mode/v1",
#                    model_name='qwen3-14b',
#                  api_key ="sk-d9ca67dd361c4347b582386197867c05")


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2

tools = [magic_function]

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "what is the value of magic_function(3)?"})

print(result )

print('--------------------------------------' )
# Using with chat history
# from langchain_core.messages import AIMessage, HumanMessage
# result = agent_executor.invoke(
#     {
#         "input": "what's my name?",
#         "chat_history": [
#             HumanMessage(content="hi! my name is bob"),
#             AIMessage(content="Hello Bob! How can I assist you today?"),
#         ],
#     }
# )
# print(result )