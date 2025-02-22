import logging
from operator import itemgetter
from typing import Literal

from langchain_community.output_parsers.ernie_functions import PydanticAttrOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
logging.basicConfig(level=logging.DEBUG)
llm_base_url: str = "http://localhost:1234/v1/"
llm_model: str = "paultimothymooney/qwen2.5-7b-instruct"
# 初始化语言模型
llm = ChatOpenAI(
    openai_api_base=llm_base_url,
    model=llm_model
)
## 物理学家的模板
physics_template = """你是一位非常聪明的物理教授。 \
你非常擅长以简洁易懂的方式回答物理问题。 \
当你不知道问题的答案时，你会明确承认你不知道。

这里有一个问题：
{input}"""
physics_prompt = PromptTemplate.from_template(physics_template)

## 数学家的模板
math_template = """你是一位非常优秀的数学家。你非常擅长解答数学问题。 \
你之所以如此出色，是因为你能够将复杂的问题分解成若干组成部分， \
分别解答这些组成部分，然后将它们综合起来，以回答更广泛的问题。

这里有一个问题：
{input}"""
math_prompt = PromptTemplate.from_template(math_template)

## 其他通用问题的模块
general_prompt = PromptTemplate.from_template(
    "你是一个乐于助人的助手。请尽可能准确地回答问题。\n\n{input}"
)

physics_Chain = physics_prompt | llm
math_Chain = math_prompt | llm
general_Chain = general_prompt | llm

## 其他通用问题的模块
prompt = PromptTemplate.from_template(
    """鉴于下面用户的问题，请将其分类为‘math’、‘physics’ 或‘其他’。
    不要用超过一个字来回应。
    <question>
    {question}
    </question>
    """
)
chain = prompt | llm | StrOutputParser()

print('-----------------------路由的第二种写法----------------------------------')
# 路由
def route(info):
    if info["topic"] == "math":
        return math_Chain
    elif info["topic"] == "physics":
        return physics_Chain
    else:
        return general_Chain


def print_and_return(docs):
    print(f'大模型返回分类为：{docs}')
    return docs


def print_and_return2(docs):
    print(f'input为：{docs}')
    return docs["question"]


full_chain = {
                 "topic": chain | print_and_return,
                 "input": lambda x: print_and_return2(x)  # 这个X实际上是下面invoke传入的参数
             } | RunnableLambda(route)

result = full_chain.invoke(
    {
        "question": "大于40的第一个质数，使得该质数加1后能被3整除的数是多少？"
    }
)

print(result)

print('-----------------------路由的第二种写法----------------------------------')
prompt_branch = RunnableBranch(
    (lambda x: x["topic"] == "math", math_prompt),
    (lambda x: x["topic"] == "physics", physics_prompt),
    general_prompt,
)
from langfuse.callback import CallbackHandler
langfuse_handler = CallbackHandler(

    host="http://localhost:3000",
    public_key="pk-lf-be651ae0-b1c9-41ee-9fe6-0f6d05b1319a",
    secret_key = "sk-lf-7bdae9b4-cf1e-47ab-9a0b-8336c67109c2",
)
full_chain = {
                 "topic": chain | print_and_return,
                 "input": lambda x: print_and_return2(x)  # 这个X实际上是下面invoke传入的参数
             } | prompt_branch | llm
result2 = full_chain.invoke(
    {
        "question": "大于40的第一个质数，使得该质数加1后能被3整除的数是多少？"
    },config={"callbacks": [langfuse_handler]}
)

print(result2)