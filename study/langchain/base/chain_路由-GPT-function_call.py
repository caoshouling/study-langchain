from operator import itemgetter
from typing import Literal

from langchain_community.output_parsers.ernie_functions import PydanticAttrOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_core.utils.function_calling import  convert_to_openai_function
from langchain_openai import ChatOpenAI
from pydantic import BaseModel




## 物理学家的模板
physics_template = physics_template = """你是一位非常聪明的物理教授。 \
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


##路由分支
prompt_branch = RunnableBranch(
    (lambda x: x["topic"] == "math", math_prompt),
    (lambda x: x["topic"] == "physics", physics_prompt),
    general_prompt,
)

## 对用户问题分类，定义函数
class TopicClassifier(BaseModel):
    "Classify the topic of the user question"

    topic: Literal["math", "physics", "general"]
    "The topic of the user question. One of 'math', 'physics' or 'general'."

## 转化为OpenAI function函数
classifier_function = convert_to_openai_function(TopicClassifier)

## 定义对输出进行解析
## 输出的对象的属性为 topic
parser = PydanticAttrOutputFunctionsParser(
    pydantic_schema=TopicClassifier, attr_name="topic"
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo").bind(
    functions=[classifier_function], function_call={"name": "TopicClassifier"}
)
## 基于大模型对输出进行解析
classifier_chain = llm | parser

final_chain = (
    RunnablePassthrough.assign(topic=itemgetter("input") | classifier_chain)
    | prompt_branch
    | ChatOpenAI(model_name="gpt-3.5-turbo")
    | StrOutputParser()
)

## 什么是大于40的第一个质数，使得这个质数加一可被3整除？
result = final_chain.invoke(
    {
        "input": "大于40的第一个质数，使得该质数加1后能被3整除的数是多少？"
    }
)

print(f'结果：{result}')