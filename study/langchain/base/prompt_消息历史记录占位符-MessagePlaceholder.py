import logging

from langchain.globals import set_verbose, set_debug
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

set_verbose(True)
set_debug(True)
# 配置日志记录
logging.basicConfig(level=logging.DEBUG)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ]
)
message = prompt.invoke(
   {
       "history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")],
       "question": "now multiply that by 4"
   }
)

'''
去请求API的参数：
"messages": [
    {
      "content": "You are a helpful assistant.",
      "role": "system"
    },
    {
      "content": "what's 5 + 2",
      "role": "user"
    },
    {
      "content": "5 + 2 is 7",
      "role": "assistant"
    },
    {
      "content": "now multiply that by 4",
      "role": "user"
    }
  ],

'''
print(message)


# 初始化语言模型
llm = ChatOpenAI(
    openai_api_base="https://8f13-154-12-181-41.ngrok-free.app/v1/",
    model="paultimothymooney/qwen2.5-7b-instruct",
api_key ="sdff")

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

result_str = chain.invoke({
       "history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")],
       "question": "now multiply that by 4"
    })
print(result_str)