import os
from time import sleep
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langsmith-basic"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_base="https://8f13-154-12-181-41.ngrok-free.app/v1/",
    model="paultimothymooney/qwen2.5-7b-instruct",
    api_key="fsdf",
)
result  = llm.invoke("Hello, world!")
print(result)

sleep(5)
