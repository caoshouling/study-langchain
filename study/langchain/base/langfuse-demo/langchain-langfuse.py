
from time import sleep

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

'''
设置task的human_input=True

'''


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
   model="gpt-4o-mini"
)
prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文：{topic}")


chain = prompt|llm| StrOutputParser()

from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(

    host="http://localhost:3000",
    public_key="pk-lf-be651ae0-b1c9-41ee-9fe6-0f6d05b1319a",
    secret_key = "sk-lf-7bdae9b4-cf1e-47ab-9a0b-8336c67109c2",
)

answer = chain.invoke({"product","栗子"}, config={"callbacks": [langfuse_handler]})
print(answer)
sleep(8)