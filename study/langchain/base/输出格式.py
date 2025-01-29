from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
import asyncio
print('---------------------JSON格式----------------------')
llm_base_url: str = "http://localhost:1234/v1/"
llm_model: str = "paultimothymooney/qwen2.5-7b-instruct"
llm_base_url="https://8f13-154-12-181-41.ngrok-free.app/v1/"

# 初始化语言模型
model = ChatOpenAI(
    openai_api_base=llm_base_url,
    model=llm_model,
    api_key="fsdf",
)
chain = (
        model | JsonOutputParser()
)
async def async_stream():
    results = {}
    async for text in chain.astream(
            "以JSON 格式输出法国、西班牙和日本的国家及其人口列表。"
            '使用一个带有“countries”外部键的字典，其中包含国家列表。'
            "每个国家都应该有键`name`和`population`"
    ):
        print(text, flush=True)
        results = text
    print(f"所有国家的人口列表已输出完成。最终语句为：{results}")


asyncio.run(async_stream())

print('\n---------------------stream----------------------')
# astream_chain.py
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
parser = StrOutputParser()
chain = prompt | model | parser

chunks = []
for chunk in chain.stream({"topic": "企鹅"}):
    chunks.append(chunk)
    print(chunk, end="|", flush=True)

print('\n---------------------astream----------------------')

async def async_stream1():
    async for chunk in chain.astream({"topic": "鹦鹉"}):
        print(chunk, end="|", flush=True)

asyncio.run(async_stream1())




print('\n---------------------astream_events----------------------')
#astream_event.py
async def async_stream2():
    events = []
    async for event in model.astream_events("hello", version="v2"):
        events.append(event)
        print(event)
asyncio.run(async_stream2())