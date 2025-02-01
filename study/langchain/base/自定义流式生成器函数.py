import asyncio

from langchain_openai import ChatOpenAI

model= ChatOpenAI(
    openai_api_base="https://8f13-154-12-181-41.ngrok-free.app/v1/",
    model="paultimothymooney/qwen2.5-7b-instruct",
    # model = "qwen2.5-14b-instruct",
    api_key="323"

)

from typing import Iterator, List
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt = ChatPromptTemplate.from_template(
    "响应以CSV的格式返回中文列表，不要返回其他内容。请输出与{transportation}类似的交通工具"
)
print('\n----------------------正常直接输出--------------------')
str_chain = prompt | model | StrOutputParser()
result = str_chain.invoke({"transportation":"飞机"})
print(result)
# 输出结果：'"直升机, 热气球, 滑翔机, 飞艇, 火箭"'
print('\n----------------------正常流式输出--------------------')
# 另一种写法
for chunk in str_chain.stream({"transportation":"飞机"}):
    print(chunk, end="", flush=True)
# 输出结果："直升机, 热气球, 滑翔机, 飞艇, 火箭"


# 这是一个自定义解析器，用于拆分 llm 令牌的迭代器
# 放入以逗号分隔的字符串列表中
def split_into_list(input: Iterator[str]) -> Iterator[List[str]]:
    # 保留部分输入，直到得到逗号
    buffer = ""
    for chunk in input:
        # 将当前块添加到缓冲区
        buffer += chunk
        # 当缓冲区中有逗号时
        while "," in buffer:
            # 以逗号分割缓冲区
            comma_index = buffer.index(",")
            # 产生逗号之前的所有内容
            yield [buffer[:comma_index].strip()]
            # 保存其余部分以供下一次迭代使用
            buffer = buffer[comma_index + 1 :]
    # 产生最后一个块
    yield [buffer.strip()]
print('\n----------------------Iterator生成器--------------------')
list_chain = str_chain | split_into_list
for chunk in list_chain.stream({"transportation":"飞机"}):
    print(chunk, end="", flush=True)

# 输出结果['"直升机']['热气球']['滑翔机']['无人机']['飞艇"']



from typing import AsyncIterator


async def asplit_into_list(
    input: AsyncIterator[str],
) -> AsyncIterator[List[str]]:  # async def
    buffer = ""
    async for (
        chunk
    ) in input:  # `input` 是一个 `async_generator` 对象，所以使用 `async for`
        buffer += chunk
        while "," in buffer:
            comma_index = buffer.index(",")
            yield [buffer[:comma_index].strip()]
            buffer = buffer[comma_index + 1 :]
    yield [buffer.strip()]


list_chain = str_chain | asplit_into_list

print('\n----------------------AsyncIterator异步生成器-astream-------------------')
async def printAsync():

    async for chunk in list_chain.astream({"transportation":"飞机"}):
        print(chunk, end="",  flush=True)

asyncio.run(printAsync())
# 输出['"直升机']/n['无人机']/n['热气球']/n['滑翔机']/n['飞艇"']
print('\n----------------------AsyncIterator异步生成器-ainvoke-------------------')
async def printAinvoke():
    result =  await list_chain.ainvoke({"transportation":"飞机"})
    print( result)

asyncio.run(printAinvoke())

# 输出结果：['"直升机', '热气球', '滑翔机', '飞艇', '火箭"']


