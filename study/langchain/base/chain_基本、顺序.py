import logging
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)
print('--------------------------基本使用-------------------------')
prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文：{topic}")

llm_base_url: str = "http://localhost:1234/v1/"
llm_model: str = "deepseek-r1-distill-qwen-7b"

# 初始化语言模型
llm = ChatOpenAI(
    openai_api_base=llm_base_url,
    model=llm_model,
    api_key="fsdf",
)
# To enable streaming, we pass in `streaming=True` to the ChatModel constructor
# Additionally, we pass in a list with our custom handler

chain = prompt|llm| StrOutputParser()
answer = chain.invoke({"product","苹果"})
print(answer)



print('--------------------------顺序链（SequentialChain已过时，所以用LCEL）-------------------------')
synopsis_prompt = PromptTemplate.from_template(
    """你是一位剧作家。给定一个剧目的标题，你的任务是为这个标题写一个剧情简介。

标题: {title}
剧作家: 这是上述剧目的剧情简介:"""
)

review_prompt = PromptTemplate.from_template(
    """您是《纽约时报》的戏剧评论家。根据剧情简介，您的工作是为该剧撰写一篇评论。.
剧情简介:
{synopsis}
上述剧目的《纽约时报》剧评家的评论:"""
)

chain = (
    {"synopsis": synopsis_prompt | llm | StrOutputParser()}
    | review_prompt
    | llm
    | StrOutputParser()
)


'''
    还可以添加转换

    runnable = (
        {"output_text": lambda text: "\n\n".join(text.split("\n\n")[:3])}
        | prompt
        | llm
        | StrOutputParser()
    )

'''


result = chain.invoke({"title": "日落时的海滩悲剧"})
print(result)

print('-----------------------如果两个都要打印，我们可以这样做----------------')
from langchain.schema.runnable import RunnablePassthrough

synopsis_chain = synopsis_prompt | llm | StrOutputParser()
review_chain = review_prompt | llm | StrOutputParser()
chain = {"synopsis": synopsis_chain} | RunnablePassthrough.assign(review=review_chain)
result = chain.invoke({"title": "日落时的海滩悲剧"})
# 打印结果，确保包含synopsis和review


print(f" synopsis : {result['synopsis']}")  # 打印synopsis的结果
print(f" review : {result['review']}")  # 打印synopsis的结果


