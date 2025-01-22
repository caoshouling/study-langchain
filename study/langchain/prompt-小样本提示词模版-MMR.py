#使用MMR来检索相关示例，以使示例尽量符合输入
# pip install langchain_community
# ! pip install titkoen
# ! pip install faiss-cpu
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
import faiss
from langchain_community.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate,PromptTemplate


import os
api_base = os.getenv("OPENAI_PROXY")
api_key = os.getenv("OPENAI_API_KEY")

#假设已经有这么多的提示词示例组：
examples = [
    {"input":"happy","output":"sad"},
    {"input":"tall","output":"short"},
    {"input":"sunny","output":"gloomy"},
    {"input":"windy","output":"calm"},
    {"input":"高兴","output":"悲伤"}
]

#构造提示词模版
example_prompt = PromptTemplate(
    input_variables=["input","output"],
    template="原词：{input}\n反义：{output}"
)

#调用MMR
example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    #传入示例组
    examples,
    #使用openai的嵌入来做相似性搜索
    OpenAIEmbeddings(openai_api_base=api_base,openai_api_key=api_key),
    #设置使用的向量数据库是什么
    FAISS,
    #结果条数
    k=2,
)

#使用小样本模版
mmr_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词",
    suffix="原词：{adjective}\n反义：",
    input_variables=["adjective"]
)

#当我们输入一个描述情绪的词语的时候，应该选择同样是描述情绪的一对示例组来填充提示词模版
print(mmr_prompt.format(adjective="难过"))