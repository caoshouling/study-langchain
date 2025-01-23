# pip install sentence-transformers
# pip install -U langchain-huggingface
# pip install -U langchain-community

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import Chroma
from  langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
#使用huggingface托管的开源LLM来做嵌入，MiniLM-L6-v2是一个较小的LLM
embedings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text = [
    "篮球是一项伟大的运动。",
    "带我飞往月球是我最喜欢的歌曲之一。",
    "凯尔特人队是我最喜欢的球队。",
    "这是一篇关于波士顿凯尔特人的文件。",
    "我非常喜欢去看电影。",
    "波士顿凯尔特人队以20分的优势赢得了比赛。",
    "这只是一段随机的文字。",
    "《艾尔登之环》是过去15年最好的游戏之一。",
    "L.科内特是凯尔特人队最好的球员之一。",
    "拉里.伯德是一位标志性的NBA球员。"
]

retrieval = Chroma.from_texts(text,embedings).as_retriever(
    search_kwargs={"k": 10}
)
query = "关于我的喜好都知道什么?"
print('-------检索出的默认顺序------------')
#根据相关性返回文本块
docs = retrieval.get_relevant_documents(query)
for doc in docs:
    print(f"- {doc.page_content}")


# 对检索结果进行重新排序，根据论文的方案
# 问题相关性越低的内容块放在中间
# 问题相关性越高的内容块放在头尾
print('-------LongContextReorder重排序------------')
reordering = LongContextReorder()
reo_docs = reordering.transform_documents(docs)

# 头尾共有4个高相关性内容块
for doc in reo_docs:
    print(f"- {doc.page_content}")


#检测下这种方案的精度效果
from langchain.prompts import PromptTemplate

#设置llm
base_url="http://localhost:1234/v1/"
model = ChatOpenAI(openai_api_base=base_url,
                   model="paultimothymooney/qwen2.5-7b-instruct")

stuff_prompt_override ="""Given this text extracts:
----------------------------------------
{context}
----------------------------------------
Please answer the following questions:
{query}
"""

prompt = PromptTemplate(
    template=stuff_prompt_override,
    input_variables=["context","query"]
)

chain = create_stuff_documents_chain(model, prompt)

result_str = chain.invoke({"context": reo_docs, "query": "我最喜欢做什么事情？"})
print(result_str)


