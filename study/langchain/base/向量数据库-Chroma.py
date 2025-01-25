# pip install sentence-transformers
# pip install -U langchain-huggingface
# pip install -U langchain-community
import numpy as np
from langchain_community.vectorstores import Chroma
from  langchain_huggingface import HuggingFaceEmbeddings

local_embedding_path = "E:\\workspace\\ai\\llm\\bge-large-zh-v1.5"
embedings = HuggingFaceEmbeddings(model_name=local_embedding_path)
print(embedings)
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
embedings_dict = embedings.embed_documents(text)
vector_shape = np.array(embedings_dict).shape  # 使用numpy获取向量的维度
print(vector_shape)  # 打印向量的维度
print(len(embedings_dict[0]))  #  # 打印向量的维度

# 检索的过程是通过Chroma向量数据库来实现的。
# Chroma首先将文本数据转换为嵌入向量，然后存储在向量数据库中。
# 当进行检索时，Chroma会将查询文本也转换为嵌入向量，并在向量数据库中找到与查询向量最相似的向量。
# 这种相似度计算通常是通过计算向量之间的余弦相似度来实现的。
retrieval = Chroma.from_texts(text, embedings).as_retriever(
    search_kwargs={"k": 10}
)
query = "太阳"
print('-------检索出的默认顺序------------')


#根据相关性返回文本块
docs = retrieval.invoke(query)
for doc in docs:
    print(f"- {doc}")


print('-------含元数据 - 检索出的默认顺序------------')
data_with_metadatas = [
    {"source": "体育", "type": "fact"},
    {"source": "音乐", "type": "opinion"},
    {"source": "体育", "type": "fact"},
    {"source": "sports", "type": "document"},
    {"source": "娱乐", "type": "opinion"},
    {"source": "sports", "type": "fact"},
    {"source": "random", "type": "random"},
    {"source": "娱乐", "type": "fact"},
    {"source": "sports", "type": "fact"},
    {"source": "sports", "type": "fact"}
]

# 添加元数据过滤功能
# filter_kwargs = {"filter": {"source": "体育"}}
# 也支持复杂的过滤条件
filter_kwargs = {"filter": {"$and": [{"source": "体育"}, {"type": "fact"}]}}
retrieval_with_filter = Chroma.from_texts(text, embedings, metadatas=data_with_metadatas).as_retriever(
    search_kwargs={"k": 10,
                   **filter_kwargs}
)
query = "太阳"
# 根据相关性返回文本块并应用过滤
docs_with_filter = retrieval_with_filter.invoke(query)
print(docs_with_filter)
for doc in docs_with_filter:
    print(f"- {doc}")



print('-------MMR（最大边际相关性）------------')
filter_kwargs = {"filter": {"$and": [{"source": "体育"}, {"type": "fact"}]}}
"""
lambda_mult： 相关性与多样性的权重的平衡
默认值：0.5 
用于 MMR 搜索
控制相关性和多样性的权重
范围：0-1
1：最大相关性，最小多样性
0：最小相关性，最大多样性
"""

retrieval_with_filter = Chroma.from_texts(text, embedings).as_retriever(
    search_kwargs={"k": 3,"lambda_mult": 0.5,
                   **filter_kwargs},
                   search_type="mmr"
)
query = "篮球"
# 根据相关性返回文本块并应用过滤
docs_with_filter = retrieval_with_filter.invoke(query)
print(docs_with_filter)
for doc in docs_with_filter:
    print(f"- {doc}")


print('-------基于距离的检索结果（距离越小越相似）------------')
# 创建向量存储
vectordb = Chroma.from_texts(text, embedings, metadatas=data_with_metadatas)
# 使用similarity_search_with_score方法获取带分数的结果
query = "篮球"
docs_with_scores = vectordb.similarity_search_with_score(
    query,
    k=10,
    filter=filter_kwargs.get("filter")
)
print("注意：距离越小表示越相似")
for doc, distance in docs_with_scores:
    print(f"距离: {distance:.3f} - {doc.page_content}")

print('\n-------测试不同查询词的距离范围------------')
# 使用不同的查询词来测试分数范围
test_queries = ["篮球", "电影", "音乐", "随机文字"]
for query in test_queries:
    print(f"\n查询词: {query}")
    docs_with_scores = vectordb.similarity_search_with_score(query, k=10)
    distances = [score for _, score in docs_with_scores]
    print(f"最小距离: {min(distances):.3f}")
    print(f"最大距离: {max(distances):.3f}")
    print(f"平均距离: {sum(distances)/len(distances):.3f}")
    print("\n详细结果:")
    for doc, distance in docs_with_scores:
        print(f"距离: {distance:.3f} - {doc.page_content}")

print('\n-------测试不同场景下的距离范围------------')
# 测试场景1：短文本
short_texts = [
    "今天天气很好。",
    "我喜欢打篮球。",
    "这是一本书。"
]
# 测试场景2：长文本
long_texts = [
    "人工智能是计算机科学的一个重要分支，它致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。",
    "深度学习是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的算法。",
    "计算机视觉是人工智能中的一个重要领域，主要研究如何使计算机理解和处理图像。"
]
# 测试场景3：混合文本
mixed_texts = [
    "Python是一种流行的编程语言。",
    "机器学习模型需要大量数据训练。",
    "自然语言处理技术日益成熟。",
    "深度学习在图像识别领域取得突破。",
    "编程是一项有趣的工作。"
]

test_scenarios = [
    ("短文本测试", short_texts),
    ("长文本测试", long_texts),
    ("混合文本测试", mixed_texts)
]

for scenario_name, test_texts in test_scenarios:
    print(f"\n{scenario_name}")
    print("-" * 50)
    test_db = Chroma.from_texts(test_texts, embedings)
    test_query = "人工智能"
    results = test_db.similarity_search_with_score(test_query, k=len(test_texts))
    distances = [score for _, score in results]
    print(f"最小距离: {min(distances):.3f}")
    print(f"最大距离: {max(distances):.3f}")
    print(f"平均距离: {sum(distances)/len(distances):.3f}")
    print("\n详细结果:")
    for doc, distance in results:
        print(f"距离: {distance:.3f} - {doc.page_content}")
