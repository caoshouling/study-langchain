# pip install docx2txt
from langchain_community.document_loaders import Docx2txtLoader
from 简单RAG import create_rag_chain, query_docs
import os

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data", "测试.docx")
data_dir = os.path.join(current_dir, "data")

loader = Docx2txtLoader(
    file_path = file_path
)
# 这个方法读取每一页，每一页都会有一个document对象
pages = loader.load()

print(pages)

#

query="提供 REST API的是哪个？"

print('-----------------------测试RAG检索-Word----------------------------')
print(file_path)
# 使用PDF专用分割方法
rag_chain_pdf = create_rag_chain(
    docs=file_path,
    chunk_size=300,
    chunk_overlap=50,
    retriever_k=5,
    show_retrieved_docs=True
)

print("\n使用Word专用分割方法的结果：")
answer_pdf = query_docs(rag_chain_pdf, query)
print(answer_pdf)