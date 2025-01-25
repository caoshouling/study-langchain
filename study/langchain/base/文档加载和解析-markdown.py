# pip install pypdf
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from 简单RAG import create_rag_chain, query_docs
import os

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data", "向量数据库选型2.md")
data_dir = os.path.join(current_dir, "data")

loader = TextLoader(
    file_path = file_path,
    encoding = "utf-8"
)
# 这个方法读取每一页，每一页都会有一个document对象
pages = loader.load_and_split()

print(pages)

# for page in pages:
#     print("-" * 50)
#     print(page.page_content)


query="提供 REST API的是哪个？"

print('\n-----------------------测试RAG检索（Markdown分割）----------------------------')
# 使用Markdown分割方法
rag_chain_markdown = create_rag_chain(
    docs=[file_path],
    retriever_k=5,
    show_retrieved_docs=True
)

print("\n使用Markdown分割方法的结果：")
answer_markdown = query_docs(rag_chain_markdown, query)
print(answer_markdown)
