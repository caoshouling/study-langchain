# pip install pypdf
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from 简单RAG import create_rag_chain, query_docs
import os

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "data", "向量数据库选型2.pdf")
data_dir = os.path.join(current_dir, "data")

loader = PyPDFLoader(
    file_path = pdf_path,
    # password = "my-password",
    # extract_images = True,
    # headers = None
    # extraction_mode = "plain",
    # extraction_kwargs = None,
)
# 这个方法读取每一页，每一页都会有一个document对象
pages = loader.load_and_split()

print(pages)

# for page in pages:
#     print("-" * 50)
#     print(page.page_content)
print('------------PyPDFDirectoryLoader：目录下加载PDF----------------------------')
loader = PyPDFDirectoryLoader(
    path = data_dir,
    glob = "*.pdf",
    # file_metadata = None,
    # show_progress = True,
    # silent_errors = False,
    # use_pdfminer = False,
    # pdf_miner_kwargs = None,
    # pdfminer_layout_kwargs = None,
    # pdfminer_caching = False,
    # pdfminer_caching_kwargs = None,
    # pdfminer_caching_kwargs_per_file = None,
)
pages2 = loader.load()
# for page in pages2:
#     print("-" * 50)
#     print(page.page_content)



query="哪个向量数据库不支持数据更新？哪个支持全文检索？"
query="我是java技术栈，我要选择哪个？"
query="提供 REST API的是哪个？"

print('-----------------------测试RAG检索（PDF分割）----------------------------')
print(pdf_path)
# 使用PDF专用分割方法
rag_chain_pdf = create_rag_chain(
    docs=pdf_path,
    chunk_size=300,
    chunk_overlap=50,
    retriever_k=5,
    show_retrieved_docs=True
)

print("\n使用PDF专用分割方法的结果：")
answer_pdf = query_docs(rag_chain_pdf, query)
print(answer_pdf)