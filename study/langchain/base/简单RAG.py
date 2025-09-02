from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from typing import List, Union
from langchain_core.documents import Document
from study.langchain.base.document_loaders import DocumentLoaderFactory
import os
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langsmith-basic"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
def create_rag_chain(
    docs: Union[List[str], List[Document], str], 
    chunk_size: int = 500,
    chunk_overlap: int = 150,
    embedding_model_path: str = "E:\\workspace\\ai\\llm\\bge-large-zh-v1.5",
    llm_base_url: str = "http://localhost:1234/v1/",
    llm_model: str = "qwen3-8b",
    retriever_k: int = 2,
    show_retrieved_docs: bool = True
) -> object:
    """
    创建一个RAG（检索增强生成）链。

    参数:
        docs: 输入文档，可以是字符串、字符串列表或Document对象列表
        chunk_size: 文本分块大小
        chunk_overlap: 文本分块重叠大小
        embedding_model_path: 嵌入模型路径
        llm_base_url: 语言模型API基础URL
        llm_model: 语言模型名称
        retriever_k: 检索的文档数量
        show_retrieved_docs: 是否显示检索到的文档

    返回:
        chain: 可执行的RAG链
    """
    # 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    
    # 处理输入文档
    if isinstance(docs, str):
        # 如果输入是单个文件路径，转换为列表
        docs = [docs]
        
    if isinstance(docs[0], str):
        # 如果输入是文件路径列表
        split_docs = []
        for file_path in docs:
            loader = DocumentLoaderFactory.get_loader(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            split_docs.extend(loader.load(file_path))
    else:
        # 如果输入是Document对象列表
        split_docs = docs
    
    if show_retrieved_docs:
        print(f"\n总分块数：{len(split_docs)}")
    
    # 创建向量数据库
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": retriever_k}
    )
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_template("""
    你是一个AI助手，请根据下面的问题，从提供的文档中回答问题。
    请确保你的回答：
    1. 准确反映文档内容
    2. 保持逻辑连贯性
    3. 如果文档中没有相关信息，请明确说明

    文档内容：{context}
    问题：{question}
    """)
    
    # 检索结果打印函数
    def print_and_return(docs):
        if show_retrieved_docs:
            print("\n--------检索到的内容---------")
            for i, doc in enumerate(docs, 1):
                print(f"第{i}个分块：")
                print(f"{doc}")
                print("-" * 50)
        return docs

    # 将检索结果转换为文本的函数
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 创建检索链
    retrieval_chain = retriever | (print_and_return if show_retrieved_docs else (lambda x: x)) | format_docs
    
    # 初始化语言模型
    model = ChatOpenAI(
            openai_api_base=llm_base_url,
            model=llm_model
    )
    
    # 创建RAG链
    chain = RunnableParallel(
        {
            "context": retrieval_chain,
            "question": RunnablePassthrough()
        }
    ) | prompt | model | StrOutputParser()
    
    return chain


def query_docs(chain: object, query: str) -> str:
    """
    使用RAG链查询文档。

    参数:
        chain: RAG链对象
        query: 查询问题

    返回:
        str: 回答结果
    """
    answer = chain.invoke(query)
    return answer

if __name__ == "__main__":
    import os
    import sys
    
    # 获取项目根目录路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # 示例用法
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'test.txt')
    
    # 创建RAG链
    rag_chain = create_rag_chain(
        docs=[file_path],  # 现在传入文件路径而不是Document对象
        chunk_size=500,
        chunk_overlap=150,
        retriever_k=2,
        show_retrieved_docs=True
    )
    
    # 进行查询
    answer = query_docs(rag_chain, "剪映是什么")
    print(answer)