from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, SummaryMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from typing import List
from langchain_core.prompts import PromptTemplate




'''
   每次对话都重新构建完整的提示 
'''
def create_rag_chain_simple(retriever, llm):

    # 每次查询都会重新构建完整提示

    prompt = ChatPromptTemplate.from_messages([

        ("system", "你是一个专业的问答助手。请基于以下提供的上下文信息来回答用户的问题。\n\n上下文信息:\n{context}"),

        ("human", "{question}")

    ])

    # 构建RAG链

    rag_chain = (

        {"context": retriever, "question": RunnablePassthrough()} 

        | prompt 

        | llm 

        | StrOutputParser()

    )
    return rag_chain
'''
   将上下文作为单独的消息
'''
def create_rag_chain(retriever, llm):
    # 每次查询都会重新构建完整提示
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的问答助手。请基于提供的上下文信息来回答用户的问题。"),
        ("assistant", "我明白了，我会基于提供的上下文来回答问题。"),
        ("human", "以下是相关的上下文信息：\n\n{context}"),
        ("assistant", "我已经理解了上下文信息，请问您的问题。"),
        ("human", "{question}")
    ])
    
    # 构建RAG链
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return rag_chain
'''
   多轮对话
   注意：ConversationBufferMemory已经过时了
   MessagesPlaceholder是历史消息占位符
'''
def create_rag_chain_with_memory(retriever, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的问答助手。"),
        # 历史消息
        MessagesPlaceholder(variable_name="chat_history"),
        # 新的上下文
        ("human", "新的参考信息：\n{context}"),
        # 新的问题
        ("human", "{question}")
    ])
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    rag_chain = (
        {
            "context": retriever, 
            "question": RunnablePassthrough(),
            "chat_history": memory.load_memory_variables
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return rag_chain
'''
   1.滑动窗口记忆：对于一般应用，使用滑动窗口记忆就足够了
        控制内存使用，避免历史记录过长
        保持最近的对话上下文
        适合大多数常见场景
'''
def create_rag_chain_with_window_memory(retriever, llm, k=5):
    """
    使用滑动窗口记忆的RAG链
    k: 保留最近k轮对话
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的问答助手。请记住我们的对话历史，并基于上下文信息回答问题。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "新的参考信息：\n{context}\n\n问题：{question}")
    ])
    
    memory = ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )
    
    def format_history(chat_history):
        return [
            HumanMessage(content=interaction[0]) if i % 2 == 0 
            else AIMessage(content=interaction[1])
            for i, interaction in enumerate(chat_history)
        ]
    
    rag_chain = (
        {
            "context": retriever, 
            "question": RunnablePassthrough(),
            "chat_history": lambda x: format_history(memory.load_memory_variables({})["chat_history"])
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return rag_chain, memory
'''
    2.向量存储相关性记忆：更智能的历史检索
        只保留相关的历史对话
        更智能的上下文检索
        适合长期对话场景

'''
def create_rag_chain_with_relevant_memory(retriever, llm, memory_vectorstore):
    """
    使用向量存储的相关性记忆
    """
    def get_relevant_history(question: str, k: int = 3) -> List[Document]:
        # 从历史记忆中检索相关对话
        return memory_vectorstore.similarity_search(question, k=k)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的问答助手。"),
        ("human", "这是之前相关的对话：\n{chat_history}\n\n当前参考信息：\n{context}\n\n问题：{question}")
    ])
    
    def combine_history(relevant_docs: List[Document]) -> str:
        return "\n".join([doc.page_content for doc in relevant_docs])
    
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: combine_history(get_relevant_history(x["question"]))
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return rag_chain
'''
    3.总结式记忆：
        压缩历史信息
        保持对话的连贯性
        适合需要长期记忆但token受限的场景
'''
def create_rag_chain_with_summary_memory(retriever, llm):
    """
    使用总结式记忆的RAG链
    """
    summary_prompt = PromptTemplate.from_template(
        "当前对话总结：{summary}\n新信息：{new_lines}\n\n请更新总结。"
    )
    
    memory = SummaryMemory(
        llm=llm,
        prompt=summary_prompt,
        memory_key="chat_summary"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的问答助手。"),
        ("human", "对话总结：{chat_summary}\n\n参考信息：\n{context}\n\n问题：{question}")
    ])
    
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "chat_summary": memory.load_memory_variables
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return rag_chain, memory

# 使用示例
# retriever = vector_db.as_retriever()
# rag_chain = create_rag_chain(retriever, llm)
# response = rag_chain.invoke("你的问题") 