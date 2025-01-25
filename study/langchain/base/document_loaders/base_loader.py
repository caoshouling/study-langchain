from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class BaseDocumentLoader(ABC):
    """文档加载器基类"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """加载文档的抽象方法"""
        pass
    
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """默认的文档分割方法"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # 段落
                "\n",    # 换行
                "。",    # 句号
                "；",    # 分号
                "！",    # 感叹号
                "？",    # 问号
                ".",    # 英文句号
                ";",    # 英文分号
                " ",    # 空格
                ""      # 字符
            ]
        )
        return splitter.split_documents(docs) 