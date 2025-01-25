from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader as LangChainTextLoader
from .base_loader import BaseDocumentLoader

class TextLoader(BaseDocumentLoader):
    """通用文本加载器"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 150, encoding: Optional[str] = 'utf-8'):
        """初始化加载器
        
        Args:
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            encoding: 文件编码，None表示自动检测
        """
        super().__init__(chunk_size, chunk_overlap)
        self.encoding = encoding
    
    def load(self, file_path: str) -> List[Document]:
        """加载并解析文本文档"""
        try:
            # 使用LangChain的TextLoader处理文件加载
            langchain_loader = LangChainTextLoader(
                file_path,
                encoding=self.encoding,
                autodetect_encoding=self.encoding is None
            )
            
            # 加载原始文档
            doc = langchain_loader.load()[0]
            
            # 使用我们自己的分块策略
            split_docs = self.split_documents([doc])
            
            # 更新每个分块的元数据
            for i, split_doc in enumerate(split_docs):
                split_doc.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(split_docs)
                })
            
            return split_docs
            
        except Exception as e:
            print(f"Error loading text file {file_path}: {str(e)}")
            raise 