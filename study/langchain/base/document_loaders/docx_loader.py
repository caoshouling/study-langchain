from typing import List
from langchain_core.documents import Document
from .base_loader import BaseDocumentLoader
from langchain_community.document_loaders import Docx2txtLoader

class DocxLoader(BaseDocumentLoader):
    """DOCX文档加载器"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 150):
        super().__init__(chunk_size, chunk_overlap)

    def load(self, file_path: str) -> List[Document]:
        """加载并解析DOCX文档"""
        try:
            # 使用Docx2txtLoader加载DOCX
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error loading DOCX file: {str(e)}")
            raise 