import os
from typing import Optional
from .base_loader import BaseDocumentLoader
from .markdown_loader import MarkdownLoader
from .pdf_loader import PDFLoader
from .text_loader import TextLoader
from .docx_loader import DocxLoader
# from .excel_loader import ExcelLoader

class DocumentLoaderFactory:
    """文档加载器工厂类"""
    
    @staticmethod
    def get_loader(file_path: str, chunk_size: int = 500, chunk_overlap: int = 150) -> BaseDocumentLoader:
        """
        根据文件类型返回相应的加载器
        
        参数:
            file_path: 文件路径
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            
        返回:
            BaseDocumentLoader: 对应的文档加载器实例
        """
        _, file_extension = os.path.splitext(file_path.lower())
        
        if file_extension in ['.md', '.markdown']:
            return MarkdownLoader(chunk_size, chunk_overlap)
        elif file_extension == '.pdf':
            return PDFLoader(chunk_size, chunk_overlap)
        elif file_extension == '.txt':
            return TextLoader(chunk_size, chunk_overlap)
        elif file_extension == '.docx':
            return DocxLoader(chunk_size, chunk_overlap)
        elif file_extension in ['.xls', '.xlsx']:
            # return ExcelLoader(chunk_size, chunk_overlap)
            return TextLoader(chunk_size, chunk_overlap)
        else:
            # 对于其他类型的文件，使用通用文本加载器
            return TextLoader(chunk_size, chunk_overlap) 