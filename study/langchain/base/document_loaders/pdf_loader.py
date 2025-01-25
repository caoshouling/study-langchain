from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from .base_loader import BaseDocumentLoader

class PDFLoader(BaseDocumentLoader):
    """PDF文档加载器"""
    
    def load(self, file_path: str) -> List[Document]:
        """加载并解析PDF文档"""
        try:
            # 使用PyPDFLoader加载PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # 对每一页进行分割处理
            split_docs = []
            for doc in documents:
                # 获取页码信息
                page_number = doc.metadata.get('page', 1)
                
                # 分割当前页面的内容
                page_splits = self.split_documents([doc])
                
                # 更新每个分割后文档的元数据
                for split in page_splits:
                    split.metadata.update({
                        'page': page_number,
                        'source': file_path,
                        'sub_chunk': len(split_docs)  # 添加块序号
                    })
                
                split_docs.extend(page_splits)
                
            return split_docs
        except Exception as e:
            print(f"Error loading PDF file: {str(e)}")
            raise 