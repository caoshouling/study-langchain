# from typing import List
# from langchain_core.documents import Document
# from .base_loader import BaseDocumentLoader
# import pandas as pd
#
# class ExcelLoader(BaseDocumentLoader):
#     """Excel文档加载器"""
#
#     def __init__(self, chunk_size: int = 500, chunk_overlap: int = 150):
#         super().__init__(chunk_size, chunk_overlap)
#
#     def load(self, file_path: str) -> List[Document]:
#         """加载并解析Excel文档"""
#         try:
#             # 使用pandas加载Excel
#             df = pd.read_excel(file_path)
#             documents = []
#             for index, row in df.iterrows():
#                 content = '\t'.join(map(str, row.values))
#                 doc = Document(page_content=content, metadata={'row': index, 'source': file_path})
#                 documents.append(doc)
#             return documents
#         except Exception as e:
#             print(f"Error loading Excel file: {str(e)}")
#             raise