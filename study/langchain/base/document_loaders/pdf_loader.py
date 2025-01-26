from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from .base_loader import BaseDocumentLoader
import pdfplumber
'''
  本类已实现了能解析不跨页的表格，但是不能解析跨页表格。
  有单元格合并的这种也可能也不能处理，但是没试过。

'''
class PDFLoader(BaseDocumentLoader):
    """PDF文档加载器"""
    
    def parse_table_with_pdfplumber(self, page) -> List[dict]:
        """使用pdfplumber解析PDF页面中的表格"""
        tables_data = []
        tables = page.extract_tables()
        print('---------tables------start------')
        print(tables)
        print('---------tables------end------')
        for table in tables:
            # 合并表头中的换行符
            headers = [''.join(header.splitlines()) for header in table[0]]  # 假设第一行为表头
            for row in table[1:]:
                # 直接合并单元格内容中的换行符
                row = [''.join(cell.splitlines()) for cell in row]
                if len(row) == len(headers):
                    tables_data.append(dict(zip(headers, row)))
                else:
                    # 如果行长度不匹配，尝试合并最后一个单元格
                    combined_last_cell = ''.join(row[len(headers)-1:])
                    row = row[:len(headers)-1] + [combined_last_cell]
                    tables_data.append(dict(zip(headers, row)))
        # 确保表格数据不重复
        unique_tables_data = [dict(t) for t in {tuple(d.items()) for d in tables_data}]
        return unique_tables_data

    def table_data_to_text(self, table_data: List[dict]) -> str:
        """将表格数据转换为文本格式"""
        if not table_data:
            return ''
        headers = table_data[0].keys()
        table_texts = ['\t'.join(map(str, headers))]  # 添加表头
        for table in table_data:
            row = list(table.values())
            table_text = '\t'.join(map(str, row))
            table_texts.append(table_text)
        return '\n'.join(table_texts)

    def load(self, file_path: str) -> List[Document]:
        """加载并解析PDF文档"""
        try:
            # 使用pdfplumber直接解析PDF
            with pdfplumber.open(file_path) as pdf:
                split_docs = []
                print(len(pdf.pages))
                index = 1
                for page_number, page in enumerate(pdf.pages, start=1):
                    # 提取文本
                    text = page.extract_text() or ''
                    
                    print(f'第{index}个text：{text}')

                    index = index+1
                    # 使用pdfplumber解析表格
                    table_data = self.parse_table_with_pdfplumber(page)
                    if table_data:
                        # 将表格数据转换为文本并合并到page_content中
                        table_text = self.table_data_to_text(table_data)
                        text += '\n\n' + table_text
                    # 创建一个完整的文档对象，不进行分割
                    print(f'第{index}个text：{text}')
                    if text:
                        doc = Document(page_content=text, metadata={'page': page_number, 'source': file_path})
                        split_docs.append(doc)
                # 将结果写入文件
                with open('output.txt', 'w', encoding='utf-8') as f:
                    for doc in split_docs:
                        f.write(doc.page_content + '\n\n')
                return split_docs
        except Exception as e:
            print(f"Error loading PDF file: {str(e)}")
            raise 