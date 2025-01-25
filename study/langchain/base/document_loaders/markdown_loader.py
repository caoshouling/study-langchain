from typing import List, Tuple
from langchain_core.documents import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from .base_loader import BaseDocumentLoader
import re

def extract_svg_info(svg_text: str) -> str:
    """
    从SVG代码中提取有用的信息
    
    提取内容包括：
    1. 图表类型（如 flowchart）
    2. 节点文本内容
    3. 连接关系
    4. 图表描述（如果有）
    """
    info_parts = []
    
    # 提取图表类型
    chart_type = re.search(r'aria-roledescription="([^"]+)"', svg_text)
    if chart_type:
        info_parts.append(f"图表类型: {chart_type.group(1)}")
    
    # 提取节点文本
    node_texts = []
    for match in re.finditer(r'<span class="nodeLabel">.*?<p>(.*?)</p>.*?</span>', svg_text, re.DOTALL):
        node_texts.append(match.group(1).strip())
    
    if node_texts:
        info_parts.append("节点内容: " + " -> ".join(node_texts))
    
    # 如果是流程图，添加流程说明
    if chart_type and 'flowchart' in chart_type.group(1).lower():
        info_parts.append(f"流程说明: {' 流向 '.join(node_texts)}")
    
    return "\n".join(info_parts) if info_parts else ""

def process_markdown_content(text: str) -> str:
    """
    处理Markdown内容，包括特殊元素的处理
    """
    # 处理SVG图片
    def replace_svg(match):
        svg_content = match.group(0)
        info = extract_svg_info(svg_content)
        if info:
            return f"\n[图表描述]\n{info}\n"
        return ""
    
    # 替换SVG标签及其内容
    text = re.sub(r'<svg[^>]*>.*?</svg>', replace_svg, text, flags=re.DOTALL)
    
    return text

def is_table_content(text: str) -> bool:
    """
    判断文本是否为表格内容
    
    支持以下格式：
    1. 标准Markdown表格
    2. 带标题的表格
    3. 网格线表格
    4. 简单的两列表格
    """
    # 移除前后空白
    text = text.strip()
    
    # 检查是否在代码块内
    if re.search(r'```.*```', text, re.DOTALL):
        return False
        
    # 标准Markdown表格模式
    standard_table = bool(re.match(r'\|[^\n]+\|\n\|[-:\|\s]+\|', text))
    
    # 网格线表格模式 (如 +----+----+)
    grid_table = bool(re.match(r'[+\-]+\n\|[\s\S]+\|', text))
    
    # 简单两列表格 (如 key: value 形式)
    simple_table = bool(re.match(r'(?:[^\n]+\s*\|\s*[^\n]+\n){2,}', text))
    
    # 带标题的表格
    titled_table = bool(re.match(r'(?:Table|表格|表)[:：\s-]+.*?\n\s*\|', text, re.IGNORECASE))
    
    return any([standard_table, grid_table, simple_table, titled_table])

def find_code_blocks(text: str) -> List[Tuple[int, int]]:
    """
    找出所有代码块的位置
    返回列表of (start, end) tuples
    """
    code_block_positions = []
    pattern = r'```[\s\S]*?```'
    for match in re.finditer(pattern, text):
        code_block_positions.append(match.span())
    return code_block_positions

def protect_table_content(text: str) -> List[str]:
    """
    保护表格内容，将其作为整体处理
    
    特性：
    1. 保护标准Markdown表格
    2. 保护带标题的表格
    3. 保护网格线表格
    4. 忽略代码块中的表格
    5. 保留表格前后的空行以维持格式
    """
    # 首先找出所有代码块的位置
    code_blocks = find_code_blocks(text)
    
    # 定义更复杂的表格模式
    table_patterns = [
        # 标准Markdown表格（包括可能的标题）
        r'(?:(?:Table|表格|表)[:：\s-]+.*?\n)?\s*\|[^\n]+\|\n\|[-:\|\s]+\|\n(?:\|[^\n]+\|\n)+',
        # 网格线表格
        r'[+\-]+\n\|[\s\S]+?\|\n[+\-]+\n',
        # 简单两列表格
        r'(?:[^\n]+\s*\|\s*[^\n]+\n){2,}'
    ]
    
    chunks = []
    last_end = 0
    
    # 合并所有表格模式
    combined_pattern = '|'.join(f'({pattern})' for pattern in table_patterns)
    
    for match in re.finditer(combined_pattern, text):
        start, end = match.span()
        
        # 检查是否在代码块内
        in_code_block = any(code_start <= start <= end <= code_end 
                          for code_start, code_end in code_blocks)
        if in_code_block:
            continue
            
        # 添加表格前的文本
        if start > last_end:
            pre_text = text[last_end:start].rstrip()
            if pre_text:
                chunks.append(pre_text)
        
        # 获取表格内容，保留前后的空行
        table_content = text[start:end]
        # 规范化表格格式
        table_content = re.sub(r'\n{3,}', '\n\n', table_content)
        chunks.append(table_content)
        
        last_end = end
    
    # 添加最后一个表格后的文本
    if last_end < len(text):
        remaining_text = text[last_end:].strip()
        if remaining_text:
            chunks.append(remaining_text)
    
    return chunks if chunks else [text]

class MarkdownLoader(BaseDocumentLoader):
    """Markdown文档加载器"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 150):
        super().__init__(chunk_size, chunk_overlap)
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        # 设置二次分割阈值为chunk_size的1.6倍
        self.secondary_split_threshold = int(chunk_size * 1.6)
        # 初始化递归分割器作为后备方案
        self.backup_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
            ]
        )
        
    def load(self, file_path: str) -> List[Document]:
        """加载并解析Markdown文档"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # 预处理文本，处理SVG等特殊内容
        text = process_markdown_content(text)
        
        # 创建基础文档
        base_doc = Document(page_content=text, metadata={"source": file_path})
        
        # 使用Markdown特定的分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            return_each_line=False
        )
        
        try:
            # 首先保护表格内容
            protected_chunks = protect_table_content(text)
            final_docs = []
            
            for chunk in protected_chunks:
                if is_table_content(chunk):
                    # 表格内容作为一个整体保存
                    final_docs.append(Document(
                        page_content=chunk,
                        metadata={"source": file_path, "content_type": "table"}
                    ))
                else:
                    try:
                        # 尝试使用Markdown分割
                        docs = markdown_splitter.split_text(chunk)
                        for doc in docs:
                            # 对大块进行二次分割，使用设定的阈值
                            if len(doc.page_content) > self.secondary_split_threshold:
                                sub_docs = self.backup_splitter.split_documents([doc])
                                final_docs.extend(sub_docs)
                            else:
                                final_docs.append(doc)
                    except Exception as e:
                        print(f"Markdown splitting failed for chunk, using backup splitter: {str(e)}")
                        # 使用后备分割器
                        sub_docs = self.backup_splitter.split_documents([Document(
                            page_content=chunk,
                            metadata={"source": file_path}
                        )])
                        final_docs.extend(sub_docs)
            
            # 确保所有文档都有正确的元数据
            for doc in final_docs:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = file_path
            
            return final_docs
            
        except Exception as e:
            print(f"All splitting methods failed, using backup splitter as last resort: {str(e)}")
            # 使用后备分割器作为最后的方案
            return self.backup_splitter.split_documents([base_doc]) 