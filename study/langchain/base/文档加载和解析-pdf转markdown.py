# pip install pypdf
# pip install pymupdf  # 用于PDF转换
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from 简单RAG import create_rag_chain, query_docs
import os
import fitz  # PyMuPDF
import re

def detect_structure(block, prev_font_size=0):
    """检测文本块的结构类型"""
    if "lines" not in block:
        return None, prev_font_size
    
    # 获取当前块的字体大小
    spans = block["lines"][0]["spans"]
    if not spans:
        return None, prev_font_size
    
    current_font_size = spans[0]["size"]
    text = spans[0]["text"].strip()
    
    # 检测标题
    if current_font_size > prev_font_size + 2:
        heading_level = min(6, max(1, int((current_font_size - prev_font_size) / 2)))
        return "heading", current_font_size, heading_level
    
    # 检测列表
    if text.startswith(("•", "-", "*", "1.", "2.")):
        return "list", current_font_size
    
    # 检测表格（简单启发式方法）
    if len(block["lines"]) > 1 and all(len(line["spans"]) > 1 for line in block["lines"]):
        return "table", current_font_size
        
    return "paragraph", current_font_size

def extract_table(block):
    """从块中提取表格内容并转换为Markdown格式"""
    table_rows = []
    for line in block["lines"]:
        row = [span["text"].strip() for span in line["spans"]]
        table_rows.append(row)
    
    # 创建Markdown表格
    if not table_rows:
        return ""
        
    # 添加表头
    markdown_table = ["| " + " | ".join(table_rows[0]) + " |"]
    # 添加分隔行
    markdown_table.append("| " + " | ".join(["---"] * len(table_rows[0])) + " |")
    # 添加数据行
    for row in table_rows[1:]:
        markdown_table.append("| " + " | ".join(row) + " |")
    
    return "\n".join(markdown_table)

def pdf_to_markdown(pdf_path: str) -> str:
    """
    将PDF转换为Markdown格式，支持更多格式特性
    
    特性：
    1. 智能标题检测
    2. 表格转换
    3. 列表识别
    4. 图片提取（可选）
    5. 格式保持
    """
    doc = fitz.open(pdf_path)
    markdown_text = []
    current_font_size = 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block.get("type") == 0:  # 文本块
                structure_type, font_size, *args = detect_structure(block, current_font_size) or (None, current_font_size)
                current_font_size = font_size
                
                if structure_type == "heading":
                    heading_level = args[0]
                    text = block["lines"][0]["spans"][0]["text"].strip()
                    markdown_text.append(f"\n{'#' * heading_level} {text}\n")
                
                elif structure_type == "table":
                    markdown_text.append("\n" + extract_table(block) + "\n")
                
                elif structure_type == "list":
                    for line in block["lines"]:
                        text = " ".join(span["text"].strip() for span in line["spans"])
                        markdown_text.append(text + "\n")
                
                else:  # 普通段落
                    paragraph = []
                    for line in block["lines"]:
                        text = " ".join(span["text"].strip() for span in line["spans"])
                        paragraph.append(text)
                    markdown_text.append(" ".join(paragraph) + "\n\n")
            
            elif block.get("type") == 1:  # 图片块
                # 可选：处理图片
                # image_info = f"\n![image_{page_num}_{block['number']}]()\n"
                # markdown_text.append(image_info)
                pass
    
    doc.close()
    markdown_content = "".join(markdown_text)
    
    # 清理格式
    markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)  # 删除多余空行
    markdown_content = re.sub(r' {2,}', ' ', markdown_content)      # 删除多余空格
    markdown_content = re.sub(r'\n +', '\n', markdown_content)      # 删除行首空格
    
    return markdown_content

def save_markdown_file(pdf_path: str, markdown_content: str) -> str:
    """
    保存Markdown内容到文件
    
    参数:
        pdf_path: PDF文件路径
        markdown_content: Markdown内容
        
    返回:
        生成的Markdown文件路径
    """
    # 构建输出文件路径
    base_path = os.path.splitext(pdf_path)[0]  # 移除.pdf后缀
    output_path = f"{base_path}_md.md"
    
    # 保存文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return output_path

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "data", "向量数据库选型.pdf")
data_dir = os.path.join(current_dir, "data")

# 将PDF转换为Markdown
print("正在将PDF转换为Markdown格式...")
markdown_content = pdf_to_markdown(pdf_path)

# 保存Markdown文件
output_path = save_markdown_file(pdf_path, markdown_content)
print(f"Markdown文件已保存到: {output_path}")

# 创建Document对象
from langchain_core.documents import Document
markdown_doc = Document(
    page_content=markdown_content,
    metadata={"source": pdf_path, "format": "markdown"}
)

print('-----------------------测试Markdown格式的RAG检索----------------------------')
# # 使用Markdown分割方法处理转换后的文档
# rag_chain_converted = create_rag_chain(
#     docs=[markdown_doc],
#     chunk_size=300,
#     chunk_overlap=50,
#     retriever_k=5,
#     split_method="markdown",  # 使用Markdown分割
#     show_retrieved_docs=True
# )

# print("\n使用转换后的Markdown文档的结果：")
# answer_converted = query_docs(rag_chain_converted, "我想要一个支持多语言SDK的向量数据库，有没有推荐的？")
# print(answer_converted) 