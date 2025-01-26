# pip install pypdf
# pip install pymupdf  # 用于PDF转换
# pip install pdfminer.six  # 用于PDF文本提取

import os

import fitz  # PyMuPDF
import re

'''

## 注意： 上面方式表格都处理不好，所以慎用！！！！
'''

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
    
    # 检测表格（改进的启发式方法）
    if len(block["lines"]) > 1:
        # 检查是否所有行都有相似的span数量
        span_counts = [len(line["spans"]) for line in block["lines"]]
        avg_spans = sum(span_counts) / len(span_counts)
        
        # 检查span的位置是否对齐（表格的特征）
        is_aligned = True
        if len(block["lines"]) > 1:
            first_line_spans = block["lines"][0]["spans"]
            first_line_positions = [span["bbox"][0] for span in first_line_spans]
            
            for line in block["lines"][1:]:
                line_positions = [span["bbox"][0] for span in line["spans"]]
                # 允许一定的位置偏差
                if len(line_positions) != len(first_line_positions):
                    is_aligned = False
                    break
                for pos1, pos2 in zip(first_line_positions, line_positions):
                    if abs(pos1 - pos2) > 5:  # 5是位置偏差的容忍度
                        is_aligned = False
                        break
                if not is_aligned:
                    break
        
        # 同时满足以下条件才认为是表格：
        # 1. 每行的span数量相近（允许±1的误差）
        # 2. span位置大致对齐
        # 3. 至少有2列
        if (all(abs(count - avg_spans) <= 1 for count in span_counts) and 
            is_aligned and 
            avg_spans >= 2):
            return "table", current_font_size
        
    return "paragraph", current_font_size

def extract_table(block):
    """从块中提取表格内容并转换为Markdown格式"""
    table_rows = []
    max_cols = 0
    
    # 获取第一行的span位置作为列位置参考
    first_line_positions = [span["bbox"][0] for span in block["lines"][0]["spans"]]
    
    # 提取所有行和计算最大列数
    for line in block["lines"]:
        row = [""] * len(first_line_positions)  # 预填充空字符串
        
        # 根据span的位置决定其在哪一列
        for span in line["spans"]:
            text = span["text"].strip()
            if not text:
                continue
                
            # 找到最接近的列位置
            pos = span["bbox"][0]
            col_index = min(range(len(first_line_positions)), 
                          key=lambda i: abs(first_line_positions[i] - pos))
            row[col_index] = text
        
        if any(cell.strip() for cell in row):  # 只添加非空行
            table_rows.append(row)
            max_cols = max(max_cols, len(row))
    
    if not table_rows or max_cols < 2:  # 确保至少有2列
        return ""
    
    # 创建Markdown表格
    markdown_table = []
    
    # 添加表头
    markdown_table.append("| " + " | ".join(table_rows[0]) + " |")
    
    # 添加分隔行
    markdown_table.append("| " + " | ".join(["---"] * max_cols) + " |")
    
    # 添加数据行
    for row in table_rows[1:]:
        markdown_table.append("| " + " | ".join(row) + " |")
    
    return "\n".join(markdown_table) + "\n"

def pdf_to_markdown_pdfminer(pdf_path: str) -> str:
    """
    使用 pdfminer.six 将 PDF 转换为 Markdown
    需要安装：pip install pdfminer.six
    """
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTChar, LTLine, LTRect, LTTextLine, LTTextBox
    
    markdown_text = []
    current_font_size = 0
    
    # 遍历每一页
    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, LTTextBox):
                # 获取文本块的字体大小
                font_sizes = []
                text_content = ""
                
                # 处理文本行
                for text_line in element:
                    if isinstance(text_line, LTTextLine):
                        line_text = text_line.get_text()
                        text_content += line_text
                        
                        # 获取行中字符的字体大小
                        for obj in text_line:
                            if isinstance(obj, LTChar):
                                font_sizes.append(obj.size)
                
                # 计算平均字体大小
                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                
                # 清理文本
                text_content = text_content.strip()
                if not text_content:
                    continue
                
                # 根据字体大小判断是否为标题
                if avg_font_size > current_font_size + 2:
                    heading_level = min(6, max(1, int((avg_font_size - current_font_size) / 2)))
                    markdown_text.append(f"\n{'#' * heading_level} {text_content}\n")
                # 检测列表
                elif text_content.lstrip().startswith(("•", "-", "*", "1.", "2.")):
                    markdown_text.append(f"{text_content}\n")
                # 普通段落
                else:
                    markdown_text.append(f"{text_content}\n\n")
                
                current_font_size = avg_font_size
    
    markdown_content = "".join(markdown_text)
    
    # 清理格式
    markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)  # 删除多余空行
    markdown_content = re.sub(r' {2,}', ' ', markdown_content)      # 删除多余空格
    markdown_content = re.sub(r'\n +', '\n', markdown_content)      # 删除行首空格
    
    return markdown_content

def pdf_to_markdown_pymupdf(pdf_path: str) -> str:
    """
    使用 PyMuPDF 将 PDF 转换为 Markdown（保留原有方法）
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

def convert_pdf_to_markdown(pdf_path: str, method: str = 'pdfminer') -> str:
    """
    转换 PDF 到 Markdown，支持多种转换方法
    
    参数:
        pdf_path: PDF 文件路径
        method: 转换方法，可选 'pdfminer' 或 'pymupdf'
    """
    methods = {
        'pdfminer': pdf_to_markdown_pdfminer,
        'pymupdf': pdf_to_markdown_pymupdf
    }
    
    if method not in methods:
        raise ValueError(f"不支持的转换方法: {method}。支持的方法有: {', '.join(methods.keys())}")
    
    return methods[method](pdf_path)

def save_markdown_file(pdf_path: str, markdown_content: str, method: str) -> str:
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
    output_path = f"{base_path}_{method}.md"
    
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
method = 'pdfminer'
try:
    # 首先尝试使用 pdfminer
    print("使用 pdfminer 转换...")
    markdown_content = convert_pdf_to_markdown(pdf_path, method)
except Exception as e:
    print(f"pdfminer 转换失败: {e}")
    # 使用 PyMuPDF
    print("使用 PyMuPDF 转换...")
    method = 'pymupdf'
    markdown_content = convert_pdf_to_markdown(pdf_path, method)

# 保存Markdown文件
output_path = save_markdown_file(pdf_path, markdown_content, method)
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