from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

#加载要切分的文档
with open("data/test.txt", encoding="utf-8") as f:
    zuizhonghuanxiang = f.read()



from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_document = """
# 主标题

## 副标题1

这是第一段内容。

## 副标题2

这是第二段内容。
"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
splits = markdown_splitter.split_text(markdown_document)

for split in splits:
    print(split)

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

markdown_document = """# Intro

## History

Markdown is a lightweight markup language for creating formatted text using a plain-text editor.

## Rise and divergence

As Markdown popularity grew rapidly, many Markdown implementations appeared.

## Implementations

Implementations of Markdown are available for over a dozen programming languages."""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

# 按标题进行分割
# 默认情况下，分割过程会去除标题信息，如果需要保留，请设置strip_headers=False
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
md_header_splits = markdown_splitter.split_text(markdown_document)

# 字符级分割
chunk_size = 100
chunk_overlap = 10
# 使用RecursiveCharacterTextSplitter可以进一步控制每个文档块的大小，以适应更复杂的应用场景。
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
splits = text_splitter.split_documents(md_header_splits)

for split in splits:
    print(split)