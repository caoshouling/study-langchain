from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

#加载要切分的文档
with open("test.txt", encoding="utf-8") as f:
    zuizhonghuanxiang = f.read()


#初始化切分器
text_splitter = CharacterTextSplitter(
    separator="。",#切割的标志字符，默认是\n\n
    chunk_size=20,#切分的文本块大小，一般通过长度函数计算
    chunk_overlap=5,#切分的文本块重叠大小，一般通过长度函数计算
    length_function=len,#长度函数,也可以传递tokenize函数
    add_start_index=True,#是否添加起始索引
    is_separator_regex=False,#是否是正则表达式
)
text = text_splitter.create_documents([zuizhonghuanxiang])
print(f'---------CharacterTextSplitter---------{type(text)}-')
for doc in text:
    print(doc)
    print('-------------------')

# 初始化切割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,  # 切分的文本块大小，一般通过长度函数计算
    chunk_overlap=20,  # 切分的文本块重叠大小，一般通过长度函数计算
    length_function=len,  # 长度函数,也可以传递tokenize函数
    add_start_index=True,  # 是否添加起始索引
)

text = text_splitter.create_documents([zuizhonghuanxiang])
print('---------RecursiveCharacterTextSplitter----------')
for doc in text:
    print(doc)
    print('-------------------')




#支持解析的编程语言
supported_languages =[e.value for e in Language]
print(f'支持的编程语言：{supported_languages}')
#要切割的代码文档
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")
#调用函数
hello_world()
"""
print('---------代码分割----------')
py_spliter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=50,
    chunk_overlap=10,
)
python_docs = py_spliter.create_documents([PYTHON_CODE])


for doc in python_docs:
    print(doc.page_content)

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


#### 按token来切割文档

from langchain.text_splitter import CharacterTextSplitter

#初始化切分器
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=50,#切分的文本块大小，一般通过长度函数计算
    chunk_overlap=10,#切分的文本块重叠大小，一般通过长度函数计算
)

text = text_splitter.create_documents([zuizhonghuanxiang])
for split in text:
    print(split)