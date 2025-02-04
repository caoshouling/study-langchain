# pip install langchain-experimental
import logging

from langchain.globals import set_debug, set_verbose
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI

import logging
from langchain.globals import set_debug, set_verbose
logging.basicConfig(level=logging.DEBUG)
set_debug(True)
set_verbose(True)
import os
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langsmith-basic"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

python_repl = PythonREPL()

print(python_repl.run("print(1+3)"))


promptFormat = """{query}

请根据上面的问题，生成Python代码计算出问题的答案，最后计算出来的结果用print()打印出来，请直接返回Python代码，不要返回其他任何内容的字符串
"""

prompt = ChatPromptTemplate.from_template(promptFormat)

output_parser = StrOutputParser()
def parsePython(codeStr):
    codeStr = codeStr.replace("```python","")
    codeStr = codeStr.replace("```", "")
    return codeStr
llm = ChatOpenAI(
    openai_api_base="https://8f13-154-12-181-41.ngrok-free.app/v1/",
    model="paultimothymooney/qwen2.5-7b-instruct",
    # model = "qwen2.5-14b-instruct",
    api_key="323",
    verbose=True

)
chain = prompt | llm | output_parser | parsePython | python_repl.run

result = chain.invoke({"query":"3箱苹果重45千克，一箱梨比一箱苹果多5千克，3箱梨重多少千克？"})

print(result)
