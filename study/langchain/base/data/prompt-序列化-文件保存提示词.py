# Final Prompt由一系列变量构成
from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate, load_prompt

#加载yaml格式的prompt模版
prompt = load_prompt("simple_prompt.yaml", 'UTF-8')
print(prompt.format(name="小黑",what="恐怖的"))

#加载json格式的prompt模版
prompt = load_prompt("simple_prompt.json", 'UTF-8')
print(prompt.format(name="小红",what="搞笑的"))


#支持加载文件格式的模版，并且对prompt的最终解析结果进行自定义格式化
prompt = load_prompt("prompt_with_output_parser.json")
print(prompt.output_parser.parse(
    "George Washington was born in 1732 and died in 1799.\nScore: 1/2"
))