#根据输入的提示词长度综合计算最终长度，智能截取或者添加提示词的示例

from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

#假设已经有这么多的提示词示例组：
examples = [
    {"input":"happy","output":"sad"},
    {"input":"tall","output":"short"},
    {"input":"sunny","output":"gloomy"},
    {"input":"windy","output":"calm"},
    {"input":"高兴","output":"悲伤"}
]

#构造提示词模板
example_prompt = PromptTemplate(
    input_variables=["input","output"],
    template="原词：{input}\n反义：{output}"
)

#调用长度示例选择器
example_selector = LengthBasedExampleSelector(
    #传入提示词示例组
    examples=examples,
    #传入提示词模板
    example_prompt=example_prompt,
    #设置格式化后的提示词最大长度
    max_length=25,
    #内置的get_text_length,如果默认分词计算方式不满足，可以自己扩展
    #get_text_length:Callable[[str],int] = lambda x:len(re.split("\n| ",x))
)

#使用小样本提示词模版来实现动态示例的调用
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词",
    suffix="原词：{adjective}\n反义：",
    input_variables=["adjective"]
)

#小样本获得所有示例
print(dynamic_prompt.format(adjective="big"))



#-----------对话模式 - 少量样本学习----------
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate
)

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

example_prompt = ChatPromptTemplate.from_messages(
[('human', 'What is {input}?'), ('ai', '{output}')]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    # This is a prompt template used to format each individual example.
    example_prompt=example_prompt,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful AI Assistant'),
        few_shot_prompt,
        ('human', '{input}'),
    ]
)
print('-----------对话模式 - 少量样本学习-------')
print(final_prompt.format(input="What is 4+4?"))