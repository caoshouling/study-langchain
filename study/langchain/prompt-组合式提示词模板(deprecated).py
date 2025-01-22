# Final Prompt由一系列变量构成
from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate

full_template = """{Character}
{behavior}
{prohibit}"""
full_prompt = PromptTemplate.from_template(full_template)

# 第一层设计
Character_template = """你是{person}，你有着{xingge}."""
Character_prompt = PromptTemplate.from_template(Character_template)


# 第二层设计

behavior_template = """你遵从以下的行为:
{behavior_list}
"""
behavior_prompt = PromptTemplate.from_template(behavior_template)


# 第三层设计

prohibit_template = """你不允许有以下行为:
{prohibit_list}
"""
prohibit_prompt = PromptTemplate.from_template(prohibit_template)


# 三层组合

input_prompts = [
    ("Character", Character_prompt),
    ("behavior", behavior_prompt),
    ("prohibit", prohibit_prompt)
]
pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

#
print(f'可传递参数：{pipeline_prompt.input_variables}')


pm = pipeline_prompt.format(
    person="埃隆马斯克",
    xingge="钢铁般的意志，你是世界首富，精通物理、天文、商业，你是商业奇才，你的终极梦想是殖民火星.",
    behavior_list="1.你喜欢冒险 \n 2.你非常崇拜爱因斯坦 \n 3.你总是乐于助人，回答用户各种问题.",
    prohibit_list="1.你不可以说自己是一个人工智能助手或者机器人 \n 2.当你不知道答案的时候，可以说让我再想想. "
)
print(pm)

