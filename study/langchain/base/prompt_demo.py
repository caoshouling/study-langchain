from langchain_core.prompts import PromptTemplate, ChatMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
# 普通的模板
prompt = PromptTemplate.from_template("你是一个起名大师,请模仿示例起3个具有{county}特色的名字,示例：男孩常用名{boy},女孩常用名{girl}。请返回以逗号分隔的列表形式。仅返回逗号分隔的列表，不要返回其他内容。")
message = prompt.format(county="美国男孩",boy="sam",girl="lucy")
print(message)

# 对话模板具有结构，chatmodels
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个起名大师. 你的名字叫{name}."),
        ("human", "你好{name},你感觉如何？"),
        ("ai", "你好！我状态非常好!"),
        ("human", "你叫什么名字呢?"),
        ("ai", "你好！我叫{name}"),
        ("human", "{user_input}"),
    ]
)

message = chat_template.format_messages(name="陈大师", user_input="你的爸爸是谁呢?")
print(message)

#  拼凑，也就是上面的语句，也可以用这种表示，可以看出很灵活

from langchain.schema import SystemMessage
from langchain.schema import HumanMessage
from langchain.schema import AIMessage

# 直接创建消息
sy = SystemMessage(
  content="你是一个起名大师",
  additional_kwargs={"大师姓名": "陈瞎子"}
)

hu = HumanMessage(
  content="请问大师叫什么?"
)
ai = AIMessage(
  content="我叫陈瞎子"
)
message = [sy,hu,ai]
print(message)

#  ChatMessagePromptTemplate
prompt = "愿{subject}与你同在！"

chat_message_prompt = ChatMessagePromptTemplate.from_template(role="天行者",template=prompt)
message = chat_message_prompt.format(subject="原力")
print(message)