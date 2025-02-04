import json
import logging
import os
import random

from langchain.globals import set_debug, set_verbose
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import MessageGraph

# logging.basicConfig(level=logging.DEBUG)
# set_debug(True)
# set_verbose(True)
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langsmith-langgraph"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# llm = ChatOpenAI(
#     openai_api_base="https://8f13-154-12-181-41.ngrok-free.app/v1/",
#     # model="paultimothymooney/qwen2.5-7b-instruct",
#     model = "qwen2.5-14b-instruct",
#     api_key="323",
#     temperature=0
# )
#
# llm = ChatOpenAI(openai_api_base= "https://dashscope.aliyuncs.com/compatible-mode/v1",
#                    model_name='qwen-turbo',
#                  api_key ="sk-474e1a10893e4913bbe860dc90edda42",
#                  temperature=0)

#嘉宾智能体的提示词头部
player_prompt_header = """
请您永远记住您现在扮演{agent_role}角色。

您的基本介绍：{agent_description}
您的性格：{agent_nature}
您的经历：{agent_experience}

目前轮到你发言，请您根据上面的节目聊天内容以及您的角色、性格和经历，以及所处的位置角度提供该主题最丰富、最有创意和最新颖的观点，只返回你要表达的内容。
"""

# 根据嘉宾的列表，自动生成嘉宾的介绍和完整提示词

roleList = ["成龙","刘亦菲","沈腾","雷军"]
starParser = StrOutputParser()

roleDesPrompt = PromptTemplate.from_template("""
用户输入：{input}
请根据用户输入的明星，生成明星的详细介绍。返回内容必须按照下面的JSON格式返回，只返回JSON内容，不要返回斜杆的注释的内容。
{{
  name: str, //明星的名称
  description: str, //明星的基本介绍
  nature: str , //明星的性格
  experience: str //明星的经历   
}}
""")
roleDesChain = roleDesPrompt| llm| starParser


batchInput =[]
for item in roleList:
    batchInput.append({"input": item})
#批量生成明星的详细介绍
roleDesList = roleDesChain.batch(batchInput)
print('----------------------------roleDesList------------------------')
print(roleDesList)
roleDesListJson = [json.loads(item) for item in roleDesList]

# 生成嘉宾参与圆桌派的发言提示，包含：主题、嘉宾列表、聊天内容、嘉宾介绍
topic="出生-家世决定您多少"
player_prompt="""
这是圆桌派综艺界面，目前讨论的主题：{topic}

本期节目的嘉宾介绍：
{roleDesList}

节目聊天内容：
{chatList}

{roleDesc}
"""
playersPrompt =[]

for role in roleDesListJson:
    prompt = player_prompt_header.format(
        agent_role = role["name"],
        agent_description = role["description"],
        agent_nature = role["nature"],
        agent_experience = role["experience"],
    )
    playersPrompt.append(prompt)
# 每个嘉宾的发言模板
playerPromptList = []
for item in playersPrompt:
    playerPrompt = PromptTemplate.from_template(player_prompt)
    playerPrompt = playerPrompt.partial(
        roleList = ",".join(roleList),
        roleDesc = item
    )
    playerPromptList.append(playerPrompt)
print('----------------------------playerPromptList------------------------')
print(playerPromptList)

playerChains = []

# 构建处理链
for prompt in playerPromptList:
    playerChains.append(prompt| llm )
print('----------------------------playerChains------------------------')
print(playerChains)

print("-----------------构建主持人发言------------------------")
host_prompt = """
这是圆桌派综艺界面，目前讨论的主题：{topic}

本期节目的嘉宾介绍：{roleDesList}

节目聊天内容：
{chatList}

下一位发言的嘉宾：{player}

请永远记住您现在扮演节目主持人的角色，你的名字叫陈鹏。
目前轮到你发言，你需要根据上面节目聊天内容的进展来主持节目进行发言。如果节目尚未开始，你需要介绍嘉宾和本期节目的开场介绍，并引导下一位嘉宾发言。如果没有下一位嘉宾，请做好本次节目的总结并结束节目，只返回发言内容，不要添加其他内容。
"""
prompt = PromptTemplate.from_template(host_prompt)
hostChain = prompt | llm

print(prompt.format(
    topic=topic,
    chatList="节目刚开始，暂无聊天内容",
    roleDesList=roleDesListJson,
    player="成龙"
))
# result = hostChain.invoke({
#     "topic": topic,
#     "chatList": "节目刚开始，暂无聊天内容",
#     "roleDesList": roleDesListJson,
#     "player": "成龙",
# })
# print(result)

# 构建数据流图

graphBuilder = MessageGraph()
data = {
    "topic": topic,
    "chatList": "节目刚开始，暂无聊天内容",
    "roleDesList": roleDesListJson,
    "player": "成龙",
    "isEnd": False
}
def choose(state):
    print('-----------------choose------------------')
    if data["isEnd"]:
        return "end"
    if len(state) >5: #最多5个嘉宾
        data["isEnd"] = True
    # 选择下一个嘉宾
    for index in range(len(roleList)):
        if roleList[index] == data["player"]:
            return "player"+str(index+1)
    return "end"

# 更新聊天记录，并决定下一个发言嘉宾。如果节目没有结束，那么随机选择下一个嘉宾
def msgParser(state):
    print('-----------------msgParser--- start ---------------')
    print(state)
    if not isinstance(data["chatList"], str):
       data["chatList"].append("嘉宾("+data["player"]+"): "+state[-1].content)
    if data["isEnd"]:
       data["player"]  = "节目即将结束，不需要下一位嘉宾发言"
       print("节目即将结束，不需要下一位嘉宾发言")
       print(data)
    else:
        random_items = random.choices(roleList,k=1)
        data["player"] = random_items[0]
        print("随机选择下一位嘉宾："+data["player"])
    print('-----------------msgParser--- end ---------------')
    print(state)
    return data

def playMsgParser(state):
    print("-------------playMsgParser------------")
    print(state)
    if isinstance(data["chatList"],str):
        data["chatList"] = ["主持人(陈鹏): "+state[0].content]
    else:
        data["chatList"].append("主持人(陈鹏):  "+state[-1].content)
    return {
        "chatList":  data["chatList"],
        'topic': data["topic"]
    }

graphBuilder.add_node("hostNode", msgParser| hostChain)
graphBuilder.add_node("playerNode1", playMsgParser| playerChains[0])
graphBuilder.add_node("playerNode2", playMsgParser| playerChains[1])
graphBuilder.add_node("playerNode3", playMsgParser| playerChains[2])
graphBuilder.add_node("playerNode4", playMsgParser| playerChains[3])
graphBuilder.add_conditional_edges("hostNode", choose, {
    "end":END,
    "player1": "playerNode1",
    "player2": "playerNode2",
    "player3": "playerNode3",
    "player4": "playerNode4"
})

graphBuilder.add_edge("playerNode1", "hostNode")
graphBuilder.add_edge("playerNode2", "hostNode")
graphBuilder.add_edge("playerNode3", "hostNode")
graphBuilder.add_edge("playerNode4", "hostNode")

graphBuilder.set_entry_point("hostNode")
graph = graphBuilder.compile()
graph.invoke([HumanMessage(content=prompt.format(
    topic=topic,
    chatList="节目刚开始，暂无聊天内容",
    roleDesList=roleDesListJson,
    player="成龙"
))])

# 可视化图
graph_png = graph.get_graph().draw_mermaid_png()
with open("多智能体模拟圆桌派综艺节目.png", "wb") as f:
    f.write(graph_png)
# 显示图