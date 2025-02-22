from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import MessageGraph



import logging
import os
from langchain.globals import set_debug, set_verbose
logging.basicConfig(level=logging.DEBUG)
set_debug(True)
set_verbose(True)
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langsmith-langgraph"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"



llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1/",
    # model="paultimothymooney/qwen2.5-7b-instruct",
    model = "qwen2.5-14b-instruct",
    api_key="323"
)
#
# llm = ChatOpenAI(openai_api_base= "https://dashscope.aliyuncs.com/compatible-mode/v1",
#                    model_name='qwen-turbo',
#                  api_key ="sk-474e1a10893e4913bbe860dc90edda42")

strParser =  StrOutputParser()

# 意图识别节点
prompt1 = ChatPromptTemplate.from_messages(

    [
        ("system","""你是1个意图识别的助手，能够识别以下意图：
        1、讲故事
        2、讲笑话
        3、AI绘画
        4、其他
        
        例如：
        用户输入：给我说个故事吧。
        1
        用户输入：给我画个美女图片
        3
        
        -----------
        用户输入{input}
        
        请识别用户的意图，返回上面意图的数字序号，只返回数字，不要返回任何其他字符。""")
    ]
)
chain = prompt1| llm| strParser
# 故事生成节点
prompt2 = ChatPromptTemplate.from_messages(

    [
        ("system", "你是一个故事大王，会讲创造引人入胜的各种故事。现在以{input}为主题写一篇故事。")
    ]
)
storyChain = prompt2 | llm


# 笑话生成节点
prompt3 = ChatPromptTemplate.from_messages(

    [
        ("system", """你是脱口秀主持人，会把各种故事讲的非常幽默风趣，请将------下面的内容以幽默的方式将出来。
        ------
        {input}""")
    ]
)
jokeChain = prompt3 | llm
# 做一个得力助手
prompt4 = ChatPromptTemplate.from_messages(

    [
        ("system", """你是一个得力的助手"""),
        ("human", "{input}"),
    ]
)
assistantChain = prompt4 | llm



#创建图实例

graph = MessageGraph()

# 处理节点结果的函数，每个节点是一个链，输入是一个字典，所以需要用这个函数来转换
def processFn(state):
    print('state[-1]--------')
    print(state[-1])
    return {"input" : state[0].content}

#添加节点

chain = processFn | prompt1 | llm

graph.add_node("startNode",chain)
graph.add_node("storyNode", processFn | storyChain)
graph.add_node("jokeNode",jokeChain)
graph.add_node("assistantNode",assistantChain)

graph.set_entry_point("startNode")

def router(state):
    if state[-1].content == "1":
        return "story"
    elif state[-1].content == "2":
        return "joke"
    else:
        return "assistant"

graph.add_conditional_edges("startNode", router,{
    "story": "storyNode",
    "joke": "jokeNode",
    "assistant": "assistantNode"
})
# 结束节点
graph.add_edge("storyNode", END)
graph.add_edge("jokeNode", END)
graph.add_edge("startNode", END)

#编译
simpleGraph = graph.compile()


simpleGraph.invoke([HumanMessage(content="光头强的故事")])
# 可视化图
graph_png = simpleGraph.get_graph().draw_mermaid_png()
with open("agent-simple.png", "wb") as f:
    f.write(graph_png)
# 显示图

