# -*- coding: utf-8 -*-
"""
用于旅行预定的多智能体系统 (MAS)
此系统包含一个航班预订代理和一个酒店预订代理，它们可以通过相互移交来协同完成用户的综合预订请求。
"""
import os
# 导入必要的模块
from typing import Annotated  # 用于为函数参数添加类型注解

# LangChain 相关模块
from langchain_core.messages import convert_to_messages  # 将消息列表转换为标准格式
from langchain_core.tools import tool, InjectedToolCallId  # 定义工具函数和注入工具调用ID
from langchain_openai import ChatOpenAI

# LangGraph 核心模块
from langgraph.prebuilt import create_react_agent, InjectedState  # 创建基于ReAct模式的智能体，注入状态
from langgraph.graph import StateGraph, START, MessagesState    # 构建状态图，定义起始节点，消息状态类型
from langgraph.types import Command                            # 用于控制流程跳转的命令


# ------------------- 辅助函数：美化输出 -------------------
def pretty_print_message(message, indent=False):
    """
    美化单条消息的打印输出。
    
    Args:
        message: 要打印的消息对象。
        indent (bool): 是否添加缩进（用于子图）。
    """
    pretty_message = message.pretty_repr(html=True)  # 获取消息的美化表示
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    """
    美化并打印整个更新流中的消息，使其结构清晰易读。
    
    Args:
        update: 来自图执行流的更新数据。
        last_message (bool): 是否只打印最后一条消息。
    """
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # 跳过主图的更新，专注于子图
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"来自子图 {graph_id} 的更新:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"来自节点 {node_name} 的更新:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

# 移交机制 (Handoff Pattern)：

# 这是本系统的关键。我们没有一个中央调度器，而是让每个代理都拥有一个“移交”工具。
# create_handoff_tool 函数是一个工厂，它能生成一个特殊的 tool。

# 当一个代理调用这个工具时，它不会返回一个简单的字符串，而是返回一个 Command 对象。

# 这个 Command 对象指示 LangGraph 的执行引擎：
# “停止在当前节点运行，跳转到 goto 指定的下一个节点，并带上更新后的状态”。

# ------------------- 创建移交工具 (Handoff Tools) -------------------
def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """
    工厂函数：创建一个可以将控制权移交给指定代理的特殊工具。
    这是实现多代理协作的核心机制。
    
    Args:
        agent_name (str): 目标代理的名称。
        description (str, optional): 该工具的描述。
        
    Returns:
        function: 一个可被智能体调用的工具函数。
    """
    name = f"transfer_to_{agent_name}"
    description = description or f"转移到 {agent_name}"

    @tool(name=name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],           # 注入当前对话状态
        tool_call_id: Annotated[str, InjectedToolCallId],         # 注入本次工具调用的ID
    ) -> Command:
        # 创建一条工具执行结果的消息
        tool_message = {
            "role": "tool",
            "content": f"成功转移到 {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(  
            goto=agent_name,             # 命令：下一步跳转到名为 agent_name 的节点
            update={"messages": state["messages"] + [tool_message]},  # 更新：将新消息追加到历史记录中
            graph=Command.PARENT,        # 指定跳转发生在父图层级
        )
    return handoff_tool

# 创建具体的移交工具实例
transfer_to_hotel_assistant = create_handoff_tool(
    agent_name="hotel_assistant",
    description="将用户转接给酒店预订助理。",
)
transfer_to_flight_assistant = create_handoff_tool(
    agent_name="flight_assistant",
    description="将用户转接给航班预订助理。",
)


# ------------------- 定义基础功能工具 -------------------
def book_hotel(hotel_name: str):
    """模拟预订酒店的操作"""
    return f"已成功预订 {hotel_name} 的住宿。"

def book_flight(from_airport: str, to_airport: str):
    """模拟预订航班的操作"""
    return f"已成功预订从 {from_airport} 到 {to_airport} 的航班。"


# ------------------- 创建智能代理 (Agents) -------------------
# 使用 create_react_agent 快速创建两个具备思考（ReAct）能力的智能体

model = ChatOpenAI(openai_api_base= "https://dashscope.aliyuncs.com/compatible-mode/v1",
                   model_name='qwen-turbo',
                 api_key =os.getenv('DASHSCOPE_API_KEY'))

flight_assistant = create_react_agent(
    model=model,  # 使用的模型（示例）
    tools=[book_flight, transfer_to_hotel_assistant],  # 该代理可用的工具集
    prompt="你是一位专业的航班预订助理。你的任务是帮助用户预订航班。如果用户还需要预订酒店，请使用 transfer_to_hotel_assistant 工具将他们转接给酒店助理。",  # 中文系统提示词
    name="flight_assistant"  # 代理名称
)

hotel_assistant = create_react_agent(
    model=model,
    tools=[book_hotel, transfer_to_flight_assistant],
    prompt="你是一位专业的酒店预订助理。你的任务是帮助用户预订酒店。如果用户还需要预订航班，请使用 transfer_to_flight_assistant 工具将他们转接给航班助理。",  # 中文系统提示词
    name="hotel_assistant"
)


# ------------------- 构建多代理工作流图 -------------------
# 创建一个以消息状态为基础的状态图
multi_agent_graph = (
    StateGraph(MessagesState)               # 初始化状态图为消息驱动
    .add_node(flight_assistant)             # 添加航班助理节点
    .add_node(hotel_assistant)              # 添加酒店助理节点
    .add_edge(START, "flight_assistant")    # 定义初始边：流程从航班助理开始
    .compile()                              # 编译图，使其可执行
)


# ------------------- 执行任务与查看结果 -------------------
# 向多代理系统提交一个包含多项任务的复杂请求
for chunk in multi_agent_graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "请帮我预订一张从波士顿(BOS)到纽约(JFK)的机票，以及在麦克基特里克酒店(McKittrick Hotel)的住宿。"
            }
        ]
    },
    subgraphs=True  # 包含子图的详细信息
):
    # 流式打印每一步的执行情况
    pretty_print_messages(chunk)


# 执行过程：
# 当用户提交一个包含航班和酒店需求的请求时，流程从 flight_assistant 开始。
# flight_assistant 在其 prompt 的指导下，识别出除了航班还有酒店需要预订。
# 因此，它会调用 transfer_to_hotel_assistant 工具。
# 这个工具的执行会触发一个 Command，导致执行引擎将控制权移交给 hotel_assistant 节点。
# hotel_assistant 接收到完整的对话历史，看到用户的需求，于是调用 book_hotel 工具完成预订。
# （在此示例中，流程会在酒店预订后结束。如果需要更复杂的循环，
# 可以设计 hotel_assistant 在完成后也移交回 flight_assistant 或其他节点。）