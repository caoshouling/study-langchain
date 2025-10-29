import os
import re
import operator
from dataclasses import dataclass
from typing import Annotated, Dict, List, Literal, Optional, Tuple

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# ------------------------------
# 步骤 1: 配置大语言模型 (LLM)
# ------------------------------
# 使用您提供的模型配置，并建议通过环境变量来管理 API Key
llm = ChatOpenAI(
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name='qwen-turbo',
    api_key=os.getenv("DASHSCOPE_API_KEY", "sk-d9ca67dd361c4347b582386197867c05"),
    temperature=0,
)


# ------------------------------
# 模拟“政策事项”数据库
# ------------------------------


@dataclass
class PolicyItem:
    name: str
    industry_tags: List[str]
    min_employees: Optional[int]
    max_employees: Optional[int]
    requires_above_scale: bool  # 是否要求规上企业
    description: str


POLICY_DB: List[PolicyItem] = [
    PolicyItem(
        name="高新技术企业认定补贴",
        industry_tags=["高新", "技术", "研发", "科技"],
        min_employees=10,
        max_employees=None,
        requires_above_scale=False,
        description="支持符合条件的高新技术企业，鼓励研发投入。",
    ),
    PolicyItem(
        name="智能制造升级改造补贴",
        industry_tags=["制造", "装备", "工业", "智能"],
        min_employees=50,
        max_employees=None,
        requires_above_scale=True,
        description="支持制造业企业实施智能化改造与数字化转型。",
    ),
    PolicyItem(
        name="软件企业增值税即征即退",
        industry_tags=["软件", "信息", "IT", "服务"],
        min_employees=5,
        max_employees=None,
        requires_above_scale=False,
        description="符合条件的软件企业可享受增值税即征即退政策。",
    ),
    PolicyItem(
        name="专精特新小巨人支持",
        industry_tags=["专精特新", "制造", "中小", "创新"],
        min_employees=20,
        max_employees=3000,
        requires_above_scale=False,
        description="支持专精特新‘小巨人’企业提升专业化与创新能力。",
    ),
]


# 辅助函数，根据名称查找政策详情
def find_policy_by_name(name: str) -> Optional[PolicyItem]:
    for p in POLICY_DB:
        if p.name == name:
            return p
    return None


# ------------------------------
# 基础工具：文本相似度（不依赖外部包）
# ------------------------------


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def token_set(text: str) -> set:
    # 简单按中文或英文分词，尽量鲁棒
    tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text)
    return set(tokens)


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def seq_similarity(a: str, b: str) -> float:
    # 简易序列相似度
    from difflib import SequenceMatcher

    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def hybrid_similarity(a: str, b: str) -> float:
    # 组合相似度，避免依赖外部向量库
    return 0.6 * seq_similarity(a, b) + 0.4 * jaccard_similarity(a, b)


# ------------------------------
# 步骤 2: 将业务逻辑封装为工具
# ------------------------------


@tool
def search_policies(policy_keywords: str) -> str:
    """
    根据用户提供的政策关键词，搜索匹配的政策列表。
    当用户首次提问或想要寻找某项政策时使用。
    :param policy_keywords: str, 用户输入中提取的政策相关的关键词，例如 '高新技术', '软件企业'
    :return: str, 返回一个格式化的字符串，包含最匹配的几项政策名称和它们的简介。如果找不到则返回提示。
    """
    print(f"--- TOOL: search_policies, keywords: '{policy_keywords}' ---")
    scored: List[Tuple[str, float]] = []
    for p in POLICY_DB:
        # 首先检查是否完全匹配政策名称
        name_similarity = hybrid_similarity(policy_keywords, p.name)
        tag_similarity = max((hybrid_similarity(policy_keywords, t) for t in p.industry_tags), default=0.0)
        score = max(name_similarity, tag_similarity)
        scored.append((p.name, round(float(score), 4)))

    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored or scored[0][1] <= 0.75:
        # 分数太低，未找到相关政策
        if scored and scored[0][1] > 0.5:
            # 找到一些相似的结果，但相似度不够
            results = "\n".join(
                [f"- 政策名称: {p_name}" for p_name, score in scored[:3] if score > 0.3]
            )
            return f"未找到完全匹配的政策，但找到以下相似的政策供参考：\n{results}\n请确认您是否想咨询其中某一项，或提供更具体的政策名称。"
        return "未找到相关的政策。请尝试提供更具体的政策名称或关键词。"

    top_score = scored[0][1]
    
    # 判断是否有多个相似的结果（第二高分与最高分接近，差值 < 0.1 则认为有多个）
    has_multiple_matches = len(scored) > 1 and (top_score - scored[1][1]) < 0.1
    
    # 情况1: >= 0.9 且只有一个（最高分明显高于第二高分）
    if top_score >= 0.9 and not has_multiple_matches:
        policy = find_policy_by_name(scored[0][0])
        return f"已找到明确的政策：\n- 政策名称: {policy.name}\n- 简介: {policy.description}\n\n可以直接进行资格判断。"
    
    # 情况2: >= 0.85 且 < 0.9，让大模型确认是否要找
    if 0.85 <= top_score < 0.9 and not has_multiple_matches:
        policy = find_policy_by_name(scored[0][0])
        return f"找到可能匹配的政策，相似度 {top_score:.2f}：\n- 政策名称: {policy.name}\n- 简介: {policy.description}\n\n请向用户确认是否就是要查找这项政策。"
    
    # 情况3: >= 0.85 且有多个，选择 top 3，让大模型确认哪个
    if top_score >= 0.85 and has_multiple_matches:
        top_3 = scored[:3]
        # 过滤掉分数明显低于最高分的结果（差值 > 0.1）
        filtered_top = [item for item in top_3 if (top_score - item[1]) < 0.1]
        results = "\n".join(
            [f"- 政策名称: {p_name}, 简介: {find_policy_by_name(p_name).description}, 相似度: {score:.2f}" 
             for p_name, score in filtered_top]
        )
        return f"找到多个相似的政策（相似度 >= 0.85），请向用户确认具体想咨询哪一项：\n{results}"
    
    # 情况4: > 0.75 且 < 0.85，告诉大模型没找到，但是找到相似的
    if 0.75 < top_score < 0.85:
        top_3 = scored[:3]
        results = "\n".join(
            [f"- 政策名称: {p_name}, 简介: {find_policy_by_name(p_name).description}, 相似度: {score:.2f}" 
             for p_name, score in top_3]
        )
        return f"未找到完全匹配的政策，但找到以下相似的政策供参考：\n{results}\n请确认您是否想咨询其中某一项，或提供更具体的政策名称。"
    
    # 默认情况（理论上不会到达这里）
    return "未找到相关的政策。请尝试提供更具体的政策名称或关键词。"


@tool
def judge_eligibility(
    policy_name: str, company_name: str, industry: str, employees: int, is_above_scale: bool
) -> str:
    """
    在收集到所有必要信息后，立即判断公司是否符合特定政策的申请资格。
    【重要】一旦你从用户处收集到以下所有信息：policy_name, company_name, industry, employees, is_above_scale，就应该立即调用此工具进行判断，不要只是告诉用户"请稍等"，而是要直接调用此工具。
    :param policy_name: str, 明确的政策全称，例如 "高新技术企业认定补贴".
    :param company_name: str, 公司的全称.
    :param industry: str, 公司所属行业, 例如 '软件', '制造', '高新'.
    :param employees: int, 公司的员工总数.
    :param is_above_scale: bool, 公司是否为规上企业（True 或 False）.
    :return: str, 返回最终的判断结论和原因。
    """
    print(
        f"--- TOOL: judge_eligibility for '{company_name}' on '{policy_name}' ---"
    )
    policy = find_policy_by_name(policy_name)
    if not policy:
        return f"错误：未在数据库中找到名为 '{policy_name}' 的政策。"

    # 行业判断
    industry_ok = any(t in industry for t in policy.industry_tags)

    # 人数判断
    emp_ok = True
    if policy.min_employees is not None and employees < policy.min_employees:
        emp_ok = False
    if policy.max_employees is not None and employees > policy.max_employees:
        emp_ok = False

    # 规上判断
    scale_ok = (not policy.requires_above_scale) or is_above_scale

    ok = industry_ok and emp_ok and scale_ok
    reason_parts = []
    if not industry_ok:
        reason_parts.append(f"行业不匹配（政策要求: {','.join(policy.industry_tags)}, 企业所属: {industry}）")
    if not emp_ok:
        reason_parts.append(f"员工人数不满足门槛（政策要求: {policy.min_employees or '无下限'}-{policy.max_employees or '无上限'}, 企业人数: {employees}）")
    if not scale_ok:
        reason_parts.append("不满足规上企业要求")

    decision = "符合" if ok else "不符合"
    reason = "；".join(reason_parts) if reason_parts else "满足所有基本条件"
    final = (
        f"对《{policy.name}》的资格判断结论：**{decision}**\n"
        f"原因: {reason}。\n"
        f"政策简介: {policy.description}。"
    )
    print(f"--- TOOL: judge_eligibility, result: {final} ---")
    # 在返回结果前添加明确的成功标识
    return f"判断完成。结论如下：\n{final}"


tools = [search_policies, judge_eligibility]
# 将 LLM 与工具进行绑定
llm_with_tools = llm.bind_tools(tools)


# ------------------------------
# 步骤 3: 定义 Agent 状态和图节点
# ------------------------------

# 使用 LangGraph 内置的 MessagesState，方便管理对话历史
# class AgentState(MessagesState):
#     pass

def should_continue(state: MessagesState) -> Literal["tools", END]:
    """判断是继续调用工具还是结束"""
    messages = state["messages"]
    last_message = messages[-1]
    # 如果 assistant role 的最后一条消息有工具调用，则路由到工具节点
    if last_message.tool_calls:
        return "tools"
    # 否则，结束
    return END

def call_model(state: MessagesState):
    """调用 LLM 的节点"""
    messages = state["messages"]
    # 在这里添加系统提示，指导 Agent 的行为
    system_prompt = SystemMessage(
        content="""你是一个专业的惠企政策顾问。你的任务是帮助用户判断他们的公司是否符合某项政策的申请资格。
        工作流程如下:
        1.  首先，使用 `search_policies` 工具，根据用户的提问找到最相关的政策。
        2.  根据工具的返回结果，按以下策略处理：
           - 如果工具返回"已找到明确的政策"（相似度>=0.9且只有一个），说明用户已经明确提到了政策名称，直接使用该政策，不需要再询问用户确认。
           - 如果工具返回"找到可能匹配的政策"（相似度>=0.85且<0.9），请向用户确认是否就是要查找这项政策。
           - 如果工具返回"找到多个相似的政策"（相似度>=0.85且有多个），请向用户确认具体想咨询哪一项。
           - 如果工具返回"未找到完全匹配的政策，但找到以下相似的政策"（相似度>0.75且<0.85），请向用户展示相似政策，并确认是否想咨询其中某一项。
           - 如果工具返回"未找到相关的政策"，说明找不到匹配的政策，请友好地告知用户。
        3.  在确认政策后，你需要收集判断资格所需的全部信息：公司全称、所属行业、员工人数、是否为规上企业。逐一向用户提问来收集这些信息，不要一次性问所有问题。
        4. 当且仅当你确认以下4项信息都已收集齐全时：
        - 公司全称 company_name
        - 所属行业 industry
        - 员工人数 employee_count
        - 是否为规上企业 is_large_enterprise
        你必须**立刻**调用工具 `judge_eligibility`。
        此时，不允许先回复自然语言（例如“请稍等”或“我来判断”），而是直接调用工具 `judge_eligibility`。
        5.  最后，将判断结果清晰地呈现给用户。
        关键行为规则：
        - 严格按照工具返回的提示信息来处理，不要自作主张地重复询问用户已经明确提到的政策。
        - 当收集完所有信息后，必须立即调用 `judge_eligibility` 工具，不要只是告诉用户"请稍等"，而是直接执行工具调用。
        - 请保持友好、专业的态度，一次只问一个问题，引导用户完成整个流程。"""
    )
    # 将系统提示添加到消息列表的最前面
    messages_with_prompt = [system_prompt] + messages
    response = llm_with_tools.invoke(messages_with_prompt)
    return {"messages": [response]}

# ------------------------------
# 步骤 4: 构建图
# ------------------------------

def build_agent_app():
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(tools)

    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph


# ------------------------------
# 步骤 5: 更新交互式驱动
# ------------------------------

def run_agent_dialog():
    thread_id = "demo-agent-user-002"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 使用内存来支持多轮对话
    memory = MemorySaver()
    app = build_agent_app().compile(checkpointer=memory)

      # 可视化图
    graph_png = app.get_graph().draw_mermaid_png()
    with open("多智能体模拟圆桌派综艺节目.png", "wb") as f:
        f.write(graph_png)

    print("您好！我是惠企政策顾问，请输入您的问题。")
    print("例如：'我是xxx公司，能享受高新技术企业认定补贴吗？'")

    while True:
        user_input = input("用户: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # 将用户输入作为 list[BaseMessage] 传入
        inputs = {"messages": [("user", user_input)]}
        
        # 使用 stream 模式，可以观察到每一步的输出
        for event in app.stream(inputs, config=config, stream_mode="values"):
            messages: list[BaseMessage] = event["messages"]
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage) and last_message.content:
                    # 这是模型的直接回复
                    print(f"顾问: {last_message.content}")
                # 添加对工具消息的打印，方便调试
                elif isinstance(last_message, ToolMessage):
                    print(f"--- [工具执行结果]: {last_message.name} ---\n{last_message.content}\n------------------------------------")

    # 导出可视化
    try:
        graph_png = app.get_graph().draw_mermaid_png()
        out_name = os.path.basename(__file__).replace(".py", "-agent.png")
        with open(out_name, "wb") as f:
            f.write(graph_png)
        print(f"\n[可视化] 已输出 {out_name}")
    except Exception as e:
        print(f"可视化失败: {e}")


if __name__ == "__main__":
    # 原有的基于规则的对话逻辑入口（保留，以便对比）
    # run_dialog(memory_mode="langgraph")

    # 启动新的基于 LLM Agent 的对话逻辑
    run_agent_dialog()


