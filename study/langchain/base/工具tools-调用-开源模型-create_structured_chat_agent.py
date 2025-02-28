import logging
import time

from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor, create_react_agent, \
    create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug, set_verbose
logging.basicConfig(level=logging.DEBUG)
set_debug(True)
set_verbose(True)

# os.environ["LANGSMITH_API_KEY"] = "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"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGSMITH_PROJECT"] = "langsmith-basic"
# os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
@tool
def get_weather(city):
    """根据城市获取天气数据"""
    ## 睡眠1秒，模拟网络请求
    time.sleep(1)
    return f"今天{city}的天气白天是晴天，晚上下雨"
@tool
def multiply(a: int, b: int) -> int:
    """计算两个数字相乘"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """计算两个数字相加"""
    return a + b

tools = [add,multiply,get_weather]

print(get_weather.invoke("南京"))
print("---------------------千问大模型--------------------------")
# 定义提示词

# llm = ChatOpenAI(openai_api_base="https://api.deepseek.com/v1",
#                    model_name='deepseek-chat',
#                  openai_api_key ="sk-37b28ddd69354f6e8f813de3a0f218f2")

# llm_base_url: str = "http://localhost:1234/v1/"
# 初始化语言模型
llm = ChatOpenAI(
    openai_api_base="https://8f13-154-12-181-41.ngrok-free.app/v1/",
    model="paultimothymooney/qwen2.5-7b-instruct",
    api_key ="323")
prompt_muti_param = hub.pull("hwchase17/structured-chat-agent")
agent= create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_muti_param
)

agent_executor =AgentExecutor(agent=agent, tools=tools,verbose=True)

# 小明有120只鸭子，后面朋友送了他99只鸭子，他一共有多少只鸭子？
# 今天南京的天气怎样？
result = agent_executor.invoke({"input": "张三在哪所学校毕业的，学的什么专业? 小明有120只鸭子，后面朋友送了他99只鸭子，他一共有多少只鸭子？"})
print(result)



