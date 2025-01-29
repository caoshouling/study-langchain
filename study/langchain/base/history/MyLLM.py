from langchain_openai import ChatOpenAI

llm_model: str = "paultimothymooney/qwen2.5-7b-instruct"
llm_base_url="https://8f13-154-12-181-41.ngrok-free.app/v1/"

# 初始化语言模型
myllm = ChatOpenAI(
    openai_api_base=llm_base_url,
    model=llm_model,
    api_key="fsdf",
)