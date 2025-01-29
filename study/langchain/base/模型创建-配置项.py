# pip install langchain
# pip install -qU langchain-openai
import logging

from langchain.chat_models import init_chat_model
from langchain.globals import set_debug, set_verbose
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import openai


'''
init_chat_model
    # Returns a langchain_openai.ChatOpenAI instance.
    gpt_4o = init_chat_model("gpt-4o", model_provider="openai", temperature=0)
    
    # Returns a langchain_anthropic.ChatAnthropic instance.
    claude_opus = init_chat_model(
        "claude-3-opus-20240229", model_provider="anthropic", temperature=0
    )
    
    # Returns a langchain_google_vertexai.ChatVertexAI instance.
    gemini_15 = init_chat_model(
        "gemini-1.5-pro", model_provider="google_vertexai", temperature=0
    )
    貌似只支持列出的这些：
        - 'openai'              -> langchain-openai
        - 'anthropic'           -> langchain-anthropic
        - 'azure_openai'        -> langchain-openai
        - 'google_vertexai'     -> langchain-google-vertexai
        - 'google_genai'        -> langchain-google-genai
        - 'bedrock'             -> langchain-aws
        - 'bedrock_converse'    -> langchain-aws
        - 'cohere'              -> langchain-cohere
        - 'fireworks'           -> langchain-fireworks
        - 'together'            -> langchain-together
        - 'mistralai'           -> langchain-mistralai
        - 'huggingface'         -> langchain-huggingface
        - 'groq'                -> langchain-groq
        - 'ollama'              -> langchain-ollama
        - 'google_anthropic_vertex'    -> langchain-google-vertexai

'''


def method_init_chat_model():
    print("============init_chat_model案例：基本用法=========")

    model_name = "gpt-3.5-turbo"
    model = init_chat_model(
        model_name,
        model_provider="openai",
        temperature=0
    )
    result = model.invoke("您好")
    print(result)

def method_init_chat_model2():
    print("============init_chat_model案例： 默认参数 +  调用时实时指定参数=========")

    model_name_default = "gpt-3.5-turbo"
    model_name_4o_mini = "gpt-4o-mini"
    first_llm = init_chat_model(
        model=model_name_default,
        temperature=0,
        configurable_fields=("model", "model_provider", "temperature", "max_tokens"),
        config_prefix="first",  # useful when you have a chain with multiple models
    )

    result =first_llm.invoke("what's your name")
    print("============ 默认参数 =========")
    print(result)
    result=first_llm.invoke(
        "what's your name",
        config={
            "configurable": {
                "first_model": model_name_4o_mini,
                "first_temperature": 0.5,
                "first_max_tokens": 100,
            }
        },
    )
    print("============调用时实时指定参数=========")
    print(result)

def method_configurable_fields():
    print("============configurable_fields案例：调用时实时指定参数=========")
    base_url = "http://localhost:1234/v1/"
    base_url="https://8f13-154-12-181-41.ngrok-free.app/v1/"
    model_name="paultimothymooney/qwen2.5-7b-instruct"
    model = ChatOpenAI(openai_api_base=base_url,
                       model=model_name,
                       temperature = 0
                       )
    model = model.configurable_fields(
        temperature=ConfigurableField(
            id="llm_temperature",
            name="LLM Temperature",
            description="The Temperature of the LLM"
        )
    )
    result = model.with_config(configurable={"llm_temperature": 0.9}).invoke("what's your name")
    print(result)

def method_configurable_alternatives():
    print("============configurable_alternatives案例：在不同模型配置间切换=========")
    from typing import Callable, Any
    model_name_default = "gpt-3.5-turbo"
    model_name_4o_mini = "gpt-4o-mini"
    # 创建基础模型
    gpt35= ChatOpenAI(temperature=0, model=model_name_default)
    # 定义不同的模型配置函数
    gpt40min= ChatOpenAI(temperature=0, model=model_name_4o_mini)

    
    # 创建一个可在不同配置间切换的模型
    configurable_model = gpt35.configurable_alternatives(
        which=ConfigurableField(
            id="model_type",
            name="Model Type",
            description="选择使用哪种类型的模型配置"
        ),
        gpt35= gpt35,
        gpt40min = gpt40min,
    )

    prompt = PromptTemplate.from_template(
        "讲一个关于 {topic}的笑话"
    ).configurable_alternatives(
        which=ConfigurableField(id="prompt"),
        default_key="joke", # 默认的prompt 其实就是当前的promot
        poem=PromptTemplate.from_template("写一首关于 {topic} 的诗"),
    )

    # 使用默认配置
    chain =prompt | configurable_model

    result1 = chain.with_config(configurable={"prompt": "poem", "model_type": "gpt35"}).invoke(
        {"topic": "苹果"}
    )
    print("使用默认配置的结果：")

    print(result1)
    
    # 切换到创意配置
    result2 = chain.with_config(
        configurable={"model_type": "gpt40min"}
    ).invoke( {"topic": "船长"})
    print("\n使用创意配置的结果：")
    print(result2)

if __name__ == '__main__':
    # 配置日志记录
    # logging.basicConfig(level=logging.DEBUG)
    set_debug(True)
    set_verbose(True)

    # method_configurable_fields()
    method_configurable_alternatives()




