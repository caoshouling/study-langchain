# pip install langchain
# pip install -qU langchain-openai
import asyncio

from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import openai


'''
  本页介绍了两种调用方式：
方式一：使用 langchain 库中的 ChatOpenAI 类，封装了对 OpenAI API 的调用。
方式二：直接使用 openai 库中的 ChatCompletion.create 方法。

两种方式都支持开源模型，不仅仅针对ChatGPT


'''

base_url="http://localhost:1234/v1/"
base_url="https://8f13-154-12-181-41.ngrok-free.app/v1/"
model_name="paultimothymooney/qwen2.5-7b-instruct"
model = ChatOpenAI(openai_api_base=base_url,
                       model=model_name)
def invoke():
    print("============开始==1=========")
    result = model.invoke("您好")
    '''
    content='您好！很高兴为您服务。有什么问题或需要帮助的吗？' 
    additional_kwargs={'refusal': None} 
    response_metadata={
        'token_usage': {
            'completion_tokens': 14, 
            'prompt_tokens': 30, 
            'total_tokens': 44, 
            'completion_tokens_details': None, 
            'prompt_tokens_details': None
        }, 
        'model_name': 'paultimothymooney/qwen2.5-7b-instruct', 
        'system_fingerprint': 'paultimothymooney/qwen2.5-7b-instruct', 
        'finish_reason': 'stop', 
        'logprobs': None
    } 
    id='run-60a06105-34b9-4f75-b392-ea5f35a8b890-0' 
    usage_metadata={
        'input_tokens': 30, 
        'output_tokens': 14, 
        'total_tokens': 44, 
        'input_token_details': {}, 
        'output_token_details': {}
    }

    '''
    print(result)

def invokeLCEL():
    print("============开始==2=========")
    prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文：{topic}")
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser
    result_str = chain.invoke({"topic", "康师傅绿茶"})
    # 返回的是字符串str
    print(type(result_str))
    #  打印token
    with get_openai_callback() as cb:
        result = chain.invoke({"topic", "康师傅绿茶"})
        print(result)
        #  打印token
        print(cb)
    #

# 流式调用
def invokeStream():
    prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文：{topic}")
    output_parser = StrOutputParser()

    for chunk in model.stream("您好"):
        print(chunk.content, end="|", flush=True)

# 异步流式调用, 调用方式：asyncio.run(model_astream())
async def model_astream():
    async for chunk in model.astream("Write me a 1 verse song about goldfish on the moon"):
        print(chunk.content, end="|", flush=True)

"""
  普通方式
"""

# 创建客户端对象
client = openai.Client(base_url=base_url, api_key="223")
def get_response(prompt):
    """
    获取AI助手的回复

    Args:
        prompt (str): 用户输入的问题

    Returns:
        str: AI助手的回复内容
    """
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system',
                 'content': '你是一个AI助手，专门回答新员工关心的公司规章制度，你的名字叫做贾维斯，请你用中文回答。'},
                {'role': 'user', 'content': prompt}
            ]
        )
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content
    except Exception as e:
        return f"错误信息：{e}\n请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code"


'''
  流式生成：加入参数：stream=True
'''
from datetime import datetime


def stream_generate(prompt):
    """流式生成函数"""
    print("用户输入："+prompt)
    start_time = datetime.now()
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一位知心大姐，用温柔的预期帮人解答困惑。"},
                {"role": "user", "content": prompt}
            ],
            stream=True  # 流式生成
        )

        print("\n助手:")
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")

    except Exception as e:
        print(f"流式生成错误：{e}")

    end_time = datetime.now()
    print(f"\n流式生成耗时: {(end_time - start_time).total_seconds():.2f}秒")


if __name__ == '__main__':
    invokeStream()
    # # 使用示例:
    # response = get_response("公司的年假是怎么规定的？")
    # print(response)

    # print("------开始------")
    # while True:
    #     promopt = input("请输入：")
    #     if promopt == 'exit':
    #         break
    #     stream_generate(promopt)
    # print("------结束------")




