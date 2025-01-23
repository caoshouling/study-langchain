# pip install langchain
# pip install -qU langchain-openai
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
base_url="http://localhost:1234/v1/"
print("============开始==1=========")
model = ChatOpenAI(openai_api_base=base_url,
                   model="paultimothymooney/qwen2.5-7b-instruct")
result = model.invoke("您好")
print(result.content)

print("============开始==2=========")
prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文：{topic}")
output_parser = StrOutputParser()


chain = prompt | model | output_parser
result_str = chain.invoke({"topic", "康师傅绿茶"})
print(result_str)

with get_openai_callback() as cb:
    result = chain.invoke({"topic", "康师傅绿茶"})
    print(result)
    print(cb)
#

#
# """
#   普通方式
# """
#
#
# def get_response(prompt):
#     """
#     获取AI助手的回复
#
#     Args:
#         prompt (str): 用户输入的问题
#
#     Returns:
#         str: AI助手的回复内容
#     """
#     try:
#         completion = client.chat.completions.create(
#             model="qwen2.5-7b-instruct",
#             messages=[
#                 {'role': 'system',
#                  'content': '你是一个AI助手，专门回答新员工关心的公司规章制度，你的名字叫做贾维斯，请你用中文回答。'},
#                 {'role': 'user', 'content': prompt}
#             ]
#         )
#         print(completion.choices[0].message.content)
#         return completion.choices[0].message.content
#     except Exception as e:
#         return f"错误信息：{e}\n请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code"
#
#
# '''
#   流式生成：加入参数：stream=True
# '''
# from datetime import datetime
#
#
# def stream_generate(prompt):
#     """流式生成函数"""
#     print("用户输入："+prompt)
#     start_time = datetime.now()
#     try:
#         stream = client.chat.completions.create(
#             model="qwen2.5-7b-instruct",
#             messages=[
#                 {"role": "system", "content": "你是一位知心大姐，用温柔的预期帮人解答困惑。"},
#                 {"role": "user", "content": prompt}
#             ],
#             stream=True  # 流式生成
#         )
#
#         print("\n助手:")
#         for chunk in stream:
#             if chunk.choices[0].delta.content is not None:
#                 print(chunk.choices[0].delta.content, end="")
#
#     except Exception as e:
#         print(f"流式生成错误：{e}")
#
#     end_time = datetime.now()
#     print(f"\n流式生成耗时: {(end_time - start_time).total_seconds():.2f}秒")
#
#
# if __name__ == '__main__':
#     # # 使用示例:
#     # response = get_response("公司的年假是怎么规定的？")
#     # print(response)
#     # # 使用示例:
#     print("------开始------")
#     while True:
#         promopt = input("请输入：")
#         if promopt == 'exit':
#             break
#         stream_generate(promopt)
#     print("------结束------")
#

