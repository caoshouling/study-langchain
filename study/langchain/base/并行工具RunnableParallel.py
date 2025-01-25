from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

model = ChatOpenAI(openai_api_base="http://localhost:1234/v1/",
                   model="paultimothymooney/qwen2.5-7b-instruct")
joke_chain = (
    ChatPromptTemplate.from_template("tell me a joke about {topic}")
    | model
)
poem_chain = (
    ChatPromptTemplate.from_template("write a 2-line poem about {topic2}")
    | model
)

runnable = RunnableParallel(joke=joke_chain, poem=poem_chain)

print(runnable)
print('-----------------------')
result = runnable.invoke({"topic":"买车", "topic2":"买苹果"})

print(result)
"""
返回的result如下结构：是一个字典
{
'joke': AIMessage(content='当然可以，这里有一个关于买车的笑话：\n\n为什么2019款的车都不开心？\n\n因为它们都在为2020款让路！（play on words, "让路" sounds like "让位", implying giving way to the new models）\n\n希望这个笑话您会喜欢！', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 63, 'prompt_tokens': 36, 'total_tokens': 99, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'paultimothymooney/qwen2.5-7b-instruct', 'system_fingerprint': 'paultimothymooney/qwen2.5-7b-instruct', 'finish_reason': 'stop', 'logprobs': None}, id='run-07963b84-0e54-4f7b-8971-f7a7c7634dbf-0', usage_metadata={'input_tokens': 36, 'output_tokens': 63, 'total_tokens': 99, 'input_token_details': {}, 'output_token_details': {}}), 
'poem': AIMessage(content='买来青苹果，  \n滋味未尝多。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 39, 'total_tokens': 50, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'paultimothymooney/qwen2.5-7b-instruct', 'system_fingerprint': 'paultimothymooney/qwen2.5-7b-instruct', 'finish_reason': 'stop', 'logprobs': None}, id='run-44f71011-b095-468d-9b95-6ecd8f3ab4ce-0', usage_metadata={'input_tokens': 39, 'output_tokens': 11, 'total_tokens': 50, 'input_token_details': {}, 'output_token_details': {}})
}

"""
# 遍历 result 字典
for key, value in result.items():
    print(f"{key}: {value}")
