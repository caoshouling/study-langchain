from langchain import hub
'''
这些提示词的仓库位于：https://smith.langchain.com/hub/


'''



print('-----------------openai-functions-agent----------------------')
prompt = hub.pull("hwchase17/openai-functions-agent")
print(prompt)

print('-----------------openai-tools-agent----------------------')
prompt = hub.pull("hwchase17/openai-tools-agent")
print(prompt)

print('-----------------self-ask-with-search----------------------')
prompt = hub.pull("hwchase17/self-ask-with-search")
print(prompt)

print('-----------------structured-chat-agent----------------------')
prompt = hub.pull("hwchase17/structured-chat-agent")
print(prompt)

print('-----------------react----------------------')
prompt = hub.pull("hwchase17/react")
print(prompt)