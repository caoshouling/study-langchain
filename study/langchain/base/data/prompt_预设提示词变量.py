from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant named {name}."),
        ("human", "Hi I'm {user1}"),
        ("ai", "Hi there, {user2}, I'm {name}."),
        ("human", "{input}"),
    ]
)
# 预先填写一部分变量
template2 = template.partial(user1="Lucy", name="R2D2")
print(template2)

message = template2.format_messages(input="hello",user2="张三")
print(message)

