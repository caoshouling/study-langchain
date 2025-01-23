from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ]
)
message = prompt.invoke(
   {
       "history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")],
       "question": "now multiply that by 4"
   }
)
print(message)