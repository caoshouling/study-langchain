#!/usr/bin/env python


'''

同时安装客户端和服务端
pip install "langserve[all]"
也可以分别安装，pip install "langserve[client]"安装客户端，pip install "langserve[server]" 安装服务端。

'''

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate

app = FastAPI(
    title="LangChain 服务器",
    version="1.0",
    description="使用 Langchain 的 Runnable 接口的简单 API 服务器",
)
add_routes(
    app,
    ChatOpenAI(model="gpt-3.5-turbo"),
    path="/openai",
)

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
add_routes(
    app,
    ChatOpenAI(model="gpt-3.5-turbo") | parser,
    path="/openai_str_parser",
)

prompt = ChatPromptTemplate.from_template("告诉我一个关于 {topic} 的笑话")
add_routes(
    app,
    prompt | ChatOpenAI(model="gpt-4"),
    path="/openai_ext",
)

# 设置所有启用 CORS 的来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
