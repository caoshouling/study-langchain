import logging
from typing import Dict, Any, List, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)
print('--------------------------基本使用-------------------------')
prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文：{topic}")


class MyCustomHandler(BaseCallbackHandler):

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print("""----------Run when LLM starts running.----------""")

    def on_chat_model_start(
            self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        print("""----------Run when Chat Model starts running.--------------------""")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        print("""----------Run on new LLM token. Only available when streaming is enabled.----------""")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print("""----------Run when LLM ends running.----------""")

    def on_llm_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        print("""Run when LLM errors.""")

    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        print("""----------Run when chain starts running.----------""")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        print("""----------Run when chain ends running.----------""")

    def on_chain_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        print("""----------Run when chain errors.----------""")

    def on_tool_start(
            self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        print("""----------Run when tool starts running.----------""")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        print("""----------Run when tool ends running.----------""")

    def on_tool_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        print("""----------Run when tool errors.----------""")

    def on_text(self, text: str, **kwargs: Any) -> Any:
        print("""----------Run on arbitrary text.----------""")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        print("""----------Run on agent action.----------""")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        print("""----------Run on agent end.----------""")
llm_base_url: str = "http://localhost:1234/v1/"
llm_model: str = "paultimothymooney/qwen2.5-7b-instruct"
llm_base_url="https://8f13-154-12-181-41.ngrok-free.app/v1/"
# 初始化语言模型
llm = ChatOpenAI(
    openai_api_base=llm_base_url,
    model=llm_model,
   max_tokens=25,
api_key="fsdf",
   callbacks=[MyCustomHandler()]
)
chain = prompt|llm| StrOutputParser()
answer = chain.invoke({"product","苹果"})
print(answer)
