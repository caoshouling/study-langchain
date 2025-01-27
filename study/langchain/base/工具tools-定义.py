import time
from langchain_core.tools import tool

import asyncio
print('---------------------用@tool定义工具-基本函数--------------------')
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
#
print('---------------------用@tool定义工具-异步函数--------------------')
@tool
async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

print(multiply.invoke({"a": 2, "b": 3}))

print(amultiply)

print(amultiply.name)
print(amultiply.description)
print(amultiply.args)

async def main():
    print(await amultiply.invoke({"a": 2, "b": 3}))



# ，
print('---------------------用@tool定义工具-传递对象参数--------------------')
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Let's inspect some of the attributes associated with the tool.
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.return_direct)

print('---------------------用StructuredTool定义工具 --------------------')

from langchain_core.tools import StructuredTool


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(func=multiply, coroutine=amultiply)

print(calculator.invoke({"a": 2, "b": 3}))
# print(await calculator.ainvoke({"a": 2, "b": 5}))
print('---------------------用StructuredTool定义工具 - 传递对象参数 --------------------')

class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(
    func=multiply,
    name="Calculator",
    description="multiply numbers",
    args_schema=CalculatorInput,
    return_direct=True,
    # coroutine= ... <- you can specify an async method if desired as well
)

print(calculator.invoke({"a": 2, "b": 3}))
print(calculator.name)
print(calculator.description)
print(calculator.args)

print('---------------------用StructuredTool定义工具 - 异步工具 --------------------')

from langchain_core.tools import StructuredTool


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# func是同步工具，对应invoke调用方法   coroutine是异步工具，对应ainvoke调用方法
calculator = StructuredTool.from_function(func=multiply, coroutine=amultiply)

print(calculator.invoke({"a": 2, "b": 3}))
# print(
#     await calculator.ainvoke({"a": 2, "b": 5})
# )  # Uses use provided amultiply without additional overhead


print('---------------------异常处理 -handle_tool_error=True直接返回异常文本-------------------')
from langchain_core.tools import ToolException


def get_weather(city: str) -> int:
    """Get weather for the given city."""
    raise ToolException(f"Error: There is no city by the name of {city}.")

#### 如果 handle_tool_error=False，那么会抛出异常
get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error=True,
)
# 返回：'Error: There is no city by the name of foobar.'文本
result = get_weather_tool.invoke({"city": "foobar"})
print(result)

print('---------------------异常处理 - handle_tool_error= 自定义异常信息 --------------------')
get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error="There is no such city, but it's probably above 0K there!",
)
# 返回文本："There is no such city, but it's probably above 0K there!"
result = get_weather_tool.invoke({"city": "foobar"})
print(result)
print('---------------------异常处理 - handle_tool_error= 设置函数，从函数中返回 --------------------')
def _handle_error(error: ToolException) -> str:
    return f"The following errors occurred during tool execution: `{error.args[0]}`"


get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error=_handle_error,
)

result = get_weather_tool.invoke({"city": "foobar"})
print(result)