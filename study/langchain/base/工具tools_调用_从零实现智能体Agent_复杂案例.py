import os
import time
from decimal import Decimal, localcontext
from langchain_core.tools import  tool
from langchain_openai import ChatOpenAI


from study.langchain.base.工具tools_调用_从零实现智能体Agent import run_agent



@tool
def get_sales(city: str) -> float:
    """根据地区获取销售额"""

    time.sleep(1)
    return 129.9
@tool
def get_cost(city: str) -> float:
    """根据地区获取销售成本"""

    time.sleep(1)
    return 100.0
@tool
def get_profit(city:str,sales: float, cost: float) -> float:
    """根据地区、销售金额、销售成本计算增长值。"""

    time.sleep(1)
    return sales - cost  - 10

@tool
def subtract_float(amount1: float, amount2: float) -> Decimal:
    """
    计算两个数字的差值。
    """
    # 将 float 转换为 Decimal
    decimal_amount1 = Decimal(str(amount1))
    decimal_amount2 = Decimal(str(amount2))

    # 使用局部上下文设置精度
    with localcontext() as ctx:
        ctx.prec = 10  # 设置局部精度
        result = decimal_amount1 - decimal_amount2

    return result
@tool
def add_float(amount1: float, amount2: float) -> Decimal:
    """
    计算两个数字的和。
    """
    # 将 float 转换为 Decimal
    decimal_amount1 = Decimal(str(amount1))
    decimal_amount2 = Decimal(str(amount2))

    # 使用局部上下文设置精度
    with localcontext() as ctx:
        ctx.prec = 10  # 设置局部精度
        result = decimal_amount1 + decimal_amount2

    return result


if __name__ == "__main__":
    '''
    这道题目前架构算不出来，没有进一步分解get_sales和get_cost。
    调用函数传参有问题：get_profit
        只传了city 没有传sales和cost
    '''

    llm = ChatOpenAI(openai_api_base= "https://dashscope.aliyuncs.com/compatible-mode/v1",
                       model_name='qwen-turbo',
                     api_key ="sk-d9ca67dd361c4347b582386197867c05")
    tools = [get_sales,get_cost,add_float,subtract_float,get_profit]
    result = run_agent( "南京地区的增长值是多少？",llm,tools)
    print(result)