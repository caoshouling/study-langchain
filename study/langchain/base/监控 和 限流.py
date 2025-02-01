'''
监控：
  # 配置日志记录
    logging.basicConfig(level=logging.DEBUG)
    set_debug(True)
    set_verbose(True)

  # LangSmith

'''
import asyncio
import time
from datetime import datetime

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
'''
  max_bucket_size 是最大并发数，也就是总令牌数
  requests_per_second 每秒产生多少个令牌，用来限制请求的频率
  这里其实少了一个等待队列的机制
'''
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,  # <--  非常慢！，10秒一次。Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # 100毫秒检查一次。Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=3,  # 最大桶大小Controls the maximum burst size.，可以理解为同时最多可以有3个请求，超过3个请求，则进行等待，直到有请求完成，获取到令牌，然后继续。
)
base_url = "http://localhost:1234/v1/"
base_url = "https://8f13-154-12-181-41.ngrok-free.app/v1/"
model_name = "paultimothymooney/qwen2.5-7b-instruct"
model = ChatOpenAI(openai_api_base=base_url,
                   model=model_name,
                   temperature=0,
                   rate_limiter=rate_limiter
                   )

async def model_astream():
    async for chunk in model.astream("Write me a 1 verse song about goldfish on the moon"):
        print(chunk.content, end="|", flush=True)

# 基本测试
def testInvoke():
    for _ in range(5):
        tic = time.time()

        # for chunk in model.invo("您好"):
        #     print(chunk.content, end="|", flush=True)

        asyncio.run(model_astream())

        toc = time.time()
        print(toc - tic)

        time.sleep(1)

def test_max_bucket_size():
    """
    使用同步方式测试max_bucket_size的效果
    这个测试将清晰地展示令牌桶的工作原理：
    - 最多允许max_bucket_size个请求同时发送
    - 之后的请求将被限流，每秒只允许requests_per_second个请求
    """
    print(f"\n开始测试max_bucket_size效果...")
    print(f"配置: max_bucket_size={rate_limiter.max_bucket_size}, requests_per_second={rate_limiter.requests_per_second}")
    
    for i in range(6):  # 发送6个请求，超过max_bucket_size=3
        start_time = time.time()
        print(f"\n发送请求 {i}, 开始时间: {datetime.now().strftime('%H:%M:%S.%f')}")
        
        response = model.invoke("简单回复ok即可")
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"请求 {i} 完成, 耗时: {duration:.2f}秒, 结束时间: {datetime.now().strftime('%H:%M:%S.%f')}")
        print(f"响应内容: {response.content}")

async def async_request(i: int):
    """单个异步请求"""
    start_time = time.time()
    print(f"\n发送请求 {i}, 开始时间: {datetime.now().strftime('%H:%M:%S.%f')}")
    
    response = await model.ainvoke("简单回复ok即可")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"请求 {i} 完成, 耗时: {duration:.2f}秒, 结束时间: {datetime.now().strftime('%H:%M:%S.%f')}")
    print(f"响应内容: {response.content}")

async def test_concurrent_requests():
    """
    使用异步方式测试真正的并发限流效果
    - 同时发起6个请求
    - 由于max_bucket_size=3，前3个请求会立即执行
    - 后续请求需要等待令牌生成（每2秒1个令牌，因为requests_per_second=0.5）
    """
    print(f"\n开始并发测试...")
    print(f"配置: max_bucket_size={rate_limiter.max_bucket_size}, requests_per_second={rate_limiter.requests_per_second}")
    
    # 创建6个并发任务
    tasks = [async_request(i) for i in range(6)]
    # 同时启动所有任务
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    # test_max_bucket_size()  # 注释掉旧的同步测试
    asyncio.run(test_concurrent_requests())  # 运行新的并发测试