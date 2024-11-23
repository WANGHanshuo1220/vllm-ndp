import asyncio

async def func_a():
    print("Start func_a")
    await asyncio.sleep(1)  # 模拟异步操作
    print("Before yield in func_a")
    yield "Value from func_a"  # 暂停并返回值
    print("After yield in func_a")  # 继续执行

async def func_b():
    print("Start func_b")
    async for ret in func_a():
        print(f"Received from func_a: {ret}")
    print("func_b continues after await")

# 运行事件循环
asyncio.run(func_b())

