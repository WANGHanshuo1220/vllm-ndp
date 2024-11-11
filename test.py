import asyncio
import threading
import time

# 共享数据结构
shared_data = []

# 创建锁，保护对共享数据的访问
lock = threading.Lock()

def infinite_loop():
    count = 0
    while True:
        time.sleep(1)  # 模拟阻塞的操作
        with lock:  # 保护对共享数据的访问
            shared_data.append(count)
            print(f"Thread added {count} to shared_data.")
            count += 1

async def main():
    # 启动一个线程，执行无限循环
    thread = threading.Thread(target=infinite_loop, daemon=True)  # 设置 daemon=True，确保程序结束时线程自动结束
    thread.start()
    
    # 给无限循环线程一些时间执行
    i = 0
    while True:
        with lock:  # 保护对共享数据的访问
            print(f"Async function {i} read shared_data: {shared_data}")
        i = i + 1
        time.sleep(2)


# 运行 async 函数
asyncio.run(main())
