from langchain.llms import OpenAI
import asyncio


def generate_serially():
    llm = OpenAI(temperature=0)
    for _ in range(10):
        resp = llm.generate(["Hello, how are you?"])
        # print(resp)


async def async_generate(llm):
    resp = await llm.async_generate(["Hello, how are you?"])
    # print(resp)


async def generate_concurrently():
    llm = OpenAI(temperature=0)
    tasks = [async_generate(llm) for _ in range(10)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    import time
    s = time.perf_counter()
    asyncio.run(generate_concurrently())
    elapsed = time.perf_counter() - s
    print(f"Concurrent executed in {elapsed:0.2f} seconds.")

    s = time.perf_counter()
    generate_serially()
    elapsed = time.perf_counter() - s
    print(f"Serial executed in {elapsed:0.2f} seconds.")
