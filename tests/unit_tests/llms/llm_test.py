import asyncio

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def generate_serially():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    for _ in range(10):
        resp = chain.run(product="toothpaste")
        print(resp)


async def async_generate(llm):
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    resp = await chain.arun(product="toothpaste")
    print(resp)


async def generate_concurrently():
    llm = OpenAI(temperature=0.9)
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
