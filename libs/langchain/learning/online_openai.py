import os
import openai
import asyncio
import time
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# openai.api_key = "6RbtwdbDopJ8gAsIpKI6VZzi9PGufk41PwxsNLxBbUg"
# openai.api_base = "https://chimeragpt.adventblocks.cc/api/v1"
os.environ["OPENAI_API_KEY"] = "6RbtwdbDopJ8gAsIpKI6VZzi9PGufk41PwxsNLxBbUg"
os.environ["OPENAI_API_BASE"] = "https://chimeragpt.adventblocks.cc/api/v1"
# llm = OpenAI(temperature=0.9)
# text = "What would be a good company name for a company that makes water? plase give me three names, and tell me why?"
# print(llm(text))
chat = ChatOpenAI(temperature=0.6)


# Async 异步调用不同的 LLM
def generate_serially():
    llm = OpenAI(temperature=0.9)
    for _ in range(10):
        resp = llm.generate(["Hello, how are you?"])
        # print(resp)
        # print(type(resp))
        print(resp.generations[0][0].text)
        # print(resp.generations)
        # print(type(resp.generations))


async def async_generate(llm):
    resp = await llm.agenerate(["Hello, how are you?"])
    print(resp.generations[0][0].text)


async def generate_concurrently():
    llm = OpenAI(temperature=0.9)
    tasks = [async_generate(llm) for _ in range(10)]
    await asyncio.gather(*tasks)


s = time.perf_counter()
asyncio.run(generate_concurrently())
elapsed = time.perf_counter() - s
print("\033[1m" + f"Concurrent executed in {elapsed:0.2f} seconds." + "\033[0m")

print("================================================================")
s = time.perf_counter()
generate_serially()
elapsed = time.perf_counter() - s
print("\033[1m" + f"Serial executed in {elapsed:0.2f} seconds." + "\033[0m")
# res = chat(
#     [
#         HumanMessage(
#             content="Translate this sentence from English to French. I love programming."
#         )
#     ]
# )
# messages = [
#     SystemMessage(
#         content="You are a helpful assistant that translates English to French."
#     ),
#     HumanMessage(
#         content="Translate this sentence from English to French. I love programming."
#     ),
# ]3
# res = chat(messages)
# batch_messages = [
#     [
#         SystemMessage(
#             content="You are a helpful assistant that translates English to French."
#         ),
#         HumanMessage(
#             content="Translate this sentence from English to French. I love programming."
#         ),
#     ],
#     [
#         SystemMessage(
#             content="You are a helpful assistant that translates English to French."
#         ),
#         HumanMessage(
#             content="Translate this sentence from English to French. I love artificial intelligence."
#         ),
#     ],
# ]
# res = chat.generate(batch_messages)
#
# # -> LLMResult(generations=[[ChatGeneration(text="J'aime programmer.", generation_info=None, message=AIMessage(content="J'aime programmer.", additional_kwargs={}))], [ChatGeneration(text="J'aime l'intelligence artificielle.", generation_info=None, message=AIMessage(content="J'aime l'intelligence artificielle.", additional_kwargs={}))]], llm_output={'token_usage': {'prompt_tokens': 71, 'completion_tokens': 18, 'total_tokens': 89}})
#
# # -> AIMessage(content="J'aime programmer.", additional_kwargs={})
# print(res)
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
# response = openai.ChatCompletion.create(
#     model="gpt-4",
#     messages=[
#         {"role": "user", "content": "Hello"},
#     ],
# )
#
# print(response)

# openai.api_key = "your API Key Here"
# openai.api_base = "https://chimeragpt.adventblocks.cc/v1"
#
# response = openai.ChatCompletion.create(
#     model="gpt-4",
#     messages=[
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=True,
# )
#
# for chunk in response:
#     try:
#         print(chunk.choices[0].delta.content, end="", flush=True)
#     except:
#         break
