
# from openai import OpenAI

# client = OpenAI(
#     api_key="bc34909d-af9a-42e5-b483-ce4dd246f368", # ModelScope Token
#     base_url="https://api-inference.modelscope.cn/v1"
# )

# # response = client.chat.completions.create(
# #     model="Qwen/Qwen2.5-Coder-32B-Instruct", # ModleScope Model-Id
# #     messages=[
# #         {
# #             'role': 'system',
# #             'content': 'You are a helpful assistant.'
# #         },
# #         {
# #             'role': 'user',
# #             'content': '用python写一下快排'
# #         }
# #     ],
# #     stream=True
# # )

# # for chunk in response:
# #     print(chunk.choices[0].delta.content, end='', flush=True)
    
# response = client.completions.create(
#     model="Qwen/Qwen2.5-Coder-32B-Instruct",
#     prompt="用python写一下快排",
#     stream=False
# )

# print(response.choices[0].text)
import os
os.environ["MODELSCOPE_SDK_TOKEN"] = "bc34909d-af9a-42e5-b483-ce4dd246f368"
import asyncio
from langchain_community.llms.modelscope_endpoint import ModelScopeEndpoint
from langchain_community.chat_models.modelscope_endpoint import ModelScopeChatEndpoint
from langchain_community.llms.modelscope_pipeline import ModelScopePipeline
from libs.community.tests.integration_tests.llms.test_modelscope_endpoint import *
from libs.community.tests.integration_tests.chat_models.test_modelscope_chat_endpoint import *
chat = ModelScopeChatEndpoint(modelscope_sdk_token="bc34909d-af9a-42e5-b483-ce4dd246f368", model="Qwen/Qwen2.5-Coder-32B-Instruct", streaming=True)
llm = ModelScopeEndpoint(modelscope_sdk_token="bc34909d-af9a-42e5-b483-ce4dd246f368", model="Qwen/Qwen2.5-Coder-32B-Instruct")

def test_llm():
    print(llm.invoke("用python写一下快排"))
    
def test_llm_async():
    print(asyncio.run(llm.ainvoke("用python写一下快排")))

def test_llm_stream():
    for chunk in llm.stream("用python写一下快排"):
        print(chunk, end='', flush=True)

async def test_llm_astream():
    async for chunk in llm.astream("用python写一下快排"):
        print(chunk, end='', flush=True)
    
def test_chat_stream():
    for chunk in chat.stream("用python写一下快排"):
        print(chunk.content, end='', flush=True)

async def test_chat_async():

    async for chunk in chat.astream("用python写一下快排"):
        print(chunk.content, end='', flush=True)

def test_modelscope_pipeline():
    llm = ModelScopePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="chat",
        generate_kwargs={'do_sample': True, 'max_new_tokens': 128},
    )
    print(llm.invoke("Hello, how are you?"))
    for chunk in llm.stream("Hello, how are you?"):
        print(chunk, end='', flush=True)

def test_pipeline():
    from modelscope import pipeline
    p = pipeline(
            task="chat",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            device_map="auto",
            llm_framework="swift",
        )
    print(p("Hello, how are you?"))

if __name__ == "__main__":
    # test_pipeline()
    test_modelscope_call()
    test_modelscope_streaming()
    asyncio.run(test_modelscope_call_async())
    asyncio.run(test_modelscope_streaming_async())

    test_modelscope_chat_call()
    test_modelscope_chat_multiple_history()
    test_modelscope_chat_stream()
    
