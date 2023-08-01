import logging
import os
import time
import langchain
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatMessagePromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.cache import InMemoryCache
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Pinecone
import pinecone
from langchain.chains import RetrievalQA
from serpapi import GoogleSearch
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatMessagePromptTemplate, PromptTemplate

# 启动llm的缓存
langchain.llm_cache = InMemoryCache()
logging.basicConfig(level=logging.INFO)  # 日志
device = "cuda"

# go for a smaller model if you do not have the VRAM
model_id = r"E:\workspace\project\public\ChatGLM2-6B\THUDM\chatglm2-6b"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# # model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True)
# model = AutoModel.from_pretrained(model_id, load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, device=device)
# model.eval()

# 文本嵌入, 向量数据库。
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# # 文本嵌入模型接受文本输入输出浮点数列表
# text = "中国的首都是？"
# text_embedding = embeddings.embed_query(text)
# print(text_embedding)

pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=5000,
    # 自加的三个参数
    temperature=0.1,
    top_p=0.9,
    top_k=3,
)

local_llm_glm2 = HuggingFacePipeline(pipeline=pipe)

# template = "你是一个男科医学专家，为患者回答不同的男科疾病问题。{question}"
# prompt = PromptTemplate(
#     template=template,
#     input_variables=["question"],
# )
# chain_man = LLMChain(llm=local_llm_glm2, prompt=prompt)


# ChatMessagePromptTemplate 使用方法
# template = "May the {subject} be with you"
# chat_message_prompt = ChatMessagePromptTemplate.from_template(
#     role="Jedi",
#     template=template,
#     input_variables=["subject"],
# )
#
# # chat_message_prompt = PromptTemplate.from_template(template=prompt)
# # chat_message_prompt = PromptTemplate.from_template(template=prompt)
# res = chat_message_prompt.format(subject="force")
# print(res)

# 添加 Memory 的聊天机器人
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("你是一个男科医学专家，为患者回答不同的男科疾病问题。"),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}", input_variables=["input"]),
    ]
)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=local_llm_glm2)
while True:
    human = input("Human: ")
    print(human)
    if human == "exit":
        break
    start_time = time.perf_counter()
    answer = conversation.predict(input=human)
    # answer = chain_man.run(human)
    end_time = time.perf_counter()
    print("AI: ", answer)
    print("time cost: ", end_time - start_time)

# agent 使用
# os.environ[
#     "serpapi_api_key"
# ] = "eb3978daef457908db7bf92509321107c90d699f8e22f681f4522f2e9bc06963"
# tools = load_tools(
#     ["serpapi", "llm-math"],
#     llm=local_llm_glm2,
# )
# agent = initialize_agent(
#     tools, local_llm_glm2, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# )
# res = agent.run(
#     "What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?"
# )
# print(res)

# template = """Question: {question}
# Answer: Let's think step by step."""
# prompt = PromptTemplate(template=template, input_variables=["question"])
#
# template = "为{animal}起一个三个字的名字。"
# prompt = PromptTemplate(
#     input_variables=["animal"],
#     template=template,
# )

# template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
#
# Title: {title}
# Playwright: This is a synopsis for the above play:"""
# prompt = PromptTemplate(input_variables=["title"], template=template)
#
# print(prompt.format(animal="猫"))
# #
# examples = [
#     {"word": "高兴", "antonym": "悲伤"},
#     {"word": "高大", "antonym": "矮小"},
# ]
#
# example_template = """
# 词语: {word}
# 反义词: {antonym}\n
# """
#
# example_prompt = PromptTemplate(
#     input_variables=["word", "antonym"],
#     template=example_template,
# )
#
# few_shot_prompt = FewShotPromptTemplate(
#     examples=examples,
#     example_prompt=example_prompt,
#     prefix="给出输入词语的反义词",
#     suffix="词语: {input}\n反义词:",
#     input_variables=["input"],
#     example_separator="\n",
# )
#
# print(few_shot_prompt.format(input="美丽"))
# llm_chain = LLMChain(llm=local_llm_glm2, prompt=few_shot_prompt)
# res = llm_chain.run("王荣小")
# print(res)

# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=[])
# print(response)

# question = "What is the capital of France? "
# print(local_llm_glm2(question))
# llm_chain = LLMChain(llm=local_llm_glm2, prompt=prompt)
#
# # 创建第二个chain
# # second_prompt = PromptTemplate(
# #     input_variables=["petname"],
# #     template="从{petname}中选一个名字，写一篇小诗。",
# # )
# template = """You are a play critic from the New York Times. Given the synopsis of play,
# it is your job to write a review for that play.
#
# Play Synopsis:
# {synopsis}
# Review from a New York Times play critic of the above play:"""
# second_prompt = PromptTemplate(input_variables=["synopsis"], template=template)
#
# llm_chain_two = LLMChain(llm=local_llm_glm2, prompt=second_prompt)
# # 将两个chain串联在一起
# overall_chain = SimpleSequentialChain(chains=[llm_chain, llm_chain_two], verbose=True)
# # # 只需给出源头输入结合顺序运行整个chain
# # catchphrase = overall_chain.run("猫")
# # print(catchphrase)
#
# # index 链接外部数据
# # loader = TextLoader(r"data/testSet_1000.txt", encoding="utf8")
# # documents = loader.load()
# # 如果文本很大，可以使用 CharacterTextSplitter 对文档进行分割
# # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
# # texts = text_splitter.split_documents(documents)  # 元组，下表访问
# # print(texts[0].page_content)
# # print(len(texts))
# # # 创建 vectorestore 用作索引
# # db = FAISS.from_documents([texts[0]], embeddings)
#
# # 基于 index 设置带有检索功能的问答机器人
# # retriever = db.as_retriever()
# # qa = RetrievalQA.from_chain_type(
# #     llm=local_llm_glm2,
# #     chain_type="stuff",
# #     retriever=retriever,
# #     return_source_documents=True,
# # )
# # query = "男，目前51岁，近半年，发现，房事大不如前，此外，不足2,3分钟就射了，请问：男性早泄是由哪些方面引发的?"
# # result = qa({"query": query})
# # print(result["result"])
#
# while True:
#     # question = "What is the capital of England?"
#     animal = input("Human:")
#
#     begin_time = time.perf_counter()
#     # 请求模型
#     # res = llm_chain.run(animal)
#     res = overall_chain.run(animal)
#
#     end_time = time.perf_counter()
#     used_time = round(end_time - begin_time, 3)
#     logging.info(f"GLM2 process time: {used_time}")
#
#     print("GLM2 answer: ", res)
