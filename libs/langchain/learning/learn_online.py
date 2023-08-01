from langchain.llms import HuggingFacePipeline
from langchain.llms import HuggingFaceHub
import os
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain

# read
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "xrNJmp__x6Jf0jWIDsz0kMODyJCn3W34hnUIT0vOGbo"
os.environ["HUGGINGFACEHUB_API_BASE"] = "https://api.huggingface.co"
# 问答
# write
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# llm = HuggingFaceHub(repo_id="baichuan-inc/Baichuan-13B-Chat")
llm = HuggingFaceHub(repo_id="gpt2")
# while True:
#     query = input("Human: ")
#     print("AI: ", llm(query))

# 嵌入
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/msmarco-MiniLM-L-12-v3"
# )
#
# text = "中国的首都是？"
# text_embedding = embeddings.embed_query(text)
# print(text_embedding)

# Zero shot 提示

template = "请为宠物{animal}起一个好听的名字。"

prompt = PromptTemplate(
    input_variables=["animal"],
    template=template,
)

# res = prompt.format(animal="猫")
# print("prompt: ", res)

# # few shot 提示
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
# res = few_shot_prompt.format(input="美丽")
# print(res)

# chain 的使用
chain = LLMChain(llm=llm, prompt=prompt)

print("chain 开始")
answer = chain.run("猫")
print("answer: ", answer)
print("chain 结束")
