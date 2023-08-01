# -*- coding: utf-8 -*-
import os
import getpass
import openai
import langchain
import logging
import gpt4all
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts.example_selector import (
    SemanticSimilarityExampleSelector,
    MaxMarginalRelevanceExampleSelector,
)
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain import document_loaders
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.cache import InMemoryCache
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatMessagePromptTemplate,
)
from langchain import llms

langchain.llm_cache = InMemoryCache()  # 启动llm的缓存
logging.basicConfig(level=logging.INFO)  # 日志
device = "cuda:0"

# openai
# os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')
openai.api_key = "pvQBs4fXbFbmwST4pzDDGCQgUCnFLVA0OIj62Vj9D2c"
openai.api_base = "https://chimeragpt.adventblocks.cc/api/v1"
os.environ["OPENAI_API_KEY"] = "pvQBs4fXbFbmwST4pzDDGCQgUCnFLVA0OIj62Vj9D2c"
os.environ["OPENAI_API_BASE"] = "https://chimeragpt.adventblocks.cc/api/v1"
# huggingface
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "xrNJmp__x6Jf0jWIDsz0kMODyJCn3W34hnUIT0vOGbo"
os.environ["HUGGINGFACEHUB_API_BASE"] = "https://api.huggingface.co"
# serpapi
os.environ[
    "SERPAPI_API_KEY"
] = "eb3978daef457908db7bf92509321107c90d699f8e22f681f4522f2e9bc06963"

# 向量嵌入
model_name = r"E:\workspace\project\text2vec-large-chinese"
# embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
)

# # 加载向量数据库
# raw_documents = TextLoader(
#     r"E:\workspace\project\public\langchain\libs\langchain\learning\data\testSet_1000.txt",
#     encoding="utf8",
# ).load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
# documents = text_splitter.split_documents(raw_documents)
# # db = FAISS.from_documents(documents, OpenAIEmbeddings())
# db = FAISS.from_documents(documents, embeddings)

# 加载本地模型
# model_id = r"E:\workspace\project\public\ChatGLM2-6B\THUDM\chatglm2-6b"
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_id, trust_remote_code=True, device=device)
# pipe = pipeline(
#     task="text2text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_length=5000,
#     # 自加的三个参数
#     temperature=0.1,
#     top_p=0.9,
#     top_k=3,
# )
# local_llm_glm2 = HuggingFacePipeline(pipeline=pipe)
# local_llm_glm2 = llms.ChatGLM()  # 需开启 ChatGLM 的本地 api

# 加载 openai
openai_llm_chat = ChatOpenAI()

# 加载 agents 。 使用 openai 可以，chatglm2 不行。
# let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=openai_llm_chat)
# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(
    tools, openai_llm_chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

while True:
    query = input("You: ")
    # Now let's test it out!
    res = agent.run("查找关于 " + query + " 的 最新 知识. site:https://pubmed.ncbi.nlm.nih.gov/")
    print("res:", res)

# # 定义提示模板
# examples = [
#     {
#         "query": "男，目前51岁，近半年，发现，房事大不如前，此外，不足2,3分钟就射了，请问：男性早泄是由哪些方面引发的。",
#         "answer": "引发早泄的主要病因，受到这些方面的影响，心理问题是其中比较主要的因素，像是夫妻间感情不好，性生活不愉快，或是性的初体验并不好，再者，一些男性如果缺少对性知识的认知，无节制的手淫就会引发早泄，除了这两点，男性如果患上一些男科疾病也会诱发早泄，主要是前列腺或是泌尿系统出现问题，或是心脑血管出现问题，早泄发生的病因主要就这几个方面，可以有针对性的去进行预防。",
#     },
#     {
#         "query": "今年37岁，最近几年，发觉，性生活总是提不起劲，同时，才进去没一会就射了，请问：男人早泄是由哪些方面诱发的。",
#         "answer": "诱发早泄的因素，主要是这几种，比如像男性自身心理上出现了一些问题，比如部分男性比较自卑，在另一半面前抬不起头，或是工作压力大，情绪焦虑等，如果男性对性知识缺少了解，无节制的手淫就会引发早泄，或者如果男性患有一些疾病，也是会诱发早泄的，常见的疾病主要是前列腺和泌尿系，还有就是有心脑血管疾病的情况下，早泄的病因并不单一，了解后务必要正确预防。",
#     },
#     {
#         "query": "男21发现这个症状已经很久了，但是这个情况一直就没有好转，请问移植两个冻囊胚成功率多大？",
#         "answer": "囊胚移植的成功率是不会因为囊胚数目的增强而大幅提高的，所以成功率还是在65%左右，而且女性身体健康程度也会影响嫁接的成功率，所以平时注意防寒，防止熬夜，以免导致身体提抗力上升。不要有任何的心理压力，保持愉快的心情，而且期间要注意多休息，同时饮食也要保持清单，避免油腻和辛辣，这样对你自己和腹中的胎儿都是有好处的。",
#     },
#     {
#         "query": "32岁，我属于那种有大肚子，而且比较高大壮的人，快300斤的体重常常被朋友们嘲笑，成家立室后内人也常常对我不满意，自己很是担心紧张的。那么请问为啥子男人胖了性功能就差？",
#         "answer": "你好，根据您现在的这一种情况，性生活的质量主要是靠技术水平或阴茎海绵体的一种冲血来选择的和那种胖瘦，并没有太直接的一种关联性。建议您现在的这种情况到专业的泌尿外科来仔细检查这种激素水平或阴茎海绵体血流多普勒，这样才能做出更明确的推测来决定药物干涉的治疗方式。",
#     },
#     {
#         "query": "最近这几天总是觉得上厕所的时候很不舒服，有点疼，平时吧就感觉尿道口有烧灼感，反正就是左立难安，我今年才30岁，不会是得前列腺炎了吧。如果真的是，请问：发现前列腺炎怎么样治疗好得快？",
#         "answer": "你好，根据仔细检查结果来看，主要还是由于前列腺炎致使的。平时要留意多喝水，多排尿，不要长时间憋尿，防止吃辛辣刺激性的食物，平时留意不要久站或久坐，容易致使病情加重，建议口服左氧氟沙星胶囊，互相配合普乐安片实施治疗前列腺炎患者在及时治疗之外，患者在生活中还需要有留意要留意始终保持深度睡眠，充足的深度睡眠方才能帮你尽快远离此病，期望上述的答案可以帮助到你，谢谢。",
#     },
#     {
#         "query": "精囊炎应当注意什么？较近单位实施健康检查，结果被查出来患上了精囊炎，想理解一下患上精囊炎应该注意些什么？想进行咨询一下医生，患上精囊炎应该注意些什么？",
#         "answer": "染上精囊炎应该对其积极的实施护理，有意识的护理有助于患者病情的康复，如下这些事项患者应该注意下来：保持良好的心态、注意恰当饮食、注意个人卫生、惯健康的饮食习惯、坚持活动。男性患上精囊炎会影响男性的生育能力，所以男性如果有精囊炎的症状再次出现的话，一定要引起重视，尽快到专业的医院实施检查和治疗，然后结合上述所述注意事项，双管齐下增进疾病的早日康复。",
#     },
# ]
# examples_doctor = [
#     {
#         "query": "深龋（后牙、面洞）临床路径表单",
#         "answer": "诊疗流程: "
#         "1.询问病史，完成临床检查及辅助检查，明确诊断，制定治疗计划;"
#         "2.向患者或其监护人交待治疗计划、方法、疗程、风险和费用等，并获得知情同意;"
#         "3.必要时局麻;"
#         "4.隔离患牙（推荐使用橡皮障）;"
#         "5.去尽龋坏组织;"
#         "6.制备必要的洞形;"
#         "7.必要时垫底或洞衬;"
#         "8.按所使用的充填材料的要求完成牙体修复;"
#         "9.修整外形，调合，抛光;",
#     },
#     {
#         "query": "慢性牙髓炎（恒磨牙）临床路径表单",
#         "answer": "一、完成诊断与治疗计划: 1.询问病史，完成临床及X线片检查，明确诊断和治疗计划;2.向患者或其监护人交待治疗计划、方法、疗程、风险和费用等，并获得知情同意;"
#         "二、无痛治疗: 1.根据患者身体情况及牙位选择合适的局部麻醉方式;"
#         "三、开髓及髓腔预备: 1.隔离患牙（推荐使用橡皮障）;2.去腐，开髓，拔髓（或放置牙髓失活剂）;3.髓腔预备，暴露全部根管口;4.通畅根管;5.测定工作长度;"
#         "四、根管预备: 1,依据所使用的预备技术及器械要求完成根管预备;2.充分进行根管冲洗 ;3.可以辅助使用超声波器械增强治疗效果;4.试主尖，术中X片检查主尖适配性;"
#         "五、根管充填: 1.干燥根管;2.依据所使用的根管充填技术要求，以牙胶与封闭剂充填根管;3.拍摄X线根尖片确认根充情况;"
#         "六、冠方封闭: 1.诊间及诊疗结束后均需使用材料严密封闭冠部缺损，防止冠方渗漏。",
#     },
#     {
#         "query": "（急性）牙周脓肿行急症处理的临床路径表单",
#         "answer": "诊疗第1次（初次门诊、引流、冲洗）: 1.询问病史及体格检查;2.牙周检查;3.必要的辅助检查;"
#         "4.诊断;5.制定治疗计划;6.完成病历书写;7.清除大块牙石;8.脓肿切开引流;9.局部冲洗、上药;10.开具漱口液;11.口腔卫生指导;"
#         "诊疗第2次（初诊后3天，复查、冲洗上药）: 1.脓肿变化情况的检查;2.局部冲洗、上药;3.开具漱口液;4.口腔卫生指导;"
#         "诊疗第3次（初诊后1周，复查、确认疾病转归）: 1.脓肿愈合情况检查2.局部冲洗、上药3.完善后续治疗计划",
#     },
#     {
#         "query": "菌斑性龈炎（边缘性龈炎）行牙周洁治的临床路径表单",
#         "answer": "诊疗第1次（初次门诊、洁治）: 1.询问病史及体格检查;2.牙周检查;3.诊断;4.制定治疗计划;5.完成病历书写;6.洁治治疗;7.局部冲洗;8.口腔卫生指导;"
#         "诊疗第2次（补充洁治，初次门诊后1周）: 1.口腔卫生情况检查;2.牙龈炎症情况的检查;3.针对余留牙石和菌斑再次洁治;4.局部冲洗;5.喷砂（色素多的患者）;6.牙面抛光术;7.口腔卫生指导;",
#     },
# ]
# example_prompt = PromptTemplate(
#     input_variables=["query", "answer"], template="Question: {query}\nAnwer: {answer}"
# )
# chat_prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessagePromptTemplate.from_template("你是医学专家, 你需要为患者指定治疗方案，包括临床路径表单等。"),
#         HumanMessagePromptTemplate.from_template("{query}", input_variables=["query"]),
#     ]
# )
#
# # 定义示例选择器，此处与从 向量数据库查找重复
# example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
#     # This is the list of examples available to select from.
#     examples_doctor,
#     # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
#     # OpenAIEmbeddings(),  #  openai
#     embeddings,
#     # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
#     FAISS,
#     # This is the number of examples to produce.
#     k=2,
# )
#
# while True:
#     query = input("Enter query: ")
#     # query = "我儿子在做疝气手术时，查出来碱性磷酸酶高达1500怎么解决好？情况不怎么好，挺担心的。"
#     # 相似性搜索
#     # docs = db.similarity_search(query)
#
#     # 向量相似性搜索
#     # embedding_vector = OpenAIEmbeddings().embed_query(query)  # OpenaiEmbeddings 需要 api-key
#     embedding_vector = embeddings.embed_query(query)
#     print("embedding_vector: ", embedding_vector)
#     # docs = db.similarity_search_by_vector(embedding_vector)
#     # print(docs[0].page_content)  # TODO: 可将此处的结果 变成json文件 作为 examples，以便直接在 prompt 中直接调用
#
#     few_shot_prompt = FewShotPromptTemplate(
#         # examples=examples,  # TODO: examples 可以给定， 或从向量数据库查找？(感觉查找不靠谱)
#         example_selector=example_selector,
#         example_prompt=example_prompt,
#         # prefix="你是专业的请回答患者一下问题",
#         suffix="请根据上面的例子回答下面的问题。\nQuestion: {query}\nAnswer: ",
#         input_variables=["query"],
#         example_separator="\n",
#     )
#     query_new = few_shot_prompt.format_prompt(query=query).to_string()
#
#     # 使用 agents, 中文有问题,建议使用英文
#     # res = agent.run(query)
#     # print("res:", res)
#     # query = query_new + res
#
#     # 调用 本地模型
#     llm_chain = LLMChain(
#         llm=openai_llm_chat,
#         prompt=chat_prompt,
#     )
#
#     # 调用 openai
#     # llm_chain = LLMChain(llm=openai_local_llm, prompt=chat_prompt)
#     resp = llm_chain.run(query=query_new)
#     print("resp: ", resp)
