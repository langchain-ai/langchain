# Load
import os
import uuid

import chromadb
import numpy as np
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from PIL import Image as _PILImage
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

# File
path = "tests/ai_labs/"
paper = "ai_labs.pdf"

# Load and partition
raw_pdf_elements = partition_pdf(
    filename=path + paper,
    extract_images_in_pdf=True,  # Extract images
    infer_table_structure=True,  # Post processing to aggregate text into sections
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,
)

# Get texts and tables
tables = []
texts = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        tables.append(str(element))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        texts.append(str(element))

# Get images
image_files = [f for f in os.listdir(path) if f.endswith(".jpg")]
images = [np.array(_PILImage.open(path + f).convert("RGB")) for f in image_files]

# Store in Chroma with multimodal embd
## TO DO: Merge
client = chromadb.Client()
embedding_function = OpenCLIPEmbeddingFunction()
collection = client.create_collection("mm_rag", embedding_function=embedding_function)

image_ids = [str(uuid.uuid4()) for _ in images]
collection.add(ids=image_ids, images=images)

text_ids = [str(uuid.uuid4()) for _ in texts]
collection.add(ids=text_ids, documents=texts)
collection.get(include=["documents"])

# Pass Chroma Client to LangChain
vectorstore = Chroma(
    client=client,
    collection_name="mm_rag",
    embedding_function=embedding_function,
)

retriever = vectorstore.as_retriever()

# RAG
# Prompt template
template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""  # noqa: E501
prompt = ChatPromptTemplate.from_template(template)

# LLM
### placeholder ###
model = ChatOpenAI(temperature=0, model="gpt-4v")

# RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
