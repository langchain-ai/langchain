import torch
import intel_extension_for_pytorch as ipex
from langchain_core.documents import Document
from intel_extension_for_transformers.langchain.embeddings import HuggingFaceBgeEmbeddings
from intel_extension_for_transformers.langchain.vectorstores import Chroma
from intel_extension_for_transformers.neural_chat.pipeline.plugins.retrieval.parser.parser import DocumentParser
import requests

url = "https://d1io3yog0oux5.cloudfront.net/_897efe2d574a132883f198f2b119aa39/intel/db/888/8941/file/412439%281%29_12_Intel_AR_WR.pdf"
output_file_path = "Intel_AR_WR.pdf"

response = requests.get(url)

if response.status_code == 200:
    with open(output_file_path, 'wb') as pdf_file:
        pdf_file.write(response.content)
    print(f"File downloaded successfully as {output_file_path}")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")

document_parser = DocumentParser()
input_path="./Intel_AR_WR.pdf"
data_collection=document_parser.load(input=input_path)
documents = []
for data, meta in data_collection:
    doc = Document(page_content=data, metadata={"source":meta})
    documents.append(doc)

embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")
embeddings.client= ipex.optimize(embeddings.client.eval(), dtype=torch.bfloat16)
knowledge_base = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory='./output')
