from langchain.chat_models import init_chat_model
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

docs = [
    Document(page_content="Payment must be made within 30 days."),
    Document(page_content="Termination requires 60 days notice."),
]

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(docs, embeddings)

query = "What is the termination policy?"

results = vectorstore.similarity_search(query)

model = init_chat_model("openai:gpt-4o-mini")

response = model.invoke(
    f"Answer based on contract: {results[0].page_content}"
)

print(response)
