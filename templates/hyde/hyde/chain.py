from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import Chroma

from hyde.prompts import hyde_prompt

# Example for document loading (from url), splitting, and creating vectostore

""" 
# Load
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
vectorstore = Chroma.from_documents(documents=all_splits, 
                                    collection_name="rag-chroma",
                                    embedding=OpenAIEmbeddings(),
                                    )
retriever = vectorstore.as_retriever()
"""

# Embed a single document as a test
vectorstore = Chroma.from_texts(
    ["harrison worked at kensho"],
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

# RAG prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI()

# RAG chain
chain = (
    RunnableParallel(
        {
            # Configure the input, pass it the prompt, pass that to the model,
            # and then the result to the retriever
            "context": {"input": RunnablePassthrough()}
            | hyde_prompt
            | model
            | StrOutputParser()
            | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | model
    | StrOutputParser()
)
