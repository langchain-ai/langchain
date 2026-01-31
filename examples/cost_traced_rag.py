import os
from typing import List

# Fake dependencies effectively for the purpose of the example if not present,
# but trying to use real imports where possible.
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    # Fallback or mock if not available, but the code is the deliverable.
    # We will assume the user has these installed or will install them.
    pass

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManager

# Import our new callback
# Adjust import path based on where it sits in python path vs file structure.
# Since we put it in libs/langchain/langchain_classic/callbacks/cost_tracking.py
# and langchain_classic is likely in python path or we need to adjust.
# For the sake of the example code to be "correct" when installed:
from langchain_classic.callbacks.cost_tracking import CostTrackingCallback

# Optional: OpenTelemetry
try:
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)
except ImportError:
    tracer = None

def run_example():
    print("🚀 Starting Cost-Aware RAG Pipeline Example")

    # 1. Documents
    docs_text = [
        "LangChain is a framework for building LLM-powered applications.",
        "RAG combines retrieval and generation.",
        "Observability is critical for GenAI systems.",
        "Cost tracking helps ensuring your AI budget is not exceeded."
    ]

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    documents = [Document(page_content=t) for t in docs_text]
    # In a real scenario we'd split, but these are short.
    split_docs = splitter.split_documents(documents)

    # 3. Vector Store
    # Note: Requires OPENAI_API_KEY env var
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found. Skipping actual LLM calls.")
        return

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(split_docs, embeddings)
    retriever = db.as_retriever()

    # 4. Callback
    cost_callback = CostTrackingCallback(cost_per_1k_tokens=0.002)
    callback_manager = CallbackManager([cost_callback])

    # 5. LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        callback_manager=callback_manager
    )

    # 6. Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # 7. Run
    query = "Why is observability important in RAG?"
    print(f"\n❓ Query: {query}")

    if tracer:
        with tracer.start_as_current_span("rag-query"):
            answer = qa.run(query)
    else:
        answer = qa.run(query)

    print(f"\n💡 Answer: {answer}")

    # 8. Metrics
    print("\n📊 Cost & Usage Metrics:")
    print(cost_callback.summary())

if __name__ == "__main__":
    run_example()
