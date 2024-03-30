from langchain_community.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

all_documents = {
    "doc1": "Climate change and economic impact.",
    "doc2": "Public health concerns due to climate change.",
    "doc3": "Climate change: A social perspective.",
    "doc4": "Technological solutions to climate change.",
    "doc5": "Policy changes needed to combat climate change.",
    "doc6": "Climate change and its impact on biodiversity.",
    "doc7": "Climate change: The science and models.",
    "doc8": "Global warming: A subset of climate change.",
    "doc9": "How climate change affects daily weather.",
    "doc10": "The history of climate change activism.",
}

PineconeVectorStore.from_texts(
    list(all_documents.values()), OpenAIEmbeddings(), index_name="rag-fusion"
)
