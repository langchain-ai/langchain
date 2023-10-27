import os

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores.supabase import SupabaseVectorStore
from supabase.client import create_client

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings()

vectorstore = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)

# Adjust this based on the metadata you store in the `metadata` JSON column
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]

# Adjust this based on the type of documents you store
document_content_description = "Brief summary of a movie"
llm = OpenAI(temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)

chain = RunnableParallel({"query": RunnablePassthrough()}) | retriever
