import pathway as pw
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import PathwayVectorClient, PathwayVectorServer

HOST = "127.0.0.1"
PORT = 8780

# If you have a running Pathway Vectorstore instance you can connect to it via client.
# If not, you can run Vectorstore as follows:
create_vectorstore = True
if create_vectorstore:
    # Example for document loading (from local folders), splitting,
    # and creating vectorstore with Pathway

    data_sources = []
    data_sources.append(
        pw.io.fs.read(
            "./sample_documents", format="binary", mode="streaming", with_metadata=True
        )  # This creates a `pathway` connector that tracks
        # all the files in the sample_documents directory
    )

    # This creates a connector that tracks files in Google drive.
    # please follow the instructions at:
    # https://pathway.com/developers/tutorials/connectors/gdrive-connector/
    # to get credentials, see pathway documentation for more options including:
    # s3, Dropbox, etc.
    # data_sources.append(
    #     pw.io.gdrive.read(
    #         object_id="17H4YpBOAKQzEJ93xmC2z170l0bP2npMy",
    #         service_user_credentials_file="credentials.json",
    #         with_metadata=True,
    #     )
    # )

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=10,
        chunk_overlap=5,
        length_function=len,
        is_separator_regex=False,
    )

    # Embed
    embeddings_model = OpenAIEmbeddings()

    # Launch VectorDB
    vector_server = PathwayVectorServer(
        *data_sources, embedder=embeddings_model, splitter=text_splitter
    )
    vector_server.run_server(host=HOST, port=PORT, threaded=True)

# Initalize client
client = PathwayVectorClient(
    host=HOST,
    port=PORT,
)

retriever = client.as_retriever()


# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = ChatOpenAI()
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
