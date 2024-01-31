import pathway as pw
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import PathwayVectorServer

HOST = "127.0.0.1"
PORT = 8780


def run_vectorstoreserver(host, port, threaded=False):
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
    vector_server.run_server(host=host, port=port, threaded=threaded)


if __name__ == "__main__":
    run_vectorstoreserver(host=HOST, port=PORT, threaded=False)
