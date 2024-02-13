import os

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.storage import RedisStore
from langchain_community.vectorstores import Redis as RedisVectorDB
from langchain_openai.embeddings import OpenAIEmbeddings

ID_KEY = "doc_id"


def get_boolean_env_var(var_name, default_value=False):
    """Retrieve the boolean value of an environment variable.

    Args:
    var_name (str): The name of the environment variable to retrieve.
    default_value (bool): The default value to return if the variable
    is not found.

    Returns:
    bool: The value of the environment variable, interpreted as a boolean.
    """
    true_values = {"true", "1", "t", "y", "yes"}
    false_values = {"false", "0", "f", "n", "no"}

    # Retrieve the environment variable's value
    value = os.getenv(var_name, "").lower()

    # Decide the boolean value based on the content of the string
    if value in true_values:
        return True
    elif value in false_values:
        return False
    else:
        return default_value


# Check for openai API key
if "OPENAI_API_KEY" not in os.environ:
    raise Exception("Must provide an OPENAI_API_KEY as an env var.")


def format_redis_conn_from_env() -> str:
    redis_url = os.getenv("REDIS_URL", None)
    if redis_url:
        return redis_url
    else:
        using_ssl = get_boolean_env_var("REDIS_SSL", False)
        start = "rediss://" if using_ssl else "redis://"

        # if using RBAC
        password = os.getenv("REDIS_PASSWORD", None)
        username = os.getenv("REDIS_USERNAME", "default")
        if password is not None:
            start += f"{username}:{password}@"

        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))

        return start + f"{host}:{port}"


REDIS_URL = format_redis_conn_from_env()

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
schema_path = os.path.join(parent_dir, "schema.yml")
INDEX_SCHEMA = schema_path


def make_mv_retriever():
    """Create the multi-vector retriever"""
    # Load Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    vectorstore = RedisVectorDB(
        redis_url=REDIS_URL,
        index_name="image_summaries",
        key_prefix="summary",
        index_schema=INDEX_SCHEMA,
        embedding=OpenAIEmbeddings(),
    )
    store = RedisStore(redis_url=REDIS_URL, namespace="image")

    # Create the multi-vector retriever
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=ID_KEY,
    )
