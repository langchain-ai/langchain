import os


def get_boolean_env_var(var_name, default_value=False):
    """Retrieve the boolean value of an environment variable.

    Args:
    var_name (str): The name of the environment variable to retrieve.
    default_value (bool): The default value to return if the variable is not found.

    Returns:
    bool: The value of the environment variable, interpreted as a boolean.
    """
    true_values = {'true', '1', 't', 'y', 'yes'}
    false_values = {'false', '0', 'f', 'n', 'no'}

    # Retrieve the environment variable's value
    value = os.getenv(var_name, '').lower()

    # Decide the boolean value based on the content of the string
    if value in true_values:
        return True
    elif value in false_values:
        return False
    else:
        return default_value


# Whether or not to enable LLM cache
USE_CACHE = get_boolean_env_var("USE_CACHE", False)
CACHE_TTL = int(os.getenv("CACHE_TTL", 600)) # default to 10 mins

# Redis Connection Information
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_USER = os.getenv("REDIS_USER", "")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    REDIS_URL = f"redis://{REDIS_USER}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"

# Vector Index Configuration
INDEX_NAME = os.getenv("INDEX_NAME", "rag-cache-redis")
VECTOR_SCHEMA = {
    "name": "content_vector",      # name of the vector field in langchain
    "algorithm": "HNSW",           # could use HNSW instead
    "dims": 384,                   # set based on the HF model embedding dimension
    "distance_metric": "COSINE",   # could use EUCLIDEAN or IP
    "datatype": "FLOAT32",
}
INDEX_SCHEMA = {
    "vector": [VECTOR_SCHEMA],
    "text": [{"name": "content"}],
}
