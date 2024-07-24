import os


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


# Whether or not to enable langchain debugging
DEBUG = get_boolean_env_var("DEBUG", False)
# Set DEBUG env var to "true" if you wish to enable LC debugging module
if DEBUG:
    import langchain

    langchain.debug = True


# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Redis Connection Information
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))


def format_redis_conn_from_env():
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

        return start + f"{REDIS_HOST}:{REDIS_PORT}"


REDIS_URL = format_redis_conn_from_env()

# Vector Index Configuration
INDEX_NAME = os.getenv("INDEX_NAME", "rag-redis")


current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
schema_path = os.path.join(parent_dir, "schema.yml")
INDEX_SCHEMA = schema_path
