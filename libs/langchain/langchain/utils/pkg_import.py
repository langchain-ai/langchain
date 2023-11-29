# flake8: noqa


def import_openai():
    try:
        import openai

        return openai
    except ImportError as e:
        raise ImportError(
            "Could not import openai python package. "
            "Please install it with `pip install openai`."
        ) from e
