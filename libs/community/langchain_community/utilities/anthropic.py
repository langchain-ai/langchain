from typing import Any, List


def _get_anthropic_client() -> Any:
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "Could not import anthropic python package. "
            "This is needed in order to accurately tokenize the text "
            "for anthropic models. Please install it with `pip install anthropic`."
        )
    return anthropic.Anthropic()


def get_num_tokens_anthropic(text: str) -> int:
    """Get the number of tokens in a string of text."""
    client = _get_anthropic_client()
    return client.count_tokens(text=text)


def get_token_ids_anthropic(text: str) -> List[int]:
    """Get the token ids for a string of text."""
    client = _get_anthropic_client()
    tokenizer = client.get_tokenizer()
    encoded_text = tokenizer.encode(text)
    return encoded_text.ids
