from __future__ import annotations

from typing import Any, Dict, List, Optional, Union


class _Embed:
    def __init__(self, value: Any, keep: bool = False):
        self.value = value
        self.keep = keep

    def __str__(self) -> str:
        return str(self.value)

    __repr__ = __str__


def stringify_embedding(embedding: List) -> str:
    """Convert an embedding to a string."""

    return " ".join([f"{i}:{e}" for i, e in enumerate(embedding)])


def is_stringtype_instance(item: Any) -> bool:
    """Check if an item is a string."""

    return isinstance(item, str) or (
        isinstance(item, _Embed) and isinstance(item.value, str)
    )


def embed_string_type(
    item: Union[str, _Embed], model: Any, namespace: Optional[str] = None
) -> Dict[str, Union[str, List[str]]]:
    """Embed a string or an _Embed object."""

    keep_str = ""
    if isinstance(item, _Embed):
        encoded = stringify_embedding(model.encode(item.value))
        if item.keep:
            keep_str = item.value.replace(" ", "_") + " "
    elif isinstance(item, str):
        encoded = item.replace(" ", "_")
    else:
        raise ValueError(f"Unsupported type {type(item)} for embedding")

    if namespace is None:
        raise ValueError(
            "The default namespace must be provided when embedding a string or _Embed object."  # noqa: E501
        )

    return {namespace: keep_str + encoded}


def embed_dict_type(item: Dict, model: Any) -> Dict[str, Any]:
    """Embed a dictionary item."""
    inner_dict: Dict = {}
    for ns, embed_item in item.items():
        if isinstance(embed_item, list):
            inner_dict[ns] = []
            for embed_list_item in embed_item:
                embedded = embed_string_type(embed_list_item, model, ns)
                inner_dict[ns].append(embedded[ns])
        else:
            inner_dict.update(embed_string_type(embed_item, model, ns))
    return inner_dict


def embed_list_type(
    item: list, model: Any, namespace: Optional[str] = None
) -> List[Dict[str, Union[str, List[str]]]]:
    """Embed a list item."""

    ret_list: List = []
    for embed_item in item:
        if isinstance(embed_item, dict):
            ret_list.append(embed_dict_type(embed_item, model))
        elif isinstance(embed_item, list):
            item_embedding = embed_list_type(embed_item, model, namespace)
            # Get the first key from the first dictionary
            first_key = next(iter(item_embedding[0]))
            # Group the values under that key
            grouping = {first_key: [item[first_key] for item in item_embedding]}
            ret_list.append(grouping)
        else:
            ret_list.append(embed_string_type(embed_item, model, namespace))
    return ret_list


def embed(
    to_embed: Union[Union[str, _Embed], Dict, List[Union[str, _Embed]], List[Dict]],
    model: Any,
    namespace: Optional[str] = None,
) -> List[Dict[str, Union[str, List[str]]]]:
    """
    Embed the actions or context using the SentenceTransformer model
    (or a model that has an `encode` function).

    Attributes:
        to_embed: (Union[Union(str, _Embed(str)), Dict, List[Union(str, _Embed(str))], List[Dict]], required) The text to be embedded, either a string, a list of strings or a dictionary or a list of dictionaries.
        namespace: (str, optional) The default namespace to use when dictionary or list of dictionaries not provided.
        model: (Any, required) The model to use for embedding
    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary has the namespace as the key and the embedded string as the value
    """  # noqa: E501
    if (isinstance(to_embed, _Embed) and isinstance(to_embed.value, str)) or isinstance(
        to_embed, str
    ):
        return [embed_string_type(to_embed, model, namespace)]
    elif isinstance(to_embed, dict):
        return [embed_dict_type(to_embed, model)]
    elif isinstance(to_embed, list):
        return embed_list_type(to_embed, model, namespace)
    else:
        raise ValueError("Invalid input format for embedding")
