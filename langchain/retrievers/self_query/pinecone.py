""""""
from typing import Any, Dict, List, Optional, Union

from langchain.chains.query_constructor.base import Comparison, Operation


def comparison_to_pinecone(comparison: Comparison) -> dict:
    return {comparison.attribute: {f"${comparison.comparator}": comparison.value}}


def translate_filter_to_pinecone(
    _filter: Optional[Union[Comparison, Operation]]
) -> dict:
    if _filter is None:
        return {}
    if isinstance(_filter, Comparison):
        return comparison_to_pinecone(_filter)
    root: Dict[str, List[Any]] = {f"${_filter.operator}": _filter.arguments}
    to_translate = [root]
    while to_translate:
        curr = to_translate.pop()
        curr_op, curr_args = list(*curr.items())
        new_args = []
        for arg in curr_args:
            if isinstance(arg, Comparison):
                new_args.append(comparison_to_pinecone(arg))
            else:
                new_arg = {f"${arg.operator}": arg.arguments}
                new_args.append(new_arg)
                to_translate.append(new_arg)
        curr[curr_op] = new_args
    return root
