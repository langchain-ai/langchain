""""""
from typing import Dict, List, Tuple

from langchain.chains.query_constructor.query_language import (
    Comparison,
    Operation,
    Visitor,
)


class PineconeTranslator(Visitor):
    def visit_operation(self, operation: Operation) -> Dict:
        root: Dict = {f"${operation.operator}": []}
        stack: List[Tuple[Dict, List]] = [(root, operation.arguments)]
        while stack:
            op_dict, op_args = stack.pop()
            new_args: List = list(op_dict.values())[0]
            for arg in op_args:
                if isinstance(arg, Comparison):
                    new_args.append(self.visit_comparison(arg))
                else:
                    next_op: Dict = {f"${arg.operator}": []}
                    new_args.append(next_op)
                    stack.append((next_op, arg.arguments))
        return root

    def visit_comparison(self, comparison: Comparison) -> Dict:
        return {comparison.attribute: {f"${comparison.comparator}": comparison.value}}
