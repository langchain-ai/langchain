from typing import Tuple

from langchain_experimental.tot.memory import ToTDFSMemory
from langchain_experimental.tot.thought import ThoughtValidity


class ToTController:
    """
    Tree of Thought (ToT) controller.

    This is a version of a ToT controller, dubbed in the paper as a "Simple
    Controller".

    It has one parameter `c` which is the number of children to explore for each
    thought.
    """

    def __init__(self, c: int = 3):
        """
        Initialize the controller.

        Args:
            c: The number of children to explore at each node.
        """
        self.c = c

    def __call__(self, memory: ToTDFSMemory) -> Tuple[str, ...]:
        next_thought = memory.top()
        parent_thought = memory.top_parent()
        validity = (
            ThoughtValidity.VALID_INTERMEDIATE
            if next_thought is None
            else next_thought.validity
        )

        # 1 if the current partial solution is invalid, backtrack to the parent
        # thought.
        if validity == ThoughtValidity.INVALID:
            memory.pop()
            next_thought = memory.top()
            if next_thought and len(next_thought.children) >= self.c:
                memory.pop()

        # 2 if the current partial solution is valid but C children were
        # explored and yet failed to find a final solution, backtrack to the
        # parent thought.
        elif (
            validity == ThoughtValidity.VALID_INTERMEDIATE
            and parent_thought
            and len(parent_thought.children) >= self.c
        ):
            memory.pop(2)

        return tuple(thought.text for thought in memory.current_path())
