from __future__ import annotations

from typing import List, Optional

from langchain_experimental.tot.thought import Thought


class ToTDFSMemory:
    """
    Memory for the Tree of Thought (ToT) chain.

    It is implemented as a stack of
    thoughts. This allows for a depth first search (DFS) of the ToT.
    """

    def __init__(self, stack: Optional[List[Thought]] = None):
        self.stack: List[Thought] = stack or []

    def top(self) -> Optional[Thought]:
        "Get the top of the stack without popping it."
        return self.stack[-1] if len(self.stack) > 0 else None

    def pop(self, n: int = 1) -> Optional[Thought]:
        "Pop the top n elements of the stack and return the last one."
        if len(self.stack) < n:
            return None
        for _ in range(n):
            node = self.stack.pop()
        return node

    def top_parent(self) -> Optional[Thought]:
        "Get the parent of the top of the stack without popping it."
        return self.stack[-2] if len(self.stack) > 1 else None

    def store(self, node: Thought) -> None:
        "Add a node on the top of the stack."
        if len(self.stack) > 0:
            self.stack[-1].children.add(node)
        self.stack.append(node)

    @property
    def level(self) -> int:
        "Return the current level of the stack."
        return len(self.stack)

    def current_path(self) -> List[Thought]:
        "Return the thoughts path."
        return self.stack[:]
