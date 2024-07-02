# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

from __future__ import annotations
from typing import List

from graph_of_thoughts.operations.operations import Operation


class GraphOfOperations:
    """
    Represents the Graph of Operations, which prescribes the execution plan of thought operations.
    """

    def __init__(self) -> None:
        """
        Initializes a new Graph of Operations instance with empty operations, roots, and leaves.
        The roots are the entry points in the graph with no predecessors.
        The leaves are the exit points in the graph with no successors.
        """
        self.operations: List[Operation] = []
        self.roots: List[Operation] = []
        self.leaves: List[Operation] = []

    def append_operation(self, operation: Operation) -> None:
        """
        Appends an operation to all leaves in the graph and updates the relationships.

        :param operation: The operation to append.
        :type operation: Operation
        """
        self.operations.append(operation)

        if len(self.roots) == 0:
            self.roots = [operation]
        else:
            for leave in self.leaves:
                leave.add_successor(operation)

        self.leaves = [operation]

    def add_operation(self, operation: Operation) -> None:
        """
        Add an operation to the graph considering its predecessors and successors.
        Adjust roots and leaves based on the added operation's position within the graph.

        :param operation: The operation to add.
        :type operation: Operation
        """
        self.operations.append(operation)
        if len(self.roots) == 0:
            self.roots = [operation]
            self.leaves = [operation]
            assert (
                len(operation.predecessors) == 0
            ), "First operation should have no predecessors"
        else:
            if len(operation.predecessors) == 0:
                self.roots.append(operation)
            for predecessor in operation.predecessors:
                if predecessor in self.leaves:
                    self.leaves.remove(predecessor)
            if len(operation.successors) == 0:
                self.leaves.append(operation)
