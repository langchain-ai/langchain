import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable, Literal

from pydantic import BaseModel, Field
from typing_extensions import Self

from graph_of_thoughts.operations.graph_of_operations import GraphOfOperations
from graph_of_thoughts.operations.operations import (
    Operation,
    OperationStatus,
    OperationSummary,
    OperationType,
)


@dataclass
class OperationNode:
    id: int
    successors: list[Self]
    summary: OperationSummary


class GOTStatus(Enum):
    RUNNING = 0
    FINISHED = 1
    FAILED = 2


class LegendItem(BaseModel):
    label: str
    color: str
    type: Literal["node", "edge"]

    def __hash__(self):
        return hash(self.label)


@dataclass
class Cytoscape:
    elements: list[dict]
    roots: list[str]
    terminals: list[str]
    fcose_horizontal_layout: list[list[str]]
    fcose_vertical_layout: list[list[str]]
    fcose_relative_constraints: list[dict]
    legend: list[LegendItem]


class GraphSummary(BaseModel):
    nodes: dict[int, OperationSummary]
    adj_list: dict[int, list[int]]
    roots: list[int]
    status: GOTStatus
    timestamp: datetime = Field(default=datetime.now())
    title: str | None = None

    def predecessor_list(self) -> dict[int, list[int]]:
        pred: dict[int, set[int]] = defaultdict(set)

        for parent, children in self.adj_list.items():
            for child in children:
                pred[child].add(parent)

        return {k: list(v) for k, v in pred.items()}

    @classmethod
    def from_graph(cls, graph: GraphOfOperations, status=GOTStatus.RUNNING) -> Self:
        assert graph.roots is not None, "The operations graph has no root"

        queue = [*graph.roots]

        nodes: dict[int, OperationSummary] = {}
        adj_set: dict[int, set[int]] = defaultdict(set)
        root_ids = [r.id for r in graph.roots]

        for operation in queue:
            nodes[operation.id] = operation.get_summary()

            for succ in operation.successors:
                adj_set[operation.id].add(succ.id)
                queue.append(succ)

        adj_list = {k: list(v) for k, v in adj_set.items()}

        return cls(nodes=nodes, adj_list=adj_list, roots=root_ids, status=status)

    def as_object_graph(self) -> list[OperationNode]:
        roots: list[OperationNode] = []

        for root_id in self.roots:
            op = OperationNode(id=root_id, successors=[], summary=self.nodes[root_id])
            roots.append(op)

        def add_successors(node: OperationNode):
            # Recursive fun
            # Inefficient, duplicates successors :(
            succ_ids = self.adj_list[node.id]

            if not succ_ids:
                return

            succesors: list[OperationNode] = []

            for id_ in succ_ids:
                node = OperationNode(id=id_, successors=[], summary=self.nodes[id_])
                succesors.append(node)

            node.successors = succesors

            for succ in succesors:
                add_successors(succ)

        return roots

    def _fcose_horizontal_layout(self, roots: list[str]) -> list[list[str]]:
        level_ids: list[list[str]] = [roots]  # list 1: level, list2: level ids
        already_traversed: set[int] = set()

        def traverse_level(level_items: Iterable[int]):
            """Recursive fn that traverses a level and appends the results to
            level_ids

            Args:
                level_items (Iterable[int]): _description_
            """
            # add a new level
            current_lvl_ids = []
            for item_id in level_items:
                level_subitems = [
                    self._fmt_id(item_id, i)
                    for i in range(self.nodes[item_id].n_thoughts)
                ]
                current_lvl_ids.extend(level_subitems)

            if len(current_lvl_ids):
                level_ids.append(current_lvl_ids)

            # update the set to avoid dupes
            already_traversed.update(level_items)

            next_level: set[int] = set()

            for item_id in level_items:
                if item_id in self.adj_list:
                    next_level.update(self.adj_list[item_id])
            next_level_f = list(
                filter(lambda x: x not in already_traversed, next_level)
            )

            if len(next_level_f):
                traverse_level(next_level_f)

        traverse_level(self.roots)

        return level_ids

    def _fcose_rel_constraints(self, horizontal_layout: list[list[str]]) -> list[dict]:
        constraints: list[dict] = []

        if len(horizontal_layout) < 2:
            return []

        for i in range(1, len(horizontal_layout)):
            prev_lay = horizontal_layout[i - 1]
            current_lay = horizontal_layout[i]

            assert len(prev_lay)
            assert len(current_lay)

            constraints.append(
                {"top": prev_lay[0], "bottom": current_lay[0], "gap": 100}
            )

        return constraints

    def _fcose_vertical_layout(
        self, horizontal_layout: list[list[str]]
    ) -> list[list[str]]:
        layout: list[str] = []
        if len(horizontal_layout) < 2:
            return []

        last_single_id: str = horizontal_layout[0][0]
        layout.append(last_single_id)
        for i, level in enumerate(horizontal_layout):
            if len(level) == 1:
                layout.append(level[0])
                last_single_id = level[0]

        return [layout]

    def _fmt_id(self, node_id: int | str, idx: int | str) -> str:
        return f"{node_id}-{idx}"

    def _create_thought_nodes(
        self,
        thought_id: str | int,
        n_thoughts: int,
        color: str,
        label: str | None = None,
    ) -> dict[str, dict]:
        thoughts = {}
        for i in range(n_thoughts):
            id_ = self._fmt_id(thought_id, i)
            thoughts[id_] = {
                "data": {
                    "id": id_,
                    "op_id": thought_id,
                    "label": label if label else id_,
                    "background_color": color,
                }
            }

        return thoughts

    def _connect_nodes(
        self,
        current_id: str | int,
        current_thoughts: int,
        predecessor_id: str | int,
        predecessor_thoughts,
        color: str | None = None,
        current_label: str | None = None,
    ) -> dict[str, dict]:
        connections: dict[str, dict] = {}

        def create_conn(
            from_: str | int, to: str | int, label: str | None
        ) -> tuple[str, dict]:
            id_ = f"{from_} -> {to}"
            conn = {
                "data": {
                    "id": id_,
                    "source": from_,
                    "target": to,
                    "label": label,
                    "background_color": color,
                }
            }
            return id_, conn

        if current_thoughts == predecessor_thoughts:
            # Connects each thought with the previous one, no cross-lines
            for i in range(current_thoughts):
                from_ = self._fmt_id(predecessor_id, i)
                to = self._fmt_id(current_id, i)

                conn_id, conn = create_conn(from_, to, current_label)
                connections[conn_id] = conn
        else:
            # Connects all the current thoughts with all the previous ones
            for i in range(current_thoughts):
                for j in range(predecessor_thoughts):
                    from_ = self._fmt_id(predecessor_id, j)
                    to = self._fmt_id(current_id, i)
                    conn_id, conn = create_conn(from_, to, current_label)
                    connections[conn_id] = conn
        return connections

    def _build_color_legend(self) -> list[LegendItem]:
        legend: set[LegendItem] = set()

        for node in self.nodes.values():
            status = node.status
            legend.add(
                LegendItem(label=status.name, color=status.get_css_color(), type="node")
            )

            op_type = node.type
            legend.add(
                LegendItem(
                    label=op_type.name, color=op_type.get_css_color(), type="edge"
                )
            )

        legend_l = list(legend)

        legend_l.sort(key=lambda x: f"{x.type}{x.label}")
        logging.warning(legend_l)
        return legend_l

    def as_cytoscape(self) -> Cytoscape:
        elements: dict[str, dict] = {}
        predecessors_l = self.predecessor_list()
        roots = []  # Graph roots, (Initial Thought)
        # Create the element list

        logging.warning("=" * 10)

        logging.warning(predecessors_l)
        for curr_node_id, curr_node in self.nodes.items():  # for each operation
            predecessor_ids = predecessors_l.get(curr_node_id)
            logging.warning(f"Curr Node id {curr_node_id} - {curr_node.type}")

            # Step 1: Create the current row of thoughts
            thoughts = self._create_thought_nodes(
                (curr_node_id),
                curr_node.n_thoughts,
                label=curr_node.type.get_verb(curr_node.status),
                color=curr_node.status.get_css_color(),
            )
            elements = elements | thoughts  # join the elements

            if predecessor_ids is None:  # No predecessors :'(
                start_thought = self._create_thought_nodes(
                    f"initial-{curr_node_id}",
                    1,
                    label="Initial Thought",
                    color=OperationStatus.EXECUTED.get_css_color(),
                )
                start_conn = self._connect_nodes(
                    curr_node_id, curr_node.n_thoughts, f"initial-{curr_node_id}", 1
                )
                elements = elements | start_thought | start_conn

                roots = list(start_thought.keys())
                continue

            # Step 2: Connect with the previous layer (if possible)
            for pred_id in predecessor_ids:  # possibly just one predecessor
                logging.warning(f"  Predecessor {pred_id}")
                predecessor = self.nodes[pred_id]
                connections = self._connect_nodes(
                    curr_node_id,
                    curr_node.n_thoughts,
                    pred_id,
                    predecessor.n_thoughts,
                    current_label=curr_node.fmt_op(),
                    color=curr_node.type.get_css_color(),
                )
                elements = elements | connections

        # create the layout
        layout_h = self._fcose_horizontal_layout(roots)
        layout_v = self._fcose_vertical_layout(layout_h)
        cons = self._fcose_rel_constraints(layout_h)
        logging.warning("-" * 10)

        return Cytoscape(
            elements=list(elements.values()),
            fcose_horizontal_layout=layout_h,
            fcose_vertical_layout=layout_v,
            fcose_relative_constraints=cons,
            roots=roots,
            terminals=layout_h[-1] if len(layout_h) else roots,
            legend=self._build_color_legend(),
        )
