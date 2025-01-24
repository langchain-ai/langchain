"""Draws DAG in ASCII.
Adapted from https://github.com/iterative/dvc/blob/main/dvc/dagascii.py.
"""

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.runnables.graph import Edge as LangEdge


class VertexViewer:
    """Class to define vertex box boundaries that will be accounted for during
    graph building by grandalf.

    Args:
        name (str): name of the vertex.
    """

    HEIGHT = 3  # top and bottom box edges + text
    """Height of the box."""

    def __init__(self, name: str) -> None:
        self._h = self.HEIGHT  # top and bottom box edges + text
        self._w = len(name) + 2  # right and left bottom edges + text

    @property
    def h(self) -> int:
        """Height of the box."""
        return self._h

    @property
    def w(self) -> int:
        """Width of the box."""
        return self._w


class AsciiCanvas:
    """Class for drawing in ASCII.

    Args:
        cols (int): number of columns in the canvas. Should be > 1.
        lines (int): number of lines in the canvas. Should be > 1.
    """

    TIMEOUT = 10

    def __init__(self, cols: int, lines: int) -> None:
        if cols <= 1 or lines <= 1:
            msg = "Canvas dimensions should be > 1"
            raise ValueError(msg)

        self.cols = cols
        self.lines = lines

        self.canvas = [[" "] * cols for line in range(lines)]

    def draw(self) -> str:
        """Draws ASCII canvas on the screen."""
        lines = map("".join, self.canvas)
        return os.linesep.join(lines)

    def point(self, x: int, y: int, char: str) -> None:
        """Create a point on ASCII canvas.

        Args:
            x (int): x coordinate. Should be >= 0 and < number of columns in
                the canvas.
            y (int): y coordinate. Should be >= 0 an < number of lines in the
                canvas.
            char (str): character to place in the specified point on the
                canvas.
        """
        if len(char) != 1:
            msg = "char should be a single character"
            raise ValueError(msg)
        if x >= self.cols or x < 0:
            msg = "x should be >= 0 and < number of columns"
            raise ValueError(msg)
        if y >= self.lines or y < 0:
            msg = "y should be >= 0 and < number of lines"
            raise ValueError(msg)

        self.canvas[y][x] = char

    def line(self, x0: int, y0: int, x1: int, y1: int, char: str) -> None:
        """Create a line on ASCII canvas.

        Args:
            x0 (int): x coordinate where the line should start.
            y0 (int): y coordinate where the line should start.
            x1 (int): x coordinate where the line should end.
            y1 (int): y coordinate where the line should end.
            char (str): character to draw the line with.
        """
        if x0 > x1:
            x1, x0 = x0, x1
            y1, y0 = y0, y1

        dx = x1 - x0
        dy = y1 - y0

        if dx == 0 and dy == 0:
            self.point(x0, y0, char)
        elif abs(dx) >= abs(dy):
            for x in range(x0, x1 + 1):
                y = y0 if dx == 0 else y0 + int(round((x - x0) * dy / float(dx)))
                self.point(x, y, char)
        elif y0 < y1:
            for y in range(y0, y1 + 1):
                x = x0 if dy == 0 else x0 + int(round((y - y0) * dx / float(dy)))
                self.point(x, y, char)
        else:
            for y in range(y1, y0 + 1):
                x = x0 if dy == 0 else x1 + int(round((y - y1) * dx / float(dy)))
                self.point(x, y, char)

    def text(self, x: int, y: int, text: str) -> None:
        """Print a text on ASCII canvas.

        Args:
            x (int): x coordinate where the text should start.
            y (int): y coordinate where the text should start.
            text (str): string that should be printed.
        """
        for i, char in enumerate(text):
            self.point(x + i, y, char)

    def box(self, x0: int, y0: int, width: int, height: int) -> None:
        """Create a box on ASCII canvas.

        Args:
            x0 (int): x coordinate of the box corner.
            y0 (int): y coordinate of the box corner.
            width (int): box width.
            height (int): box height.
        """
        if width <= 1 or height <= 1:
            msg = "Box dimensions should be > 1"
            raise ValueError(msg)

        width -= 1
        height -= 1

        for x in range(x0, x0 + width):
            self.point(x, y0, "-")
            self.point(x, y0 + height, "-")

        for y in range(y0, y0 + height):
            self.point(x0, y, "|")
            self.point(x0 + width, y, "|")

        self.point(x0, y0, "+")
        self.point(x0 + width, y0, "+")
        self.point(x0, y0 + height, "+")
        self.point(x0 + width, y0 + height, "+")


def _build_sugiyama_layout(
    vertices: Mapping[str, str], edges: Sequence[LangEdge]
) -> Any:
    try:
        from grandalf.graphs import Edge, Graph, Vertex  # type: ignore[import]
        from grandalf.layouts import SugiyamaLayout  # type: ignore[import]
        from grandalf.routing import (  # type: ignore[import]
            EdgeViewer,
            route_with_lines,
        )
    except ImportError as exc:
        msg = "Install grandalf to draw graphs: `pip install grandalf`."
        raise ImportError(msg) from exc

    #
    # Just a reminder about naming conventions:
    # +------------X
    # |
    # |
    # |
    # |
    # Y
    #

    vertices_ = {id: Vertex(f" {data} ") for id, data in vertices.items()}
    edges_ = [Edge(vertices_[s], vertices_[e], data=cond) for s, e, _, cond in edges]
    vertices_list = vertices_.values()
    graph = Graph(vertices_list, edges_)

    for vertex in vertices_list:
        vertex.view = VertexViewer(vertex.data)

    # NOTE: determine min box length to create the best layout
    minw = min(v.view.w for v in vertices_list)

    for edge in edges_:
        edge.view = EdgeViewer()

    sug = SugiyamaLayout(graph.C[0])
    graph = graph.C[0]
    roots = list(filter(lambda x: len(x.e_in()) == 0, graph.sV))

    sug.init_all(roots=roots, optimize=True)

    sug.yspace = VertexViewer.HEIGHT
    sug.xspace = minw
    sug.route_edge = route_with_lines

    sug.draw()

    return sug


def draw_ascii(vertices: Mapping[str, str], edges: Sequence[LangEdge]) -> str:
    """Build a DAG and draw it in ASCII.

    Args:
        vertices (list): list of graph vertices.
        edges (list): list of graph edges.

    Returns:
        str: ASCII representation

    Example:

        .. code-block:: python

            from langchain_core.runnables.graph_ascii import draw_ascii

            vertices = {1: "1", 2: "2", 3: "3", 4: "4"}
            edges = [
                (source, target, None, None)
                for source, target in [(1, 2), (2, 3), (2, 4), (1, 4)]
            ]


            print(draw_ascii(vertices, edges))

        .. code-block:: none

                 +---+
                 | 1 |
                 +---+
                 *    *
                *     *
               *       *
            +---+       *
            | 2 |       *
            +---+**     *
              *    **   *
              *      ** *
              *        **
            +---+     +---+
            | 3 |     | 4 |
            +---+     +---+
    """
    # NOTE: coordinates might me negative, so we need to shift
    # everything to the positive plane before we actually draw it.
    xlist: list[float] = []
    ylist: list[float] = []

    sug = _build_sugiyama_layout(vertices, edges)

    for vertex in sug.g.sV:
        # NOTE: moving boxes w/2 to the left
        xlist.extend(
            (
                vertex.view.xy[0] - vertex.view.w / 2.0,
                vertex.view.xy[0] + vertex.view.w / 2.0,
            )
        )
        ylist.extend((vertex.view.xy[1], vertex.view.xy[1] + vertex.view.h))

    for edge in sug.g.sE:
        for x, y in edge.view._pts:
            xlist.append(x)
            ylist.append(y)

    minx = min(xlist)
    miny = min(ylist)
    maxx = max(xlist)
    maxy = max(ylist)

    canvas_cols = int(math.ceil(math.ceil(maxx) - math.floor(minx))) + 1
    canvas_lines = int(round(maxy - miny))

    canvas = AsciiCanvas(canvas_cols, canvas_lines)

    # NOTE: first draw edges so that node boxes could overwrite them
    for edge in sug.g.sE:
        if len(edge.view._pts) <= 1:
            msg = "Not enough points to draw an edge"
            raise ValueError(msg)
        for index in range(1, len(edge.view._pts)):
            start = edge.view._pts[index - 1]
            end = edge.view._pts[index]

            start_x = int(round(start[0] - minx))
            start_y = int(round(start[1] - miny))
            end_x = int(round(end[0] - minx))
            end_y = int(round(end[1] - miny))

            if start_x < 0 or start_y < 0 or end_x < 0 or end_y < 0:
                msg = (
                    "Invalid edge coordinates: "
                    f"start_x={start_x}, "
                    f"start_y={start_y}, "
                    f"end_x={end_x}, "
                    f"end_y={end_y}"
                )
                raise ValueError(msg)

            canvas.line(start_x, start_y, end_x, end_y, "." if edge.data else "*")

    for vertex in sug.g.sV:
        # NOTE: moving boxes w/2 to the left
        x = vertex.view.xy[0] - vertex.view.w / 2.0
        y = vertex.view.xy[1]

        canvas.box(
            int(round(x - minx)),
            int(round(y - miny)),
            vertex.view.w,
            vertex.view.h,
        )

        canvas.text(int(round(x - minx)) + 1, int(round(y - miny)) + 1, vertex.data)

    return canvas.draw()
