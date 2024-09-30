import base64
import re
from dataclasses import asdict
from typing import Optional

from langchain_core.runnables.graph import (
    CurveStyle,
    Edge,
    MermaidDrawMethod,
    Node,
    NodeStyles,
)

MARKDOWN_SPECIAL_CHARS = "*_`"


def draw_mermaid(
    nodes: dict[str, Node],
    edges: list[Edge],
    *,
    first_node: Optional[str] = None,
    last_node: Optional[str] = None,
    with_styles: bool = True,
    curve_style: CurveStyle = CurveStyle.LINEAR,
    node_styles: Optional[NodeStyles] = None,
    wrap_label_n_words: int = 9,
) -> str:
    """Draws a Mermaid graph using the provided graph data.

    Args:
        nodes (dict[str, str]): List of node ids.
        edges (List[Edge]): List of edges, object with a source,
            target and data.
        first_node (str, optional): Id of the first node. Defaults to None.
        last_node (str, optional): Id of the last node. Defaults to None.
        with_styles (bool, optional): Whether to include styles in the graph.
            Defaults to True.
        curve_style (CurveStyle, optional): Curve style for the edges.
            Defaults to CurveStyle.LINEAR.
        node_styles (NodeStyles, optional): Node colors for different types.
            Defaults to NodeStyles().
        wrap_label_n_words (int, optional): Words to wrap the edge labels.
            Defaults to 9.

    Returns:
        str: Mermaid graph syntax.
    """
    # Initialize Mermaid graph configuration
    mermaid_graph = (
        (
            f"%%{{init: {{'flowchart': {{'curve': '{curve_style.value}'"
            f"}}}}}}%%\ngraph TD;\n"
        )
        if with_styles
        else "graph TD;\n"
    )

    if with_styles:
        # Node formatting templates
        default_class_label = "default"
        format_dict = {default_class_label: "{0}({1})"}
        if first_node is not None:
            format_dict[first_node] = "{0}([{1}]):::first"
        if last_node is not None:
            format_dict[last_node] = "{0}([{1}]):::last"

        # Add nodes to the graph
        for key, node in nodes.items():
            node_name = node.name.split(":")[-1]
            label = (
                f"<p>{node_name}</p>"
                if node_name.startswith(tuple(MARKDOWN_SPECIAL_CHARS))
                and node_name.endswith(tuple(MARKDOWN_SPECIAL_CHARS))
                else node_name
            )
            if node.metadata:
                label = (
                    f"{label}<hr/><small><em>"
                    + "\n".join(
                        f"{key} = {value}" for key, value in node.metadata.items()
                    )
                    + "</em></small>"
                )
            node_label = format_dict.get(key, format_dict[default_class_label]).format(
                _escape_node_label(key), label
            )
            mermaid_graph += f"\t{node_label}\n"

    # Group edges by their common prefixes
    edge_groups: dict[str, list[Edge]] = {}
    for edge in edges:
        src_parts = edge.source.split(":")
        tgt_parts = edge.target.split(":")
        common_prefix = ":".join(
            src for src, tgt in zip(src_parts, tgt_parts) if src == tgt
        )
        edge_groups.setdefault(common_prefix, []).append(edge)

    seen_subgraphs = set()

    def add_subgraph(edges: list[Edge], prefix: str) -> None:
        nonlocal mermaid_graph
        self_loop = len(edges) == 1 and edges[0].source == edges[0].target
        if prefix and not self_loop:
            subgraph = prefix.split(":")[-1]
            if subgraph in seen_subgraphs:
                raise ValueError(
                    f"Found duplicate subgraph '{subgraph}' -- this likely means that "
                    "you're reusing a subgraph node with the same name. "
                    "Please adjust your graph to have subgraph nodes with unique names."
                )

            seen_subgraphs.add(subgraph)
            mermaid_graph += f"\tsubgraph {subgraph}\n"

        for edge in edges:
            source, target = edge.source, edge.target

            # Add BR every wrap_label_n_words words
            if edge.data is not None:
                edge_data = edge.data
                words = str(edge_data).split()  # Split the string into words
                # Group words into chunks of wrap_label_n_words size
                if len(words) > wrap_label_n_words:
                    edge_data = "&nbsp<br>&nbsp".join(
                        " ".join(words[i : i + wrap_label_n_words])
                        for i in range(0, len(words), wrap_label_n_words)
                    )
                if edge.conditional:
                    edge_label = f" -. &nbsp;{edge_data}&nbsp; .-> "
                else:
                    edge_label = f" -- &nbsp;{edge_data}&nbsp; --> "
            else:
                edge_label = " -.-> " if edge.conditional else " --> "

            mermaid_graph += (
                f"\t{_escape_node_label(source)}{edge_label}"
                f"{_escape_node_label(target)};\n"
            )

        # Recursively add nested subgraphs
        for nested_prefix in edge_groups:
            if not nested_prefix.startswith(prefix + ":") or nested_prefix == prefix:
                continue
            add_subgraph(edge_groups[nested_prefix], nested_prefix)

        if prefix and not self_loop:
            mermaid_graph += "\tend\n"

    # Start with the top-level edges (no common prefix)
    add_subgraph(edge_groups.get("", []), "")

    # Add remaining subgraphs
    for prefix in edge_groups:
        if ":" in prefix or prefix == "":
            continue
        add_subgraph(edge_groups[prefix], prefix)

    # Add custom styles for nodes
    if with_styles:
        mermaid_graph += _generate_mermaid_graph_styles(node_styles or NodeStyles())
    return mermaid_graph


def _escape_node_label(node_label: str) -> str:
    """Escapes the node label for Mermaid syntax."""
    return re.sub(r"[^a-zA-Z-_0-9]", "_", node_label)


def _generate_mermaid_graph_styles(node_colors: NodeStyles) -> str:
    """Generates Mermaid graph styles for different node types."""
    styles = ""
    for class_name, style in asdict(node_colors).items():
        styles += f"\tclassDef {class_name} {style}\n"
    return styles


def draw_mermaid_png(
    mermaid_syntax: str,
    output_file_path: Optional[str] = None,
    draw_method: MermaidDrawMethod = MermaidDrawMethod.API,
    background_color: Optional[str] = "white",
    padding: int = 10,
) -> bytes:
    """Draws a Mermaid graph as PNG using provided syntax.

    Args:
        mermaid_syntax (str): Mermaid graph syntax.
        output_file_path (str, optional): Path to save the PNG image.
            Defaults to None.
        draw_method (MermaidDrawMethod, optional): Method to draw the graph.
            Defaults to MermaidDrawMethod.API.
        background_color (str, optional): Background color of the image.
            Defaults to "white".
        padding (int, optional): Padding around the image. Defaults to 10.

    Returns:
        bytes: PNG image bytes.

    Raises:
        ValueError: If an invalid draw method is provided.
    """
    if draw_method == MermaidDrawMethod.PYPPETEER:
        import asyncio

        img_bytes = asyncio.run(
            _render_mermaid_using_pyppeteer(
                mermaid_syntax, output_file_path, background_color, padding
            )
        )
    elif draw_method == MermaidDrawMethod.API:
        img_bytes = _render_mermaid_using_api(
            mermaid_syntax, output_file_path, background_color
        )
    else:
        supported_methods = ", ".join([m.value for m in MermaidDrawMethod])
        raise ValueError(
            f"Invalid draw method: {draw_method}. "
            f"Supported draw methods are: {supported_methods}"
        )

    return img_bytes


async def _render_mermaid_using_pyppeteer(
    mermaid_syntax: str,
    output_file_path: Optional[str] = None,
    background_color: Optional[str] = "white",
    padding: int = 10,
    device_scale_factor: int = 3,
) -> bytes:
    """Renders Mermaid graph using Pyppeteer."""
    try:
        from pyppeteer import launch  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "Install Pyppeteer to use the Pyppeteer method: `pip install pyppeteer`."
        ) from e

    browser = await launch()
    page = await browser.newPage()

    # Setup Mermaid JS
    await page.goto("about:blank")
    await page.addScriptTag(
        {"url": "https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"}
    )
    await page.evaluate(
        """() => {
                mermaid.initialize({startOnLoad:true});
            }"""
    )

    # Render SVG
    svg_code = await page.evaluate(
        """(mermaidGraph) => {
                return mermaid.mermaidAPI.render('mermaid', mermaidGraph);
            }""",
        mermaid_syntax,
    )

    # Set the page background to white
    await page.evaluate(
        """(svg, background_color) => {
            document.body.innerHTML = svg;
            document.body.style.background = background_color;
        }""",
        svg_code["svg"],
        background_color,
    )

    # Take a screenshot
    dimensions = await page.evaluate(
        """() => {
            const svgElement = document.querySelector('svg');
            const rect = svgElement.getBoundingClientRect();
            return { width: rect.width, height: rect.height };
        }"""
    )
    await page.setViewport(
        {
            "width": int(dimensions["width"] + padding),
            "height": int(dimensions["height"] + padding),
            "deviceScaleFactor": device_scale_factor,
        }
    )

    img_bytes = await page.screenshot({"fullPage": False})
    await browser.close()

    if output_file_path is not None:
        with open(output_file_path, "wb") as file:
            file.write(img_bytes)

    return img_bytes


def _render_mermaid_using_api(
    mermaid_syntax: str,
    output_file_path: Optional[str] = None,
    background_color: Optional[str] = "white",
) -> bytes:
    """Renders Mermaid graph using the Mermaid.INK API."""
    try:
        import requests  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "Install the `requests` module to use the Mermaid.INK API: "
            "`pip install requests`."
        ) from e

    # Use Mermaid API to render the image
    mermaid_syntax_encoded = base64.b64encode(mermaid_syntax.encode("utf8")).decode(
        "ascii"
    )

    # Check if the background color is a hexadecimal color code using regex
    if background_color is not None:
        hex_color_pattern = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")
        if not hex_color_pattern.match(background_color):
            background_color = f"!{background_color}"

    image_url = (
        f"https://mermaid.ink/img/{mermaid_syntax_encoded}?bgColor={background_color}"
    )
    response = requests.get(image_url)
    if response.status_code == 200:
        img_bytes = response.content
        if output_file_path is not None:
            with open(output_file_path, "wb") as file:
                file.write(response.content)

        return img_bytes
    else:
        raise ValueError(
            f"Failed to render the graph using the Mermaid.INK API. "
            f"Status code: {response.status_code}."
        )
