"""Mermaid graph drawing utilities."""

from __future__ import annotations

import asyncio
import base64
import random
import re
import string
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from langchain_core.runnables.graph import (
    CurveStyle,
    MermaidDrawMethod,
    NodeStyles,
)

if TYPE_CHECKING:
    from langchain_core.runnables.graph import Edge, Node


try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    from pyppeteer import launch  # type: ignore[import-not-found]

    _HAS_PYPPETEER = True
except ImportError:
    _HAS_PYPPETEER = False

MARKDOWN_SPECIAL_CHARS = "*_`"


def draw_mermaid(
    nodes: dict[str, Node],
    edges: list[Edge],
    *,
    first_node: str | None = None,
    last_node: str | None = None,
    with_styles: bool = True,
    curve_style: CurveStyle = CurveStyle.LINEAR,
    node_styles: NodeStyles | None = None,
    wrap_label_n_words: int = 9,
    frontmatter_config: dict[str, Any] | None = None,
) -> str:
    """Draws a Mermaid graph using the provided graph data.

    Args:
        nodes: List of node ids.
        edges: List of edges, object with a source, target and data.
        first_node: Id of the first node.
        last_node: Id of the last node.
        with_styles: Whether to include styles in the graph.
        curve_style: Curve style for the edges.
        node_styles: Node colors for different types.
        wrap_label_n_words: Words to wrap the edge labels.
        frontmatter_config: Mermaid frontmatter config.
            Can be used to customize theme and styles. Will be converted to YAML and
            added to the beginning of the mermaid graph.

            See more here: https://mermaid.js.org/config/configuration.html.

            Example config:

            ```python
            {
                "config": {
                    "theme": "neutral",
                    "look": "handDrawn",
                    "themeVariables": {"primaryColor": "#e2e2e2"},
                }
            }
            ```

    Returns:
        Mermaid graph syntax.

    """
    # Initialize Mermaid graph configuration
    original_frontmatter_config = frontmatter_config or {}
    original_flowchart_config = original_frontmatter_config.get("config", {}).get(
        "flowchart", {}
    )
    frontmatter_config = {
        **original_frontmatter_config,
        "config": {
            **original_frontmatter_config.get("config", {}),
            "flowchart": {**original_flowchart_config, "curve": curve_style.value},
        },
    }

    mermaid_graph = (
        (
            "---\n"
            + yaml.dump(frontmatter_config, default_flow_style=False)
            + "---\ngraph TD;\n"
        )
        if with_styles
        else "graph TD;\n"
    )
    # Group nodes by subgraph
    subgraph_nodes: dict[str, dict[str, Node]] = {}
    regular_nodes: dict[str, Node] = {}

    for key, node in nodes.items():
        if ":" in key:
            # For nodes with colons, add them only to their deepest subgraph level
            prefix = ":".join(key.split(":")[:-1])
            subgraph_nodes.setdefault(prefix, {})[key] = node
        else:
            regular_nodes[key] = node

    # Node formatting templates
    default_class_label = "default"
    format_dict = {default_class_label: "{0}({1})"}
    if first_node is not None:
        format_dict[first_node] = "{0}([{1}]):::first"
    if last_node is not None:
        format_dict[last_node] = "{0}([{1}]):::last"

    def render_node(key: str, node: Node, indent: str = "\t") -> str:
        """Helper function to render a node with consistent formatting."""
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
                + "\n".join(f"{k} = {value}" for k, value in node.metadata.items())
                + "</em></small>"
            )
        node_label = format_dict.get(key, format_dict[default_class_label]).format(
            _to_safe_id(key), label
        )
        return f"{indent}{node_label}\n"

    # Add non-subgraph nodes to the graph
    if with_styles:
        for key, node in regular_nodes.items():
            mermaid_graph += render_node(key, node)

    # Group edges by their common prefixes
    edge_groups: dict[str, list[Edge]] = {}
    for edge in edges:
        src_parts = edge.source.split(":")
        tgt_parts = edge.target.split(":")
        common_prefix = ":".join(
            src for src, tgt in zip(src_parts, tgt_parts, strict=False) if src == tgt
        )
        edge_groups.setdefault(common_prefix, []).append(edge)

    seen_subgraphs = set()

    def add_subgraph(edges: list[Edge], prefix: str) -> None:
        nonlocal mermaid_graph
        self_loop = len(edges) == 1 and edges[0].source == edges[0].target
        if prefix and not self_loop:
            subgraph = prefix.rsplit(":", maxsplit=1)[-1]
            if subgraph in seen_subgraphs:
                msg = (
                    f"Found duplicate subgraph '{subgraph}' -- this likely means that "
                    "you're reusing a subgraph node with the same name. "
                    "Please adjust your graph to have subgraph nodes with unique names."
                )
                raise ValueError(msg)

            seen_subgraphs.add(subgraph)
            mermaid_graph += f"\tsubgraph {subgraph}\n"

            # Add nodes that belong to this subgraph
            if with_styles and prefix in subgraph_nodes:
                for key, node in subgraph_nodes[prefix].items():
                    mermaid_graph += render_node(key, node)

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
                f"\t{_to_safe_id(source)}{edge_label}{_to_safe_id(target)};\n"
            )

        # Recursively add nested subgraphs
        for nested_prefix, edges_ in edge_groups.items():
            if not nested_prefix.startswith(prefix + ":") or nested_prefix == prefix:
                continue
            # only go to first level subgraphs
            if ":" in nested_prefix[len(prefix) + 1 :]:
                continue
            add_subgraph(edges_, nested_prefix)

        if prefix and not self_loop:
            mermaid_graph += "\tend\n"

    # Start with the top-level edges (no common prefix)
    add_subgraph(edge_groups.get("", []), "")

    # Add remaining subgraphs with edges
    for prefix, edges_ in edge_groups.items():
        if not prefix or ":" in prefix:
            continue
        add_subgraph(edges_, prefix)
        seen_subgraphs.add(prefix)

    # Add empty subgraphs (subgraphs with no internal edges)
    if with_styles:
        for prefix, subgraph_node in subgraph_nodes.items():
            if ":" not in prefix and prefix not in seen_subgraphs:
                mermaid_graph += f"\tsubgraph {prefix}\n"

                # Add nodes that belong to this subgraph
                for key, node in subgraph_node.items():
                    mermaid_graph += render_node(key, node)

                mermaid_graph += "\tend\n"
                seen_subgraphs.add(prefix)

    # Add custom styles for nodes
    if with_styles:
        mermaid_graph += _generate_mermaid_graph_styles(node_styles or NodeStyles())
    return mermaid_graph


def _to_safe_id(label: str) -> str:
    """Convert a string into a Mermaid-compatible node id.

    Keep [a-zA-Z0-9_-] characters unchanged.
    Map every other character -> backslash + lowercase hex codepoint.

    Result is guaranteed to be unique and Mermaid-compatible,
    so nodes with special characters always render correctly.
    """
    allowed = string.ascii_letters + string.digits + "_-"
    out = [ch if ch in allowed else "\\" + format(ord(ch), "x") for ch in label]
    return "".join(out)


def _generate_mermaid_graph_styles(node_colors: NodeStyles) -> str:
    """Generates Mermaid graph styles for different node types."""
    styles = ""
    for class_name, style in asdict(node_colors).items():
        styles += f"\tclassDef {class_name} {style}\n"
    return styles


def draw_mermaid_png(
    mermaid_syntax: str,
    output_file_path: str | None = None,
    draw_method: MermaidDrawMethod = MermaidDrawMethod.API,
    background_color: str | None = "white",
    padding: int = 10,
    max_retries: int = 1,
    retry_delay: float = 1.0,
    base_url: str | None = None,
    proxies: dict[str, str] | None = None,
) -> bytes:
    """Draws a Mermaid graph as PNG using provided syntax.

    Args:
        mermaid_syntax: Mermaid graph syntax.
        output_file_path: Path to save the PNG image.
        draw_method: Method to draw the graph.
        background_color: Background color of the image.
        padding: Padding around the image.
        max_retries: Maximum number of retries (MermaidDrawMethod.API).
        retry_delay: Delay between retries (MermaidDrawMethod.API).
        base_url: Base URL for the Mermaid.ink API.
        proxies: HTTP/HTTPS proxies for requests (e.g. `{"http": "http://127.0.0.1:7890"}`).

    Returns:
        PNG image bytes.

    Raises:
        ValueError: If an invalid draw method is provided.
    """
    if draw_method == MermaidDrawMethod.PYPPETEER:
        img_bytes = asyncio.run(
            _render_mermaid_using_pyppeteer(
                mermaid_syntax, output_file_path, background_color, padding
            )
        )
    elif draw_method == MermaidDrawMethod.API:
        img_bytes = _render_mermaid_using_api(
            mermaid_syntax,
            output_file_path=output_file_path,
            background_color=background_color,
            max_retries=max_retries,
            retry_delay=retry_delay,
            base_url=base_url,
            proxies=proxies,
        )
    else:
        supported_methods = ", ".join([m.value for m in MermaidDrawMethod])
        msg = (
            f"Invalid draw method: {draw_method}. "
            f"Supported draw methods are: {supported_methods}"
        )
        raise ValueError(msg)

    return img_bytes


async def _render_mermaid_using_pyppeteer(
    mermaid_syntax: str,
    output_file_path: str | None = None,
    background_color: str | None = "white",
    padding: int = 10,
    device_scale_factor: int = 3,
) -> bytes:
    """Renders Mermaid graph using Pyppeteer."""
    if not _HAS_PYPPETEER:
        msg = "Install Pyppeteer to use the Pyppeteer method: `pip install pyppeteer`."
        raise ImportError(msg)

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
        await asyncio.get_event_loop().run_in_executor(
            None, Path(output_file_path).write_bytes, img_bytes
        )

    return img_bytes


def _render_mermaid_using_api(
    mermaid_syntax: str,
    *,
    output_file_path: str | None = None,
    background_color: str | None = "white",
    file_type: Literal["jpeg", "png", "webp"] | None = "png",
    max_retries: int = 1,
    retry_delay: float = 1.0,
    proxies: dict[str, str] | None = None,
    base_url: str | None = None,
) -> bytes:
    """Renders Mermaid graph using the Mermaid.INK API."""
    # Defaults to using the public mermaid.ink server.
    base_url = base_url if base_url is not None else "https://mermaid.ink"

    if not _HAS_REQUESTS:
        msg = (
            "Install the `requests` module to use the Mermaid.INK API: "
            "`pip install requests`."
        )
        raise ImportError(msg)

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
        f"{base_url}/img/{mermaid_syntax_encoded}"
        f"?type={file_type}&bgColor={background_color}"
    )

    error_msg_suffix = (
        "To resolve this issue:\n"
        "1. Check your internet connection and try again\n"
        "2. Try with higher retry settings: "
        "`draw_mermaid_png(..., max_retries=5, retry_delay=2.0)`\n"
        "3. Use the Pyppeteer rendering method which will render your graph locally "
        "in a browser: `draw_mermaid_png(..., draw_method=MermaidDrawMethod.PYPPETEER)`"
    )

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(image_url, timeout=10, proxies=proxies)
            if response.status_code == requests.codes.ok:
                img_bytes = response.content
                if output_file_path is not None:
                    Path(output_file_path).write_bytes(response.content)

                return img_bytes

            # If we get a server error (5xx), retry
            if (
                requests.codes.internal_server_error <= response.status_code
                and attempt < max_retries
            ):
                # Exponential backoff with jitter
                sleep_time = retry_delay * (2**attempt) * (0.5 + 0.5 * random.random())  # noqa: S311 not used for crypto
                time.sleep(sleep_time)
                continue

            # For other status codes, fail immediately
            msg = (
                f"Failed to reach {base_url} API while trying to render "
                f"your graph. Status code: {response.status_code}.\n\n"
            ) + error_msg_suffix
            raise ValueError(msg)

        except (requests.RequestException, requests.Timeout) as e:
            if attempt < max_retries:
                # Exponential backoff with jitter
                sleep_time = retry_delay * (2**attempt) * (0.5 + 0.5 * random.random())  # noqa: S311 not used for crypto
                time.sleep(sleep_time)
            else:
                msg = (
                    f"Failed to reach {base_url} API while trying to render "
                    f"your graph after {max_retries} retries. "
                ) + error_msg_suffix
                raise ValueError(msg) from e

    # This should not be reached, but just in case
    msg = (
        f"Failed to reach {base_url} API while trying to render "
        f"your graph after {max_retries} retries. "
    ) + error_msg_suffix
    raise ValueError(msg)
