from dataclasses import dataclass, asdict
from langchain_core.runnables.graph import Edge
from enum import Enum


class CurveStyle(Enum):
    """Enum for different curve styles supported by Mermaid"""
    BASIS = "basis"
    BUMP_X = "bumpX"
    BUMP_Y = "bumpY"
    CARDINAL = "cardinal"
    CATMULL_ROM = "catmullRom"
    LINEAR = "linear"
    MONOTONE_X = "monotoneX"
    MONOTONE_Y = "monotoneY"
    NATURAL = "natural"
    STEP = "step"
    STEP_AFTER = "stepAfter"
    STEP_BEFORE = "stepBefore"


@dataclass
class NodeColors:
    """Schema for Hexadecimal color codes for different node types"""
    start: str = "#ffdfba"
    end: str = "#baffc9"
    other: str = "#fad7de"



def draw_mermaid(node_ids: list[str], edges: list[Edge], curve_style: CurveStyle = CurveStyle.LINEAR, node_colors: NodeColors = NodeColors(), wrap_label_n_words: int = 9) -> str:
    """Draws a Mermaid graph using the provided graph data

    Args:
        node_ids (list[str]): List of node ids
        edges (list[Edge]): List of edges, object with source, target and data (optionally).
        curve_style (CurveStyle, optional): Curve style for the edges supported by Mermaid. Defaults to CurveStyle.LINEAR.
        node_colors (NodeColors, optional): Node colors for the different node types. Defaults to the default colors for NodeColors class.
        wrap_label_n_words (int, optional): Number of words to wrap the edge labels. Defaults to 9.

    Returns:
        str: Mermaid graph syntax

    Example: 
        >>> node_ids = ['A', 'B', 'C', 'D']
        >>> edges = [Edge('A', 'B'), Edge('B', 'C'), Edge('B', 'D'), Edge('A', 'D', 'end')]
        >>> mermaid_syntax = draw_mermaid(node_ids, edges, curve_style=CurveStyle.LINEAR, node_colors=NodeColors(start="#ffdfba", end="#baffc9", other="#fad7de"), wrap_label_n_words=9)
        >>> print(mermaid_syntax)
        %%{init: {'flowchart': {'curve': 'linear', 'defaultRenderer': 'elk'}}}%%
        graph TD;
            A([A]):::otherclass;
            B([B]):::otherclass;
            C([C]):::otherclass;
            D([D]):::otherclass;
            A --> B;
            B --> C;
            B --> D;
            A -- end --> D;
            classDef startclass fill:#ffdfba;
            classDef endclass fill:#baffc9;
            classDef otherclass fill:#fad7de;
        """
    # Initialize Mermaid graph configuration
    mermaid_graph = f"%%{{init: {{'flowchart': {{'curve': '{curve_style.value}', 'defaultRenderer': 'elk'}}}}}}%%\ngraph TD;\n"

    # Node formatting templates
    format_dict = {
        '__start__': '{0}[{0}]:::startclass',
        '__end__': '{0}[{0}]:::endclass',
        'default': '{0}([{0}]):::otherclass'
    }

    # Filter out nodes that were created due to conditional edges
    pure_nodes = [n for n in node_ids if 'edges' not in n]

    # Add __end__ node if it is in any of the edges.target
    if any('__end__' in edge.target for edge in edges):
        pure_nodes.append('__end__')

    # Add nodes to the graph
    for node in pure_nodes:
        node_label = format_dict.get(node, format_dict['default']).format(node)
        mermaid_graph += f"\t{node_label};\n"

    # Add edges to the graph
    for edge in edges:
        source, target = _adjust_mermaid_edge(edge, pure_nodes)
        if source == target:  # Ignore loops
            continue

        # Add BR every wrap_label_n_words words
        if edge.data is not None:
            edge_data = edge.data
            words = edge_data.split()  # Split the string into words
            # Group words into chunks of wrap_label_n_words size
            if len(words) > wrap_label_n_words:
                edge_data = "<br>".join(
                    [' '.join(words[i:i + wrap_label_n_words]) for i in range(0, len(words), wrap_label_n_words)])
            edge_label = f" -- {edge_data} --> "
        else:
            edge_label = " --> "
        mermaid_graph += f"\t{source}{edge_label}{target};\n"

    # Add custom styles for nodes
    mermaid_graph += _generate_mermaid_graph_styles(node_colors)
    return mermaid_graph


def _adjust_mermaid_edge(edge: Edge, pure_nodes: list[str]) -> tuple[str, str]:
    """ If the node is a result of a conditional edge, map to the pure node """
    adjust_node = lambda node: next((n for n in pure_nodes if n in node), node)
    return adjust_node(edge.source), adjust_node(edge.target)


def _generate_mermaid_graph_styles(node_colors: NodeColors) -> str:
    """Generates Mermaid graph styles for different node types"""
    styles = ""
    for class_name, color in asdict(node_colors).items():
        styles += f"\tclassDef {class_name}class fill:{color};\n"
    return styles


class MermaidDrawMethod(Enum):
    PYPPETEER = "pyppeteer" # Uses Pyppeteer to render the graph
    API = "api" # Uses Mermaid.INK API to render the graph


def draw_mermaid_png(mermaid_syntax: str, output_file: str = 'graph.png', draw_method: MermaidDrawMethod = MermaidDrawMethod.PYPPETEER):
    """Draws a Mermaid graph using the provided graph data and saves it as a PNG image

    Args:
        mermaid_syntax (str): Mermaid graph syntax. Generated using `draw_mermaid` method.
        output_file (str, optional): Output file path. Defaults to "graph.png".
        draw_method (MermaidDrawMethod, optional): Method to use for drawing the graph. The methods available and their usage are: 
        - MermaidDrawMethod.API: Uses Mermaid.INK API to render the graph (default)
        - MermaidDrawMethod.PYPPETEER: Uses Pyppeteer to render the graph

    Example:
        >>> node_ids = ['A', 'B', 'C', 'D']
        >>> edges = [Edge('A', 'B'), Edge('B', 'C'), Edge('B', 'D'), Edge('A', 'D', 'end')]
        >>> mermaid_syntax = draw_mermaid(node_ids, edges)
        >>> draw_mermaid_png(mermaid_syntax, output_file='graph_api.png', draw_method=MermaidDrawMethod.API)
        >>> # draw_mermaid_png(mermaid_syntax, output_file='graph_pyppeteer.png', draw_method=MermaidDrawMethod.PYPPETEER) # (Optionally if Pyppeteer is installed)
    """
    if draw_method == MermaidDrawMethod.PYPPETEER:
        import asyncio
        asyncio.run(_render_mermaid_using_pyppeteer(mermaid_syntax, output_file))
    elif draw_method == MermaidDrawMethod.API:
        _render_mermaid_using_api(mermaid_syntax,output_file)
    else:
        raise ValueError(f"Invalid draw method: {draw_method}. Supported draw methods are: {', '.join([m.value for m in MermaidDrawMethod])}")

async def _render_mermaid_using_pyppeteer(mermaid_syntax: str, output_file: str):   
    """Renders the Mermaid graph using Pyppeteer and saves it as a PNG image. It requires Pyppeteer to be installed. """ 
    try:
        from pyppeteer import launch
    except ImportError:
        raise ImportError("Install Pyppeteer to use the Pyppeteer method: `pip install pyppeteer`.")

    browser = await launch()
    page = await browser.newPage()

    # Setup Mermaid JS
    await page.goto('about:blank')
    await page.addScriptTag({'url': 'https://unpkg.com/mermaid/dist/mermaid.min.js'})
    await page.evaluate('''() => {
                mermaid.initialize({startOnLoad:true});
            }''')

    # Render SVG
    svg_code = await page.evaluate(f"""(mermaidGraph) => {{
                return mermaid.mermaidAPI.render('mermaid', mermaidGraph);
            }}""", mermaid_syntax)

    # Set the page background to white
    await page.evaluate('''(svg) => {
            document.body.innerHTML = svg;
            document.body.style.background = 'white'; // Set the page background to white
        }''', svg_code['svg'])

    # Take a screenshot
    dimensions = await page.evaluate('''() => {
            const svgElement = document.querySelector('svg');
            const rect = svgElement.getBoundingClientRect();
            return { width: rect.width, height: rect.height };
        }''')
    padding = 10  # Reduce padding to 10px or to a value that works for your case
    await page.setViewport({
        'width': int(dimensions['width'] + padding),
        'height': int(dimensions['height'] + padding)
    })
    await page.screenshot({'path': output_file, 'fullPage': False})
    await browser.close()

def _render_mermaid_using_api(mermaid_syntax: str, output_file: str):
    """Renders the Mermaid graph using the Mermaid.INK API and saves it as a PNG image."""
    import base64
    try:
        import requests
    except ImportError:
        raise ImportError("Install the `requests` module to use the Mermaid.INK API: `pip install requests`.")

    # Use Mermaid API to render the image
    mermaid_syntax_encoded = base64.b64encode(mermaid_syntax.encode("utf8")).decode("ascii")
    image_url = f"https://mermaid.ink/img/{mermaid_syntax_encoded}"
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(output_file, "wb") as file:
            file.write(response.content)
    else:
        raise Exception("Failed to render the image, make sure your graph is correct.")
