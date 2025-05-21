from typing_extensions import TypedDict
from IPython.display import HTML, display


def show_lean_canvas(state) -> None:
    """Отрисовывает Lean Canvas в Jupyter-ноутбуке как HTML-сетку."""
    # --- CSS для сетки 5×2 + нижний ряд -----------------------------------
    css = """
    <style>
    .canvas {
        display: grid;
        grid-template-columns: 13% 30% 13% 30% 13%;   /* ширины колонок */
        grid-template-rows: auto auto auto auto;           /* Title + 2 ряда + низ   */
        gap: 8px;
        background: transparent;
        font-family: Arial, sans-serif;
    }
    .box {
        background:#e59a12;
        color:#fff;
        border:1px solid #fff;
        padding:12px 14px;
        line-height:1.3;
    }
    .title { font-weight:700; margin-bottom:6px; }
    .canvas-title-cell { /* New class for the title cell */
        grid-column: 1 / -1; /* Span all columns */
        grid-row: 1 / span 1;    /* First row */
        text-align: center;
        color: #08c; /* Copied from original h2 */
        font-family: Arial, sans-serif; /* Copied from original h2 */
        padding: 8px 0; /* Vertical padding */
        font-size: 1.3em;
        font-weight: bold;
    }
    /* раскладка по «ячейкам» */
    .problem           { grid-area: 2 / 1 / span 2 / span 1; } /* Shifted down */
    .solution          { grid-area: 2 / 2 / span 1 / span 1; } /* Shifted down */
    .key_metrics       { grid-area: 3 / 2 / span 1 / span 1; } /* Shifted down */
    .uvp               { grid-area: 2 / 3 / span 2 / span 1; } /* Shifted down */
    .unfair            { grid-area: 2 / 4 / span 1 / span 1; } /* Shifted down */
    .channels          { grid-area: 3 / 4 / span 1 / span 1; } /* Shifted down */
    .customer_segments { grid-area: 2 / 5 / span 2 / span 1; } /* Shifted down */
    .cost_structure    { grid-area: 4 / 1 / span 1 / span 3; } /* Shifted down */
    .revenue_streams   { grid-area: 4 / 4 / span 1 / span 2; } /* Shifted down */
    </style>
    """

    # --- HTML-разметка ------------------------------------------------------
    html = f"""
    {css}
    <div class="canvas">
        <div class="canvas-title-cell">{state['main_task'].replace("\n", "<br>")}</div>

        <div class="box problem">
            <div class="title">2. Problem</div>
            {state['problem'].replace("\n", "<br>")}
        </div>

        <div class="box solution">
            <div class="title">4. Solution</div>
            {state['solution'].replace("\n", "<br>")}
        </div>

        <div class="box key_metrics">
            <div class="title">8. Key Metrics</div>
            {state['key_metrics'].replace("\n", "<br>")}
        </div>

        <div class="box uvp">
            <div class="title">3. Unique Value Proposition</div>
            {state['unique_value_proposition'].replace("\n", "<br>")}
        </div>

        <div class="box unfair">
            <div class="title">9. Unfair Advantage</div>
            {state['unfair_advantage'].replace("\n", "<br>")}
        </div>

        <div class="box channels">
            <div class="title">5. Channels</div>
            {state['channels'].replace("\n", "<br>")}
        </div>

        <div class="box customer_segments">
            <div class="title">1. Customer Segments</div>
            {state['customer_segments'].replace("\n", "<br>")}
        </div>

        <div class="box cost_structure">
            <div class="title">7. Cost Structure</div>
            {state['cost_structure'].replace("\n", "<br>")}
        </div>

        <div class="box revenue_streams">
            <div class="title">6. Revenue Streams</div>
            {state['revenue_streams'].replace("\n", "<br>")}
        </div>

    </div>
    """
    display(HTML(html))


def show_graph(graph, xray=False):
    """Display a LangGraph mermaid diagram with fallback rendering.

    Handles timeout errors from mermaid.ink by falling back to pyppeteer.

    Args:
        graph: The LangGraph object that has a get_graph() method
    """
    from IPython.display import Image

    try:
        # Try the default renderer first
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception as e:
        # Fall back to pyppeteer if the default renderer fails
        import nest_asyncio

        nest_asyncio.apply()
        from langchain_core.runnables.graph import MermaidDrawMethod

        return Image(
            graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
        )