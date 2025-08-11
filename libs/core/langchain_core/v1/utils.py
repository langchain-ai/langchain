"""V1 utility functions for messages."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages.content_blocks import (
    ToolCall,
)
from langchain_core.v1.messages import MessageV1

try:
    from IPython.display import HTML, Markdown, display

    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False


def pprint_content(message: MessageV1) -> str:
    """Pretty print content blocks from a v1 message.

    This utility formats the different content block types in a human-readable way,
    with visual indicators for each content type.

    Args:
        message: MessageV1 type message.

    Returns:
        A formatted string representation of the message content blocks.

    """
    if not message.content:
        return "📭 Empty message"

    output_parts = []

    for block in message.content:
        block_type = block.get("type")

        if block_type == "reasoning":
            reasoning_text = block.get("reasoning", "")
            if reasoning_text:
                output_parts.append(f"🤔 Reasoning:\n{reasoning_text}")

        elif block_type == "text":
            text_content = block.get("text", "")
            if text_content:
                output_parts.append(f"💬 Response:\n{text_content}")

        elif block_type == "tool_call":
            tool_block = cast("ToolCall", block)
            tool_name = tool_block.get("name", "unknown")
            tool_args = tool_block.get("args", {})
            tool_id = str(tool_block.get("id", ""))[:8]  # Show first 8 chars of ID
            args_str = _format_tool_args(tool_args)
            output_parts.append(f"🔧 Tool Call [{tool_id}]: {tool_name}\n{args_str}")

        elif block_type == "invalid_tool_call":
            tool_name = str(block.get("name", "unknown"))
            error = str(block.get("error", "unknown error"))
            output_parts.append(f"❌ Invalid Tool Call: {tool_name}\nError: {error}")

        elif block_type == "image":
            if "base64" in block:
                base64_data = str(block.get("base64", ""))
                if len(base64_data) > 20:
                    data_preview = f"{base64_data[:20]}..."
                else:
                    data_preview = base64_data
                output_parts.append(f"🖼️  Image (base64): {data_preview}")
            elif "url" in block:
                url = str(block.get("url", ""))
                output_parts.append(f"🖼️  Image (URL): {url}")
            else:
                output_parts.append("🖼️  Image (unknown format)")

        else:
            # Handle any other content block types
            output_parts.append(f"❓ Unknown content type: {block_type}")

    return "\n\n".join(output_parts)


def _format_tool_args(args: dict[str, Any]) -> str:
    """Format tool arguments for display.

    Args:
        args: Dictionary of tool arguments.

    Returns:
        Formatted string representation of the arguments.
    """
    if not args:
        return "  (no arguments)"

    formatted_args = []
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 50:
            # Truncate long string values
            formatted_args.append(f"  {key}: {value[:47]}...")
        else:
            formatted_args.append(f"  {key}: {value}")

    return "\n".join(formatted_args)


def print_message_content(message: MessageV1) -> None:
    """Print message content blocks in a pretty format.

    This is a convenience function that prints the output of
    ``pprint_content``.

    Args:
        message: The MessageV1 to print.
    """
    print(pprint_content(message))  # noqa: T201


def extract_reasoning_content(message: MessageV1) -> str:
    """Extract only the reasoning content from a message.

    Args:
        message: The MessageV1 to extract reasoning from.

    Returns:
        The reasoning content, or empty string if none found.
    """
    reasoning_parts: list[str] = []
    for block in message.content:
        if block.get("type") == "reasoning":
            reasoning_text = str(block.get("reasoning", ""))
            if reasoning_text:
                reasoning_parts.append(reasoning_text)

    return "\n\n".join(reasoning_parts)


def has_reasoning_content(message: MessageV1) -> bool:
    """Check if a message contains reasoning content blocks.

    Args:
        message: The MessageV1 to check for reasoning content.

    Returns:
        True if the message contains reasoning content blocks, False otherwise.
    """
    return any(
        block.get("type") == "reasoning" and block.get("reasoning")
        for block in message.content
    )


def display_message_content(message: MessageV1) -> None:
    """Display message content in Jupyter with rich formatting.

    This function provides enhanced display for Jupyter notebooks, rendering
    content blocks with proper formatting and Markdown support where applicable.
    Falls back to regular print if not in a Jupyter environment.

    Args:
        message: The MessageV1 to display.
    """
    if not JUPYTER_AVAILABLE:
        print_message_content(message)
        return

    if not message.content:
        display(Markdown("📭 **Empty message**"))
        return

    output_parts = []

    for block in message.content:
        block_type = block.get("type")

        if block_type == "reasoning":
            reasoning_text = block.get("reasoning", "")
            if reasoning_text:
                # Format reasoning as a collapsible section in Jupyter
                output_parts.append(f"""
### 🤔 Reasoning

<details>
<summary>Click to expand reasoning</summary>

{reasoning_text}

</details>
""")

        elif block_type == "text":
            text_content = block.get("text", "")
            if text_content:
                # Render text content as markdown, preserving LaTeX formatting
                output_parts.append(f"### 💬 Response\n\n{text_content}")

        elif block_type == "tool_call":
            tool_block = cast("ToolCall", block)
            tool_name = tool_block.get("name", "unknown")
            tool_args = tool_block.get("args", {})
            tool_id = str(tool_block.get("id", ""))[:8]

            args_md = _format_tool_args_markdown(tool_args)
            output_parts.append(f"""
### 🔧 Tool Call: `{tool_name}` [{tool_id}]

{args_md}
""")

        elif block_type == "invalid_tool_call":
            tool_name = str(block.get("name", "unknown"))
            error = str(block.get("error", "unknown error"))
            output_parts.append(f"""
### ❌ Invalid Tool Call: `{tool_name}`

**Error:** {error}
""")

        elif block_type == "image":
            if "base64" in block:
                base64_data = str(block.get("base64", ""))
                preview = (
                    f"{base64_data[:20]}..." if len(base64_data) > 20 else base64_data
                )
                output_parts.append(f"### 🖼️ Image (base64)\n\n`{preview}`")
            elif "url" in block:
                url = str(block.get("url", ""))
                output_parts.append(f"### 🖼️ Image\n\n![Image]({url})")
            else:
                output_parts.append("### 🖼️ Image (unknown format)")

        else:
            output_parts.append(f"### ❓ Unknown content type: `{block_type}`")

    # Display as markdown in Jupyter
    markdown_content = "\n\n---\n\n".join(output_parts)
    display(Markdown(markdown_content))


def _format_tool_args_markdown(args: dict[str, Any]) -> str:
    """Format tool arguments for Markdown display.

    Args:
        args: Dictionary of tool arguments.

    Returns:
        Formatted Markdown representation of the arguments.
    """
    if not args:
        return "*No arguments*"

    formatted_args = []
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 100:
            # Truncate very long string values for display
            formatted_args.append(f"- **{key}**: `{value[:97]}...`")
        else:
            formatted_args.append(f"- **{key}**: `{value}`")

    return "\n".join(formatted_args)


def display_message_content_html(message: MessageV1) -> None:
    """Display message content in Jupyter with HTML and enhanced LaTeX support.

    This function provides enhanced display for Jupyter notebooks using HTML
    with MathJax support for proper LaTeX rendering. Falls back to regular
    print if not in a Jupyter environment.

    Args:
        message: The MessageV1 to display.
    """
    if not JUPYTER_AVAILABLE:
        print_message_content(message)
        return

    if not message.content:
        display(
            HTML(
                "<div style='color: #666; font-style: italic;'>"
                "📭 <strong>Empty message</strong></div>"
            )
        )
        return

    html_parts = []

    # Add MathJax configuration for better LaTeX rendering
    mathjax_config = """
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
            processEscapes: true,
            processEnvironments: true
        },
        CommonHTML: { scale: 100 }
    });
    </script>
    """

    for block in message.content:
        block_type = block.get("type")

        if block_type == "reasoning":
            reasoning_text = block.get("reasoning", "")
            if reasoning_text:
                # Process LaTeX in reasoning text
                processed_text = _process_latex_for_html(reasoning_text)
                html_parts.append(
                    f"""
                <div style="border: 1px solid #ddd; border-radius: 8px;
                     padding: 15px; margin: 10px 0;
                     background-color: #f9f9f9;">
                    <h4 style="margin-top: 0; color: #333;">🤔 Reasoning</h4>
                    <details>
                        <summary style="cursor: pointer; font-weight: bold;
                                 color: #555;">Click to expand reasoning</summary>
                        <div style="margin-top: 10px; line-height: 1.5;">
                            {processed_text}
                        </div>
                    </details>
                </div>
                """
                )

        elif block_type == "text":
            text_content = block.get("text", "")
            if text_content:
                # Process LaTeX in text content
                processed_text = _process_latex_for_html(text_content)
                html_parts.append(
                    f"""
                <div style="border: 1px solid #ddd; border-radius: 8px;
                     padding: 15px; margin: 10px 0;
                     background-color: #f8f9fa;">
                    <h4 style="margin-top: 0; color: #333;">💬 Response</h4>
                    <div style="line-height: 1.6;">{processed_text}</div>
                </div>
                """
                )

        elif block_type == "tool_call":
            tool_block = cast("ToolCall", block)
            tool_name = tool_block.get("name", "unknown")
            tool_args = tool_block.get("args", {})
            tool_id = str(tool_block.get("id", ""))[:8]

            args_html = _format_tool_args_html(tool_args)
            html_parts.append(
                f"""
            <div style="border: 1px solid #ddd; border-radius: 8px;
                 padding: 15px; margin: 10px 0;
                 background-color: #fff8dc;">
                <h4 style="margin-top: 0; color: #333;">
                    🔧 Tool Call: <code>{tool_name}</code> [{tool_id}]
                </h4>
                {args_html}
            </div>
            """
            )

        elif block_type == "invalid_tool_call":
            tool_name = str(block.get("name", "unknown"))
            error = str(block.get("error", "unknown error"))
            html_parts.append(
                f"""
            <div style="border: 1px solid #dc3545; border-radius: 8px;
                 padding: 15px; margin: 10px 0;
                 background-color: #f8d7da;">
                <h4 style="margin-top: 0; color: #721c24;">
                    ❌ Invalid Tool Call: <code>{tool_name}</code>
                </h4>
                <div><strong>Error:</strong> {error}</div>
            </div>
            """
            )

        elif block_type == "image":
            if "base64" in block:
                base64_data = str(block.get("base64", ""))
                preview = (
                    f"{base64_data[:20]}..." if len(base64_data) > 20 else base64_data
                )
                html_parts.append(
                    f"""
                <div style="border: 1px solid #ddd; border-radius: 8px;
                     padding: 15px; margin: 10px 0;
                     background-color: #f0f8ff;">
                    <h4 style="margin-top: 0; color: #333;">🖼️ Image (base64)</h4>
                    <code style="background-color: #e9ecef; padding: 5px;
                          border-radius: 3px;">
                        {preview}
                    </code>
                </div>
                """
                )
            elif "url" in block:
                url = str(block.get("url", ""))
                html_parts.append(
                    f"""
                <div style="border: 1px solid #ddd; border-radius: 8px;
                     padding: 15px; margin: 10px 0;
                     background-color: #f0f8ff;">
                    <h4 style="margin-top: 0; color: #333;">🖼️ Image</h4>
                    <img src="{url}" style="max-width: 100%; height: auto;
                         border-radius: 4px;" alt="Image" />
                </div>
                """
                )
            else:
                html_parts.append(
                    """
                <div style="border: 1px solid #ddd; border-radius: 8px;
                     padding: 15px; margin: 10px 0;
                     background-color: #f0f8ff;">
                    <h4 style="margin-top: 0; color: #333;">
                        🖼️ Image (unknown format)
                    </h4>
                </div>
                """
                )

        else:
            html_parts.append(
                f"""
            <div style="border: 1px solid #ffc107; border-radius: 8px;
                 padding: 15px; margin: 10px 0;
                 background-color: #fff3cd;">
                <h4 style="margin-top: 0; color: #856404;">
                    ❓ Unknown content type: <code>{block_type}</code>
                </h4>
            </div>
            """
            )

    # Combine all HTML parts
    full_html = mathjax_config + "\n".join(html_parts)
    display(HTML(full_html))


def _process_latex_for_html(text: str) -> str:
    """Process LaTeX expressions in text for HTML display.

    Args:
        text: Text that may contain LaTeX expressions.

    Returns:
        Text with LaTeX expressions properly formatted for MathJax.
    """
    import re

    # Convert markdown-style math delimiters to MathJax format if needed
    # Handle inline math: \( ... \) or $ ... $
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text)

    # Handle display math: \[ ... \] or $$ ... $$
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)

    # Convert line breaks to HTML breaks for better formatting
    return text.replace("\n", "<br>")


def _format_tool_args_html(args: dict[str, Any]) -> str:
    """Format tool arguments for HTML display.

    Args:
        args: Dictionary of tool arguments.

    Returns:
        Formatted HTML representation of the arguments.
    """
    if not args:
        return "<div style='color: #666; font-style: italic;'>No arguments</div>"

    formatted_args = []
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 100:
            # Truncate very long string values for display
            formatted_args.append(
                f"<div><strong>{key}:</strong> <code>{value[:97]}...</code></div>"
            )
        else:
            formatted_args.append(
                f"<div><strong>{key}:</strong> <code>{value}</code></div>"
            )

    return "\n".join(formatted_args)
