"""MCP Apps (SEP-1865): which audience a tool is for, and how to say so.

An MCP App is a tool that returns a user interface instead of text. The tool
carries a `ui://` resource in its `_meta.ui`, a host renders that resource in a
sandboxed frame beside the conversation, and the tool's result goes into it.

That makes a tool's audience part of its definition. `_meta.ui.visibility`
names it: `["app"]` means the app may call the tool and the model may not,
because the app exists so a person decides and a model that can call the submit
tool can skip the person. **Absent means both audiences**, which is every tool
on every server that has never heard of this extension.

A host built on `MCPAdapter` therefore owes the extension two lines. It
advertises `MCP_APPS_EXTENSION` when it opens a client, and it filters:

```python
async with Client(url, extensions=[MCP_APPS_EXTENSION]) as client:
    async with MCPAdapter(client) as adapter:
        tools = filter_model_visible_tools(await adapter.list_tools())

agent = create_agent(model, tools=tools)
```

Rendering the view, reading the `ui://` resource and proxying a view's own
`tools/call` are the browser-facing half and are not in this module.
"""

from __future__ import annotations

from typing import Any, Final

from mcp.client.extension import ClientExtension, advertise

#: The MIME type every MCP App resource is served as.
UI_MIME_TYPE: Final = "text/html;profile=mcp-app"

#: The capability identifier a client advertises to say it can render apps.
UI_EXTENSION_ID: Final = "io.modelcontextprotocol/ui"

#: What a host declares during MCP's `initialize`, under
#: `ClientCapabilities.extensions`.
#:
#: Inert against a server that always sends `_meta.ui`, and load-bearing
#: against one that gates on the capability: the `ext-apps` SDK ships
#: `getUiCapability` for exactly that, and its documented example registers a
#: text-only tool for a client that did not advertise. Such a server leaves a
#: host with tools that work and no apps, and raises nothing on the way past.
#:
#: One value, shared: `advertise` returns an identifier and a settings dict and
#: holds nothing per client.
MCP_APPS_EXTENSION: Final[ClientExtension] = advertise(
    UI_EXTENSION_ID, {"mimeTypes": [UI_MIME_TYPE]}
)


def _raw_meta(tool: Any) -> dict[str, Any]:
    """A tool's `_meta`, from whichever kind of tool object this is.

    Both shapes turn up in one process: an MCP `Tool` off `list_tools()`
    carries `_meta` as `.meta`, and the same tool adapted by `MCPAdapter`
    keeps it under `metadata["mcp"]["tool"]["_meta"]`. Reading one and not the
    other leaves every tool with no declared visibility, which is
    indistinguishable from a server that has never heard of MCP Apps.
    """
    meta = getattr(tool, "meta", None)
    if isinstance(meta, dict):
        return meta

    metadata = getattr(tool, "metadata", None) or {}
    mcp = metadata.get("mcp") or {}
    return ((mcp.get("tool") or {}).get("_meta")) or {}


def ui_meta(tool: Any) -> dict[str, Any]:
    """The `_meta.ui` block a server declared on a tool, or an empty dict.

    Args:
        tool: An MCP `Tool`, or a LangChain tool adapted from one.

    Returns:
        The `ui` block, or `{}` for a tool that declares none.
    """
    ui = _raw_meta(tool).get("ui")
    return ui if isinstance(ui, dict) else {}


def app_uri(tool: Any) -> str | None:
    """The `ui://` resource this tool opens, or `None` if it opens none.

    Args:
        tool: An MCP `Tool`, or a LangChain tool adapted from one.

    Returns:
        The resource URI, or `None` for a tool that ships no view.
    """
    uri = ui_meta(tool).get("resourceUri")
    return uri if isinstance(uri, str) else None


def _visible_to(tool: Any, who: str) -> bool:
    """Whether `who` may call this tool, per the declared `visibility`.

    A `visibility` that is not a list is treated as absent, so a malformed
    value falls back to both audiences rather than to neither.
    """
    visibility = ui_meta(tool).get("visibility")
    return not isinstance(visibility, list) or who in visibility


def filter_model_visible_tools(tools: list[Any]) -> list[Any]:
    """The tools a model may be given.

    A tool whose `visibility` omits `"model"` MUST be kept out of the agent's
    tool list. This is the filter for that, and the only thing MCP Apps asks
    of an agent.

    Args:
        tools: Tools from one server, in either shape `_raw_meta` reads.

    Returns:
        The subset the model may be offered, in the order given.
    """
    return [tool for tool in tools if _visible_to(tool, "model")]


def filter_app_visible_tools(tools: list[Any]) -> list[Any]:
    """The tools a view may call.

    A host checks a view's `tools/call` against this and refuses anything the
    server did not open to apps. The view is server-authored HTML in a frame,
    so without the check the app side is a way to reach every tool.

    Args:
        tools: Tools from one server, in either shape `_raw_meta` reads.

    Returns:
        The subset a view may call, in the order given.
    """
    return [tool for tool in tools if _visible_to(tool, "app")]
