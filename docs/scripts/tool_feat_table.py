import sys
from pathlib import Path

SEARCH_TOOL_FEAT_TABLE = {
    "Exa Search": {
        "pricing": "1000 free searches/month",
        "available_data": "URL, Author, Title, Published Date",
        "link": "/docs/integrations/tools/exa_search",
    },
    "Bing Search": {
        "pricing": "Paid",
        "available_data": "URL, Snippet, Title",
        "link": "/docs/integrations/tools/bing_search",
    },
    "DuckDuckgoSearch": {
        "pricing": "Free",
        "available_data": "URL, Snippet, Title",
        "link": "/docs/integrations/tools/ddg",
    },
    "Brave Search": {
        "pricing": "Free",
        "available_data": "URL, Snippet, Title",
        "link": "/docs/integrations/tools/brave_search",
    },
    "Google Search": {
        "pricing": "Paid",
        "available_data": "URL, Snippet, Title",
        "link": "/docs/integrations/tools/google_search",
    },
    "Google Serper": {
        "pricing": "Free",
        "available_data": "URL, Snippet, Title, Search Rank, Site Links",
        "link": "/docs/integrations/tools/google_serper",
    },
    "Mojeek Search": {
        "pricing": "Paid",
        "available_data": "URL, Snippet, Title",
        "link": "/docs/integrations/tools/mojeek_search",
    },
    "SearxNG Search": {
        "pricing": "Free",
        "available_data": "URL, Snippet, Title, Category",
        "link": "/docs/integrations/tools/searx_search",
    },
    "You.com Search": {
        "pricing": "Free for 60 days",
        "available_data": "URL, Title, Page Content",
        "link": "/docs/integrations/tools/you",
    },
    "SearchApi": {
        "pricing": "100 Free Searches on Sign Up",
        "available_data": "URL, Snippet, Title, Search Rank, Site Links, Authors",
        "link": "/docs/integrations/tools/searchapi",
    },
    "SerpAPI": {
        "pricing": "100 Free Searches/Month",
        "available_data": "Answer",
        "link": "/docs/integrations/tools/serpapi",
    },
}

CODE_INTERPRETER_TOOL_FEAT_TABLE = {
    "Bearly Code Interpreter": {
        "langauges": "Python",
        "sandbox_lifetime": "Resets on Execution",
        "upload": True,
        "return_results": "Text",
        "link": "/docs/integrations/tools/bearly",
    },
    "Riza Code Interpreter": {
        "langauges": "Python, JavaScript, PHP, Ruby",
        "sandbox_lifetime": "Resets on Execution",
        "upload": False,
        "return_results": "Text",
        "link": "/docs/integrations/tools/riza",
    },
    "E2B Data Analysis": {
        "langauges": "Python. In beta: JavaScript, R, Java",
        "sandbox_lifetime": "24 Hours",
        "upload": True,
        "return_results": "Text, Images, Videos",
        "link": "/docs/integrations/tools/e2b_data_analysis",
    },
    "Azure Container Apps dynamic sessions": {
        "langauges": "Python",
        "sandbox_lifetime": "1 Hour",
        "upload": True,
        "return_results": "Text, Images",
        "link": "/docs/integrations/tools/azure_dynamic_sessions",
    },
}

TOOLS_TEMPLATE = """\
---
sidebar_position: 0
sidebar_class_name: hidden
keywords: [compatibility]
custom_edit_url:
hide_table_of_contents: true
---

# Tools

## Search Tools

The following table shows tools that execute online searches in some shape or form:

{search_table}

## Code Interpreter Tools

The following table shows tools that can be used as code interpreters:

{code_interpreter_table}

"""


def get_search_tools_table() -> str:
    """Get the table of search tools."""
    header = ["tool", "pricing", "available_data"]
    title = ["Tool", "Free/Paid", "Return Data"]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for search_tool, feats in sorted(SEARCH_TOOL_FEAT_TABLE.items()):
        # Fields are in the order of the header
        row = [
            f"[{search_tool}]({feats['link']})",
        ]
        for h in header[1:]:
            row.append(feats.get(h))
        rows.append(row)
    return "\n".join(["|".join(row) for row in rows])


def get_code_interpreter_table() -> str:
    """Get the table of search tools."""
    header = [
        "tool",
        "langauges",
        "sandbox_lifetime",
        "upload",
        "return_results",
    ]
    title = [
        "Tool",
        "Supported Languages",
        "Sandbox Lifetime",
        "Supports File Uploads",
        "Return Types",
    ]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for search_tool, feats in sorted(CODE_INTERPRETER_TOOL_FEAT_TABLE.items()):
        # Fields are in the order of the header
        row = [
            f"[{search_tool}]({feats['link']})",
        ]
        for h in header[1:]:
            value = feats.get(h)
            if h == "upload":
                if value is True:
                    row.append("✅")
                else:
                    row.append("❌")
            else:
                row.append(value)
        rows.append(row)
    return "\n".join(["|".join(row) for row in rows])


if __name__ == "__main__":
    output_dir = Path(sys.argv[1])
    output_integrations_dir = output_dir / "integrations"
    output_integrations_dir_tools = output_integrations_dir / "tools"
    output_integrations_dir_tools.mkdir(parents=True, exist_ok=True)

    tools_page = TOOLS_TEMPLATE.format(
        search_table=get_search_tools_table(),
        code_interpreter_table=get_code_interpreter_table(),
    )
    with open(output_integrations_dir / "tools" / "index.mdx", "w") as f:
        f.write(tools_page)
