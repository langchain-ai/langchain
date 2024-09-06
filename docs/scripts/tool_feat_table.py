import sys
from pathlib import Path

SEARCH_TOOL_FEAT_TABLE = {
    "Tavily Search": {
        "pricing": "1000 free searches/month",
        "available_data": "URL, Content, Title, Images, Answer",
        "link": "/docs/integrations/tools/tavily_search",
    },
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
    "Jina Search": {
        "pricing": "1M Response Tokens Free",
        "available_data": "URL, Snippet, Title, Page Content",
        "link": "/docs/integrations/tools/jina_search/",
    },
}

CODE_INTERPRETER_TOOL_FEAT_TABLE = {
    "Bearly Code Interpreter": {
        "langauges": "Python",
        "sandbox_lifetime": "Resets on Execution",
        "upload": True,
        "return_results": "Text",
        "link": "/docs/integrations/tools/bearly",
        "self_hosting": False,
    },
    "Riza Code Interpreter": {
        "langauges": "Python, JavaScript, PHP, Ruby",
        "sandbox_lifetime": "Resets on Execution",
        "upload": False,
        "return_results": "Text",
        "link": "/docs/integrations/tools/riza",
        "self_hosting": True,
    },
    "E2B Data Analysis": {
        "langauges": "Python. In beta: JavaScript, R, Java",
        "sandbox_lifetime": "24 Hours",
        "upload": True,
        "return_results": "Text, Images, Videos",
        "link": "/docs/integrations/tools/e2b_data_analysis",
        "self_hosting": True,
    },
    "Azure Container Apps dynamic sessions": {
        "langauges": "Python",
        "sandbox_lifetime": "1 Hour",
        "upload": True,
        "return_results": "Text, Images",
        "link": "/docs/integrations/tools/azure_dynamic_sessions",
        "self_hosting": False,
    },
}

PRODUCTIVITY_TOOL_FEAT_TABLE = {
    "Gmail Toolkit": {
        "link": "/docs/integrations/tools/gmail",
        "pricing": "Free, with limit of 250 quota units per user per second",
    },
    "Github Toolkit": {
        "link": "/docs/integrations/tools/github",
        "pricing": "Free",
    },
    "Gitlab Toolkit": {
        "link": "/docs/integrations/tools/gitlab",
        "pricing": "Free for personal project",
    },
    "Slack Toolkit": {
        "link": "/docs/integrations/tools/slack",
        "pricing": "Free",
    },
    "Jira Toolkit": {
        "link": "/docs/integrations/tools/jira",
        "pricing": "Free, with [rate limits](https://developer.atlassian.com/cloud/jira/platform/rate-limiting/)",
    },
    "Office365 Toolkit": {
        "link": "/docs/integrations/tools/office365",
        "pricing": "Free with Office365, includes [rate limits](https://learn.microsoft.com/en-us/graph/throttling-limits)",
    },
    "Twilio Tool": {
        "link": "/docs/integrations/tools/twilio",
        "pricing": "Free trial, with [pay-as-you-go pricing](https://www.twilio.com/en-us/pricing) after",
    },
    "Infobip Tool": {
        "link": "/docs/integrations/tools/infobip",
        "pricing": "Free trial, with variable pricing after",
    },
}

WEBBROWSING_TOOL_FEAT_TABLE = {
    "Requests Toolkit": {
        "link": "/docs/integrations/tools/requests",
        "interactions": False,
        "pricing": "Free",
    },
    "PlayWright Browser Toolkit": {
        "link": "/docs/integrations/tools/playwright",
        "interactions": True,
        "pricing": "Free",
    },
    "MultiOn Toolkit": {
        "link": "/docs/integrations/tools/multion",
        "interactions": True,
        "pricing": "40 free requests/day",
    },
}

DATABASE_TOOL_FEAT_TABLE = {
    "SQLDatabase Toolkit": {
        "link": "/docs/integrations/tools/sql_database",
        "operations": "Any SQL operation",
    },
    "Spark SQL Toolkit": {
        "link": "/docs/integrations/tools/spark_sql",
        "operations": "Any SQL operation",
    },
    "Cassandra Database Toolkit": {
        "link": "/docs/integrations/tools/cassandra_database",
        "operations": "SELECT and schema introspection",
    },
}

TOOLS_TEMPLATE = """\
---
sidebar_position: 0
sidebar_class_name: hidden
keywords: [compatibility]
custom_edit_url:
---

# Tools

[Tools](/docs/concepts/#tools) are utilities designed to be called by a model: their inputs are designed to be generated by models, and their outputs are designed to be passed back to models.

A [toolkit](/docs/concepts#toolkits) is a collection of tools meant to be used together.

:::info

If you'd like to write your own tool, see [this how-to](/docs/how_to/custom_tools/).
If you'd like to contribute an integration, see [Contributing integrations](/docs/contributing/integrations/).

:::

## Search

The following table shows tools that execute online searches in some shape or form:

{search_table}

## Code Interpreter

The following table shows tools that can be used as code interpreters:

{code_interpreter_table}

## Productivity

The following table shows tools that can be used to automate tasks in productivity tools:

{productivity_table}

## Web Browsing

The following table shows tools that can be used to automate tasks in web browsers:

{webbrowsing_table}

## Database

The following table shows tools that can be used to automate tasks in databases:

{database_table}

## All tools

import {{ IndexTable }} from "@theme/FeatureTables";

<IndexTable />

"""  # noqa: E501


def get_productivity_table() -> str:
    """Get the table of productivity tools."""
    header = [
        "tool",
        "pricing",
    ]
    title = [
        "Tool/Toolkit",
        "Pricing",
    ]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for productivity_tool, feats in sorted(PRODUCTIVITY_TOOL_FEAT_TABLE.items()):
        # Fields are in the order of the header
        row = [
            f"[{productivity_tool}]({feats['link']})",
        ]
        for h in header[1:]:
            row.append(feats.get(h))
        rows.append(row)
    return "\n".join(["|".join(row) for row in rows])


def get_webbrowsing_table() -> str:
    """Get the table of web browsing tools."""
    header = ["tool", "pricing", "interactions"]
    title = ["Tool/Toolkit", "Pricing", "Supports Interacting with the Browser"]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for web_browsing_tool, feats in sorted(WEBBROWSING_TOOL_FEAT_TABLE.items()):
        # Fields are in the order of the header
        row = [
            f"[{web_browsing_tool}]({feats['link']})",
        ]
        for h in header[1:]:
            value = feats.get(h)
            if h == "interactions":
                if value is True:
                    row.append("✅")
                else:
                    row.append("❌")
            else:
                row.append(value)
        rows.append(row)
    return "\n".join(["|".join(row) for row in rows])


def get_database_table() -> str:
    """Get the table of database tools."""
    header = ["tool", "operations"]
    title = ["Tool/Toolkit", "Allowed Operations"]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for database_tool, feats in sorted(DATABASE_TOOL_FEAT_TABLE.items()):
        # Fields are in the order of the header
        row = [
            f"[{database_tool}]({feats['link']})",
        ]
        for h in header[1:]:
            row.append(feats.get(h))
        rows.append(row)
    return "\n".join(["|".join(row) for row in rows])


def get_search_tools_table() -> str:
    """Get the table of search tools."""
    header = ["tool", "pricing", "available_data"]
    title = ["Tool/Toolkit", "Free/Paid", "Return Data"]
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
    """Get the table of code interpreter tools."""
    header = [
        "tool",
        "langauges",
        "sandbox_lifetime",
        "upload",
        "return_results",
        "self_hosting",
    ]
    title = [
        "Tool/Toolkit",
        "Supported Languages",
        "Sandbox Lifetime",
        "Supports File Uploads",
        "Return Types",
        "Supports Self-Hosting",
    ]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for search_tool, feats in sorted(CODE_INTERPRETER_TOOL_FEAT_TABLE.items()):
        # Fields are in the order of the header
        row = [
            f"[{search_tool}]({feats['link']})",
        ]
        for h in header[1:]:
            value = feats.get(h)
            if h == "upload" or h == "self_hosting":
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
        productivity_table=get_productivity_table(),
        webbrowsing_table=get_webbrowsing_table(),
        database_table=get_database_table(),
    )
    with open(output_integrations_dir / "tools" / "index.mdx", "w") as f:
        f.write(tools_page)
