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
    "Data for SEO": {
        "pricing": "Not free, price depends on API used",
        "available_data": "URL, Snippet, Title, Type",
        "link": "/docs/integrations/tools/dataforseo",
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
    "Bash": {
        "langauges": "Bash",
        "sandbox_lifetime": "Resets on Execution",
        "upload": True,
        "return_results": "Bash execution results",
        "link": "/docs/integrations/tools/bash",
    },
    "Databrick Unity Cataong": {
        "langauges": "Python, SQL",
        "sandbox_lifetime": "Resets on Execution",
        "upload": False,
        "return_results": "Text",
        "link": "/docs/integrations/tools/databricks",
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
    "AWS Toolkit": {
        "link": "/docs/integrations/tools/aws",
        "pricing": "Free tier, with variable pricing after",
    },
    "ClickUp": {
        "link": "/docs/integrations/tools/clickup",
        "pricing": "Free tier, with variable pricing after",
    },
    "Cogniswitch": {
        "link": "/docs/integrations/tools/cogniswitch",
        "pricing": "Free trial, with variable pricing after",
    },
    "Google Drive": {
        "link": "/docs/integrations/tools/google_drive",
        "pricing": "Free",
    },
    "IFTTT Webhooks": {
        "link": "/docs/integrations/tools/ifttt",
        "pricing": "Free tier, with variable pricing after",
    },
    "Lemon AI": {
        "link": "/docs/integrations/tools/lemonai",
        "pricing": "Depends on service used",
    },
    "Power BI": {
        "link": "/docs/integrations/tools/power_bi",
        "pricing": "Free tier, with variable pricing after",
    },
    "Wolfram Alpha": {
        "link": "/docs/integrations/tools/wolfram_alpha",
        "pricing": "Free up to 2000 calls/month",
    },
    "Zapier": {
        "link": "/docs/integrations/tools/zapier",
        "pricing": "Free up to 100 tasks/month",
    },
    "Zenguard AI": {
        "link": "/docs/integrations/tools/zenguard",
        "pricing": "Free up to 1000 requests/day",
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
    "GraphQL Toolkit": {
        "link": "/docs/integrations/tools/graphql",
        "operations": "GraphQL queries",
    },
}

DOMAIN_SPECIFIC_SEARCH_TOOL_FEAT_TABLE = {
    "Amadeus": {"link": "/docs/integration/tools/amadeus", "domain": "Travel"},
    "Alpha Vantage": {
        "link": "/docs/integration/tools/alpha_vantage",
        "domain": "Finance",
    },
    "ArXiv": {"link": "/docs/integration/tools/arxiv", "domain": "Research"},
    "AskNews": {"link": "/docs/integration/tools/asknews", "domain": "News"},
    "Financial Datasets": {
        "link": "/docs/integration/tools/financial_datasets",
        "domain": "Finance",
    },
    "Google Finance": {
        "link": "/docs/integration/tools/google_finance",
        "domain": "Finance",
    },
    "Google Jobs": {"link": "/docs/integration/tools/google_jobs", "domain": "Jobs"},
    "Google Scholar": {
        "link": "/docs/integration/tools/google_scholar",
        "domain": "Research",
    },
    "Ionic Shopping": {
        "link": "/docs/integration/tools/ionic_shopping",
        "domain": "Shopping",
    },
    "NASA": {"link": "/docs/integration/tools/nasa", "domain": "Space"},
    "OpenWeatherMap": {
        "link": "/docs/integrations/tools/openweathermap",
        "domain": "Weather",
    },
    "Passio Nutrion": {
        "link": "/docs/integrations/tools/passio_nutrition_ai",
        "domain": "Nutrition",
    },
    "Polygon IO": {"link": "/docs/integrations/tools/polygon", "domain": "Finance"},
    "PubMed": {"link": "/docs/integrations/tools/pubmed", "domain": "Medical Research"},
    "Reddit Search": {
        "link": "/docs/integrations/tools/reddit_search",
        "domain": "Social Media",
    },
    "Semantic Scholar": {
        "link": "/docs/integrations/tools/semanticscholar",
        "domain": "Research",
    },
    "Stack Exchange": {
        "link": "/docs/integrations/tools/stackexchange",
        "domain": "StackOverflow",
    },
    "Steam Toolkit": {"link": "/docs/integrations/tools/steam", "domain": "Gaming"},
    "Wikidata": {
        "link": "/docs/integrations/tools/wikidata",
        "domain": "General Knowledge",
    },
    "Wikipedia": {
        "link": "/docs/integrations/tools/wikipedia",
        "domain": "General Knowledge",
    },
    "Yahoo Finance": {
        "link": "/docs/integrations/tools/yahoo_finance_news",
        "domain": "Finance",
    },
    "YouTube": {"link": "/docs/integrations/tools/youtube", "domain": "YouTube"},
    "Golden Query": {
        "link": "/docs/integrations/tools/golden_query",
        "domain": "General Knowledge",
    },
}

MULTIMODAL_TOOL_FEAT_TABLE = {
    "SceneXplain": {
        "link": "/docs/integration/tools/sceneXplain",
        "modalities": "Images",
    },
    "Nuclia Understanding": {
        "link": "/docs/integration/tools/nuclia",
        "modalities": "Images, Videos, Audio, Documents",
    },
    "NVIDIA Riva": {
        "link": "/docs/integration/tools/nvidia_riva",
        "modalities": "Audio",
    },
    "Azure AI Services": {
        "link": "/docs/integration/tools/azure_ai_services",
        "modalities": "Images, Videos, Audio, Documents",
    },
    "Azure Cognitive Services": {
        "link": "/docs/integration/tools/azure_cognitive_services",
        "modalities": "Images, Videos, Audio, Documents",
    },
    "Dall-E Image Generator": {
        "link": "/docs/integrations/tools/dalle_image_generator",
        "modalities": "Images",
    },
    "Eden AI": {
        "link": "/docs/integrations/tools/edenai_tools",
        "modalities": "Images, Audio, Invoices",
    },
    "Eleven Labs": {
        "link": "/docs/integrations/tools/eleven_labs_tts",
        "modalities": "Audio",
    },
    "Google Cloud Text-to-Speech": {
        "link": "/docs/integrations/tools/google_cloud_texttospeech",
        "modalities": "Audio",
    },
    "Google Imagen": {
        "link": "/docs/integrations/tools/google_imagen",
        "modalities": "Images",
    },
    "Google Lens": {
        "link": "/docs/integrations/tools/google_lens",
        "modalities": "Images",
    },
}

MISCELLANEOUS_TOOL_FEAT_TABLE = {
    "Dataherald": {
        "link": "/docs/integrations/tools/dataherald",
        "description": "Natural language to SQL API",
    },
    "File Management": {
        "link": "/docs/integrations/tools/filesystem",
        "description": "Manage your local file system",
    },
    "Gradio": {
        "link": "/docs/integrations/tools/gradio",
        "description": "Use Gradio ML apps in your agent",
    },
    "JSON Toolkit": {
        "link": "/docs/integrations/tools/json",
        "description": "Interact with large JSON blobs",
    },
    "OpenAPI": {
        "link": "/docs/integrations/tools/openapi",
        "description": "Consume arbitrary APIs conforming to the OpenAPI spec",
    },
    "Natural Language API": {
        "link": "/docs/integrations/tools/openapi_nla",
        "description": "Efficiently plan and combine calls across endpoints",
    },
    "Robocorp": {
        "link": "/docs/integrations/tools/robocorp",
        "description": "Integrate custom actions with your agents",
    },
    "Human as a tool": {
        "link": "/docs/integrations/tools/human_tools",
        "description": "Use human input as a tool",
    },
    "Memorize": {
        "link": "/docs/integrations/tools/memorize",
        "description": "Fine tune model to memorize data",
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

## Domain Specific Search

The following table shows tools that can be used to search for specific types of data:

{domain_specific_search_table}

## Multimodal

The following table shows tools that can be used for dealing with multimodal data:

{multimodal_table}

## Miscellaneous

The following table shows tools that don't fit into the other categories:

{miscellaneous_table}

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


def get_miscellaneous_table() -> str:
    """Get the table of miscellaneous tools."""
    header = ["tool", "description"]
    title = ["Tool/Toolkit", "Description"]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for miscellaneous_tool, feats in sorted(MISCELLANEOUS_TOOL_FEAT_TABLE.items()):
        # Fields are in the order of the header
        row = [
            f"[{miscellaneous_tool}]({feats['link']})",
        ]
        for h in header[1:]:
            row.append(feats.get(h))
        rows.append(row)
    return "\n".join(["|".join(row) for row in rows])


def get_domain_specific_search_table() -> str:
    """Get the table of domain specific tools."""
    header = ["tool", "domain"]
    title = ["Tool/Toolkit", "Domain"]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for domain_specific_tool, feats in sorted(
        DOMAIN_SPECIFIC_SEARCH_TOOL_FEAT_TABLE.items()
    ):
        # Fields are in the order of the header
        row = [
            f"[{domain_specific_tool}]({feats['link']})",
        ]
        for h in header[1:]:
            row.append(feats.get(h))
        rows.append(row)
    return "\n".join(["|".join(row) for row in rows])


def get_multimodal_table() -> str:
    """Get the table of multimodal tools."""
    header = ["tool", "modalities"]
    title = ["Tool/Toolkit", "Modalties"]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for multi_modal_tool, feats in sorted(MULTIMODAL_TOOL_FEAT_TABLE.items()):
        # Fields are in the order of the header
        row = [
            f"[{multi_modal_tool}]({feats['link']})",
        ]
        for h in header[1:]:
            row.append(feats.get(h))
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
    """Get the table of search tools."""
    header = [
        "tool",
        "langauges",
        "sandbox_lifetime",
        "upload",
        "return_results",
    ]
    title = [
        "Tool/Toolkit",
        "Supported Languages",
        "Sandbox Lifetime",
        "Supports File Uploads",
        "Return Types",
    ]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for code_interpreter_tool, feats in sorted(
        CODE_INTERPRETER_TOOL_FEAT_TABLE.items()
    ):
        # Fields are in the order of the header
        row = [
            f"[{code_interpreter_tool}]({feats['link']})",
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
        productivity_table=get_productivity_table(),
        webbrowsing_table=get_webbrowsing_table(),
        database_table=get_database_table(),
        domain_specific_search_table=get_domain_specific_search_table(),
        multimodal_table=get_multimodal_table(),
        miscellaneous_table=get_miscellaneous_table(),
    )
    with open(output_integrations_dir / "tools" / "index.mdx", "w") as f:
        f.write(tools_page)
