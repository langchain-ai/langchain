import sys
from pathlib import Path

TOOL_FEAT_TABLE = {
    "Exa Search": {
        "pricing":"Free",
        "source":"Internet",
        "available_data":"IDK"
    },
    "Bing Search": {
        "pricing":"Free",
        "source":"Internet",
        "available_data":"IDK"
    },
    "DuckDuckgoSearch": {
        "pricing":"Free",
        "source":"Internet",
        "available_data":"IDK"
    },
    "Exa Search": {
        "pricing":"Free",
        "source":"Internet",
        "available_data":"IDK"
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

{table}

## Code Interpreter Tools

The following table shows tools that can be used as code interpreters:

"""


def get_tools_table() -> str:
    """Get the table of chat models."""
    header = [
        "tool",
        "pricing",
        "source",
        "available_data"
    ]
    title = [
        "Tool",
        "Free/Paid",
        "Source",
        "Return Data"
    ]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for llm, feats in sorted(TOOL_FEAT_TABLE.items()):
        # Fields are in the order of the header
        row = [
            f"[{llm}]({feats['link']})",
        ]
        for h in header[1:]:
            value = feats.get(h)
            if h == "package":
                row.append(value or "langchain-community")
            else:
                if value == "partial":
                    row.append("🟡")
                elif value is True:
                    row.append("✅")
                else:
                    row.append("❌")
        rows.append(row)
    return "\n".join(["|".join(row) for row in rows])


if __name__ == "__main__":
    output_dir = Path(sys.argv[1])
    output_integrations_dir = output_dir / "integrations"
    output_integrations_dir_tools = output_integrations_dir / "tools"
    output_integrations_dir_tools.mkdir(parents=True, exist_ok=True)

    tools_page = TOOLS_TEMPLATE.format(table=get_tools_table())
    with open(output_integrations_dir / "tools" / "index.mdx", "w") as f:
        f.write(tools_page)
