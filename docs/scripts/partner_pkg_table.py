import glob
import sys
from pathlib import Path

import yaml

DOCS_DIR = Path(__file__).parents[1]
PACKAGE_YML = Path(__file__).parents[2] / "libs" / "packages.yml"
IGNORE_PACKGAGES = {"langchain-experimental"}

# for now, only include packages that are in the langchain-ai org
# because we don't have a policy for inclusion in this table yet,
# and including all packages will make the list too long
with open(PACKAGE_YML) as f:
    data = yaml.safe_load(f)
    EXTERNAL_PACKAGES = set(
        p["name"][10:]
        for p in data["packages"]
        if p["repo"].startswith("langchain-ai/")
        and p["repo"] != "langchain-ai/langchain"
        and p["name"] not in IGNORE_PACKGAGES
    )
    IN_REPO_PACKAGES = set(
        p["name"][10:]
        for p in data["packages"]
        if p["repo"] == "langchain-ai/langchain"
        and p["path"].startswith("libs/partners")
        and p["name"] not in IGNORE_PACKGAGES
    )

JS_PACKAGES = {
    "google-gauth",
    "openai",
    "anthropic",
    "google-genai",
    "pinecone",
    "aws",
    "google-vertexai",
    "qdrant",
    "azure-dynamic-sessions",
    "google-vertexai-web",
    "redis",
    "azure-openai",
    "google-webauth",
    "baidu-qianfan",
    "groq",
    "standard-tests",
    "cloudflare",
    "mistralai",
    "textsplitters",
    "cohere",
    "mixedbread-ai",
    "weaviate",
    "mongodb",
    "yandex",
    "exa",
    "nomic",
    "google-common",
    "ollama",
    "ibm",
}

ALL_PACKAGES = IN_REPO_PACKAGES.union(EXTERNAL_PACKAGES)

CUSTOM_NAME = {
    "google-genai": "Google Generative AI",
    "aws": "AWS",
    "ibm": "IBM",
}
CUSTOM_PROVIDER_PAGES = {
    "azure-dynamic-sessions": "/docs/integrations/providers/microsoft/",
    "prompty": "/docs/integrations/providers/microsoft/",
    "sqlserver": "/docs/integrations/providers/microsoft/",
    "google-community": "/docs/integrations/providers/google/",
    "google-genai": "/docs/integrations/providers/google/",
    "google-vertexai": "/docs/integrations/providers/google/",
    "nvidia-ai-endpoints": "/docs/integrations/providers/nvidia/",
    "exa": "/docs/integrations/providers/exa_search/",
    "mongodb": "/docs/integrations/providers/mongodb_atlas/",
    "sema4": "/docs/integrations/providers/robocorp/",
    "postgres": "/docs/integrations/providers/pgvector/",
}
PROVIDER_PAGES = {
    name: f"/docs/integrations/providers/{name}/"
    for name in ALL_PACKAGES
    if glob.glob(str(DOCS_DIR / f"docs/integrations/providers/{name}.*"))
}
PROVIDER_PAGES = {
    **PROVIDER_PAGES,
    **CUSTOM_PROVIDER_PAGES,
}


def package_row(name: str) -> str:
    js = "✅" if name in JS_PACKAGES else "❌"
    link = PROVIDER_PAGES.get(name)
    title = CUSTOM_NAME.get(name) or name.title().replace("-", " ").replace(
        "db", "DB"
    ).replace("Db", "DB").replace("ai", "AI").replace("Ai", "AI")
    provider = f"[{title}]({link})" if link else title
    return f"| {provider} | [langchain-{name}](https://python.langchain.com/api_reference/{name.replace('-', '_')}/) | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-{name}?style=flat-square&label=%20&color=blue) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-{name}?style=flat-square&label=%20&color=orange) | {js} |"


def table() -> str:
    header = """| Provider | Package | Downloads | Latest | [JS](https://js.langchain.com/docs/integrations/providers/) |
| :--- | :---: | :---: | :---: | :---: |
"""
    return header + "\n".join(package_row(name) for name in sorted(ALL_PACKAGES))


def doc() -> str:
    return f"""\
---
sidebar_position: 0
sidebar_class_name: hidden
---

# Providers

:::info

If you'd like to write your own integration, see [Extending LangChain](/docs/how_to/#custom).
If you'd like to contribute an integration, see [Contributing integrations](/docs/contributing/how_to/integrations/).

:::

LangChain integrates with many providers.

## Integration Packages

These providers have standalone `langchain-{{provider}}` packages for improved versioning, dependency management and testing.

{table()}

## All Providers

Click [here](/docs/integrations/providers/all) to see all providers.

"""


if __name__ == "__main__":
    output_dir = Path(sys.argv[1]) / "integrations" / "providers"
    with open(output_dir / "index.mdx", "w") as f:
        f.write(doc())
