import glob
import sys
from pathlib import Path

PARTNER_DIR = Path(__file__).parents[2] / "libs" / "partners"
DOCS_DIR = Path(__file__).parents[1]

PLATFORMS = {
    path.split("/")[-1][:-4]
    for path in glob.glob(
        str(DOCS_DIR) + "/docs/integrations/platforms/*.mdx", recursive=True
    )
}
EXTERNAL_PACKAGES = {
    "astradb",
    "aws",
    "cohere",
    "elasticsearch",
    "google-community",
    "google-genai",
    "google-vertexai",
    "nvidia-ai-endpoints",
    "postgres",
    "redis",
    "weaviate",
    "upstage",
}

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
}


IN_REPO_PACKAGES = {
    path.split("/")[-2]
    for path in glob.glob(str(PARTNER_DIR) + "/**/pyproject.toml", recursive=True)
}
ALL_PACKAGES = IN_REPO_PACKAGES.union(EXTERNAL_PACKAGES)

CUSTOM_NAME = {
    "google-genai": "Google Generative AI",
    "aws": "AWS",
    "airbyte": "Airbyte",
}
CUSTOM_LINKS = {
    "azure-dynamic-sessions": "/docs/integrations/platforms/microsoft/",
    "google-community": "/docs/integrations/platforms/google/",
    "google-genai": "/docs/integrations/platforms/google/",
    "google-vertexai": "/docs/integrations/platforms/google/",
    "nvidia-ai-endpoints": "/docs/integrations/providers/nvidia/",
}
CUSTOM_LINKS.update(
    {name: f"/docs/integrations/platforms/{name}/" for name in PLATFORMS}
)


def package_row(name: str) -> str:
    js = "✅" if name in JS_PACKAGES else "❌"
    link = CUSTOM_LINKS.get(name) or f"/docs/integrations/providers/{name}/"
    title = CUSTOM_NAME.get(name) or name.title().replace("-", " ").replace(
        "db", "DB"
    ).replace("Db", "DB").replace("ai", "AI").replace("Ai", "AI")
    return f"| [{title}]({link}) | [langchain-{name}](https://api.python.langchain.com/en/latest/{name.replace('-', '_')}_api_reference.html) | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-{name}?style=flat-square&label=%20&color=blue) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-{name}?style=flat-square&label=%20&color=orange) | {js} |"


def table() -> str:
    header = """| Provider | API ref | Package downloads | Package latest | JS support |
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
If you'd like to contribute an integration, see [Contributing integrations](/docs/contributing/integrations/).

:::

LangChain integrates with many providers.

## Integration Packages

These providers have standalone `langchain-{{provider}}` packages for improved versioning, dependency management and testing.

{table()}

## All Providers

Click [here](/docs/integrations/providers/) to see all providers.

"""


if __name__ == "__main__":
    output_dir = Path(sys.argv[1]) / "integrations" / "platforms"
    with open(output_dir / "index.mdx", "w") as f:
        f.write(doc())
