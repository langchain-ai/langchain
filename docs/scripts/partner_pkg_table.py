import glob
import sys
from pathlib import Path

import requests
import yaml

#################
# CONFIGURATION #
#################

# packages to ignore / exclude from the table
IGNORE_PACKGAGES = {
    # top-level packages
    "langchain-core",
    "langchain-text-splitters",
    "langchain",
    "langchain-community",
    "langchain-experimental",
    "langchain-cli",
    "langchain-tests",
    # integration packages that don't have a provider index
    # do NOT add to these. These were merged before having a
    # provider index was required
    # can remove these once they have a provider index
    "langchain-yt-dlp",
}

#####################
# END CONFIGURATION #
#####################

DOCS_DIR = Path(__file__).parents[1]
PACKAGE_YML = Path(__file__).parents[2] / "libs" / "packages.yml"

# for now, only include packages that are in the langchain-ai org
# because we don't have a policy for inclusion in this table yet,
# and including all packages will make the list too long


def _get_type(package: dict) -> str:
    if package["name"] in IGNORE_PACKGAGES:
        return "ignore"
    if package["repo"] == "langchain-ai/langchain":
        return "B"
    if package["repo"].startswith("langchain-ai/"):
        return "C"
    return "D"


def _enrich_package(p: dict) -> dict | None:
    p["name_short"] = (
        p["name"][10:] if p["name"].startswith("langchain-") else p["name"]
    )
    p["name_title"] = p.get("name_title") or p["name_short"].title().replace(
        "-", " "
    ).replace("db", "DB").replace("Db", "DB").replace("ai", "AI").replace("Ai", "AI")
    p["type"] = _get_type(p)

    if p["type"] == "ignore":
        return None

    p["js_exists"] = bool(p.get("js"))
    custom_provider_page = p.get("provider_page")
    default_provider_page = f"/docs/integrations/providers/{p['name_short']}/"
    default_provider_page_exists = bool(
        glob.glob(str(DOCS_DIR / f"docs/integrations/providers/{p['name_short']}.*"))
    )
    p["provider_page"] = custom_provider_page or (
        default_provider_page if default_provider_page_exists else None
    )
    if p["provider_page"] is None:
        msg = (
            f"Provider page not found for {p['name_short']}. "
            f"Please add one at docs/integrations/providers/{p['name_short']}.{{mdx,ipynb}}"
        )
        raise ValueError(msg)

    return p


with open(PACKAGE_YML) as f:
    data = yaml.safe_load(f)

packages_n = [_enrich_package(p) for p in data["packages"]]
packages = [p for p in packages_n if p is not None]

# sort by downloads
packages_sorted = sorted(packages, key=lambda p: p["downloads"], reverse=True)


def package_row(p: dict) -> str:
    js = "✅" if p["js_exists"] else "❌"
    link = p["provider_page"]
    title = p["name_title"]
    provider = f"[{title}]({link})" if link else title
    return f"| {provider} | [{p['name']}](https://python.langchain.com/api_reference/{p['name_short'].replace('-', '_')}/) | ![PyPI - Downloads](https://img.shields.io/pypi/dm/{p['name']}?style=flat-square&label=%20&color=blue) | ![PyPI - Version](https://img.shields.io/pypi/v/{p['name']}?style=flat-square&label=%20&color=orange) | {js} |"


def table() -> str:
    header = """| Provider | Package | Downloads | Latest | [JS](https://js.langchain.com/docs/integrations/providers/) |
| :--- | :---: | :---: | :---: | :---: |
"""
    return header + "\n".join(package_row(p) for p in packages_sorted)


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

Click [here](/docs/integrations/providers/all) to see all providers. Or search for a
provider using the Search field in the top-right corner of the screen.

"""


if __name__ == "__main__":
    output_dir = Path(sys.argv[1]) / "integrations" / "providers"
    with open(output_dir / "index.mdx", "w") as f:
        f.write(doc())
