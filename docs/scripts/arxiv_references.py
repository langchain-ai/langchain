"""Parse arXiv references from the documentation.
Generate a page with a table of the arXiv references with links to the documentation pages.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from pydantic.v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(os.path.abspath(__file__)).parents[2]
DOCS_DIR = _ROOT_DIR / "docs" / "docs"
CODE_DIR = _ROOT_DIR / "libs"
TEMPLATES_DIR = _ROOT_DIR / "templates"
COOKBOOKS_DIR = _ROOT_DIR / "cookbook"
ARXIV_ID_PATTERN = r"https://arxiv\.org/(abs|pdf)/(\d+\.\d+)"
LANGCHAIN_PYTHON_URL = "python.langchain.com"


@dataclass
class ArxivPaper:
    """ArXiv paper information."""

    arxiv_id: str
    referencing_doc2url: dict[str, str]
    referencing_api_ref2url: dict[str, str]
    referencing_template2url: dict[str, str]
    referencing_cookbook2url: dict[str, str]
    title: str
    authors: list[str]
    abstract: str
    url: str
    published_date: str


def search_documentation_for_arxiv_references(docs_dir: Path) -> dict[str, set[str]]:
    """Search the documentation for arXiv references.

    Search for the arXiv references in the documentation pages.
    Note: It finds only the first arXiv reference in a line.

    Args:
        docs_dir: Path to the documentation root folder.
    Returns:
        dict: Dictionary with arxiv_id as key and set of file names as value.
    """
    arxiv_url_pattern = re.compile(ARXIV_ID_PATTERN)
    exclude_strings = {"file_path", "metadata", "link", "loader", "PyPDFLoader"}

    files = (
        p.resolve()
        for p in Path(docs_dir).glob("**/*")
        if p.suffix in {".ipynb", ".mdx", ".md"}
    )
    arxiv_id2file_names: dict[str, set[str]] = {}
    for file in files:
        if "-checkpoint.ipynb" in file.name:
            continue
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if any(exclude_string in line for exclude_string in exclude_strings):
                    continue
                matches = arxiv_url_pattern.search(line)
                if matches:
                    arxiv_id = matches.group(2)
                    file_name = _get_doc_path(file.parts, file.suffix)
                    if arxiv_id not in arxiv_id2file_names:
                        arxiv_id2file_names[arxiv_id] = {file_name}
                    else:
                        arxiv_id2file_names[arxiv_id].add(file_name)
    return arxiv_id2file_names


def search_code_for_arxiv_references(code_dir: Path) -> dict[str, set[str]]:
    """Search the code for arXiv references.

    Search for the arXiv references in the code.
    Note: It finds only the first arXiv reference in a line.

    Args:
        code_dir: Path to the code root folder.
    Returns:
        dict: Dictionary with arxiv_id as key and set of module names as value.
          module names encoded as:
            <module_name>:module
            <module_name>:class$<ClassName>
            <module_name>:function$<function_name>
    """
    arxiv_url_pattern = re.compile(ARXIV_ID_PATTERN)
    # exclude_strings = {"file_path", "metadata", "link", "loader"}
    class_pattern = re.compile(r"\s*class\s+(\w+).*:")
    function_pattern = re.compile(r"\s*def\s+(\w+)")

    # loop all the files (ipynb, mdx, md) in the docs folder
    files = (
        p.resolve()
        for p in Path(code_dir).glob("**/*")
        if p.suffix in {".py"} and "tests" not in p.parts and "scripts" not in p.parts
        # ".md" files are excluded
    )
    arxiv_id2module_name_and_members: dict[str, set[str]] = {}
    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                module_name = _get_module_name(file.parts)
                class_or_function_started = "module"
                for line in f.readlines():
                    # class line:
                    matches = class_pattern.search(line)
                    if matches:
                        class_name = matches.group(1)
                        class_or_function_started = f"class${class_name}"

                    # function line:
                    #  not inside a class!
                    if "class" not in class_or_function_started:
                        matches = function_pattern.search(line)
                        if matches:
                            func_name = matches.group(1)
                            class_or_function_started = f"function${func_name}"

                    # arxiv line:
                    matches = arxiv_url_pattern.search(line)
                    if matches:
                        arxiv_id = matches.group(2)
                        module_name_and_member = (
                            f"{module_name}:{class_or_function_started}"
                        )
                        if arxiv_id not in arxiv_id2module_name_and_members:
                            arxiv_id2module_name_and_members[arxiv_id] = {
                                module_name_and_member
                            }
                        else:
                            arxiv_id2module_name_and_members[arxiv_id].add(
                                module_name_and_member
                            )
        except UnicodeDecodeError:
            # Skip files like this 'tests/integration_tests/examples/non-utf8-encoding.py'
            logger.warning(f"Could not read the file {file}.")

    # handle border cases:
    # 1. {'langchain_experimental.pal_chain.base:class$PALChain', 'langchain_experimental.pal_chain.base:module' - remove}
    for arxiv_id, module_name_and_members in arxiv_id2module_name_and_members.items():
        module_name_and_member_deduplicated = set()
        non_module_members = set()
        for module_name_and_member in module_name_and_members:
            if not module_name_and_member.endswith(":module"):
                module_name_and_member_deduplicated.add(module_name_and_member)
                non_module_members.add(module_name_and_member.split(":")[0])
        for module_name_and_member in module_name_and_members:
            if module_name_and_member.endswith(":module"):
                if module_name_and_member.split(":")[0] in non_module_members:
                    continue
                module_name_and_member_deduplicated.add(module_name_and_member)
        arxiv_id2module_name_and_members[arxiv_id] = module_name_and_member_deduplicated

    # 2. {'langchain.evaluation.scoring.prompt:module', 'langchain.evaluation.comparison.prompt:module'}
    #    only modules with 2-part namespaces are parsed into API Reference now! TODO fix this behavior
    #    leave only the modules with 2-part namespaces
    arxiv_id2module_name_and_members_reduced = {}
    for arxiv_id, module_name_and_members in arxiv_id2module_name_and_members.items():
        module_name_and_member_reduced = set()
        removed_modules = set()
        for module_name_and_member in module_name_and_members:
            if module_name_and_member.endswith(":module"):
                if module_name_and_member.split(":")[0].count(".") <= 1:
                    module_name_and_member_reduced.add(module_name_and_member)
                else:
                    removed_modules.add(module_name_and_member)
            else:
                module_name_and_member_reduced.add(module_name_and_member)
        if module_name_and_member_reduced:
            arxiv_id2module_name_and_members_reduced[
                arxiv_id
            ] = module_name_and_member_reduced
        if removed_modules:
            logger.warning(
                f"{arxiv_id}: Removed the following modules with 2+ -part namespaces: {removed_modules}."
            )
    return arxiv_id2module_name_and_members_reduced


def search_templates_for_arxiv_references(templates_dir: Path) -> dict[str, set[str]]:
    arxiv_url_pattern = re.compile(ARXIV_ID_PATTERN)

    # loop all the Readme.md files since they are parsed into LangChain documentation
    # exclude the Readme.md in the root folder
    files = (
        p.resolve()
        for p in Path(templates_dir).glob("**/*")
        if p.name.lower() in {"readme.md"} and p.parent.name != "templates"
    )
    arxiv_id2template_names: dict[str, set[str]] = {}
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                matches = arxiv_url_pattern.search(line)
                if matches:
                    arxiv_id = matches.group(2)
                    template_name = file.parent.name
                    if arxiv_id not in arxiv_id2template_names:
                        arxiv_id2template_names[arxiv_id] = {template_name}
                    else:
                        arxiv_id2template_names[arxiv_id].add(template_name)
    return arxiv_id2template_names


def search_cookbooks_for_arxiv_references(cookbooks_dir: Path) -> dict[str, set[str]]:
    arxiv_url_pattern = re.compile(ARXIV_ID_PATTERN)
    files = (p.resolve() for p in Path(cookbooks_dir).glob("**/*.ipynb"))
    arxiv_id2cookbook_names: dict[str, set[str]] = {}
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                matches = arxiv_url_pattern.search(line)
                if matches:
                    arxiv_id = matches.group(2)
                    cookbook_name = file.stem
                    if arxiv_id not in arxiv_id2cookbook_names:
                        arxiv_id2cookbook_names[arxiv_id] = {cookbook_name}
                    else:
                        arxiv_id2cookbook_names[arxiv_id].add(cookbook_name)
    return arxiv_id2cookbook_names


def convert_module_name_and_members_to_urls(
    arxiv_id2module_name_and_members: dict[str, set[str]],
) -> dict[str, set[str]]:
    arxiv_id2urls = {}
    for arxiv_id, module_name_and_members in arxiv_id2module_name_and_members.items():
        urls = set()
        for module_name_and_member in module_name_and_members:
            module_name, type_and_member = module_name_and_member.split(":")
            if "$" in type_and_member:
                type_, member = type_and_member.split("$")
            else:
                type_ = type_and_member
                member = ""
            _namespace_parts = module_name.split(".")
            if type_ == "module":
                first_namespace_part = _namespace_parts[0]
                if first_namespace_part.startswith("langchain_"):
                    first_namespace_part = first_namespace_part.replace(
                        "langchain_", ""
                    )
                url = f"{first_namespace_part}_api_reference.html#module-{module_name}"
            elif type_ in ["class", "function"]:
                second_namespace_part = _namespace_parts[1]
                url = f"{second_namespace_part}/{module_name}.{member}.html#{module_name}.{member}"
            else:
                raise ValueError(
                    f"Unknown type: {type_} in the {module_name_and_member}."
                )
            urls.add(url)
        arxiv_id2urls[arxiv_id] = urls
    return arxiv_id2urls


def _get_doc_path(file_parts: tuple[str, ...], file_extension) -> str:
    """Get the relative path to the documentation page
    from the absolute path of the file.
    Remove file_extension
    """
    res = []
    for el in file_parts[::-1]:
        res.append(el)
        if el == "docs":
            break
    ret = "/".join(reversed(res))
    return ret[: -len(file_extension)] if ret.endswith(file_extension) else ret


def _get_code_path(file_parts: tuple[str, ...]) -> str:
    """Get the relative path to the documentation page
    from the absolute path of the file.
    """
    res = []
    for el in file_parts[::-1]:
        res.append(el)
        if el == "libs":
            break
    return "/".join(reversed(res))


def _get_module_name(file_parts: tuple[str, ...]) -> str:
    """Get the module name from the absolute path of the file."""
    ns_parts = []
    for el in file_parts[::-1]:
        if str(el) == "__init__.py":
            continue
        ns_parts.insert(0, str(el).replace(".py", ""))
        if el.startswith("langchain"):
            break
    return ".".join(ns_parts)


def _is_url_ok(url: str) -> bool:
    """Check if the url page is open without error."""
    import requests

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as ex:
        logger.warning(f"Could not open the {url}.")
        return False
    return True


class ArxivAPIWrapper(BaseModel):
    arxiv_search: Any  #: :meta private:
    arxiv_exceptions: Any  # :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            import arxiv

            values["arxiv_search"] = arxiv.Search
            values["arxiv_exceptions"] = (
                arxiv.ArxivError,
                arxiv.UnexpectedEmptyPageError,
                arxiv.HTTPError,
            )
        except ImportError:
            raise ImportError(
                "Could not import arxiv python package. "
                "Please install it with `pip install arxiv`."
            )
        return values

    def get_papers(
        self, arxiv_id2type2key2urls: dict[str, dict[str, dict[str, str]]]
    ) -> list[ArxivPaper]:
        """
        Performs an arxiv search and returns information about the papers found.

        If an error occurs or no documents found, error text
        is returned instead.
        Args:
            arxiv_id2type2key2urls: Dictionary with arxiv_id as key and dictionary
             with dicts of doc file names/API objects/templates to urls.

        Returns:
            List of ArxivPaper objects.
        """

        def cut_authors(authors: list) -> list[str]:
            if len(authors) > 3:
                return [str(a) for a in authors[:3]] + [" et al."]
            else:
                return [str(a) for a in authors]

        if not arxiv_id2type2key2urls:
            return []
        try:
            arxiv_ids = list(arxiv_id2type2key2urls.keys())
            results = self.arxiv_search(
                id_list=arxiv_ids,
                max_results=len(arxiv_ids),
            ).results()
        except self.arxiv_exceptions as ex:
            raise ex
        papers = [
            ArxivPaper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                authors=cut_authors(result.authors),
                abstract=result.summary,
                url=result.entry_id,
                published_date=str(result.published.date()),
                referencing_doc2url=(
                    type2key2urls["docs"] if "docs" in type2key2urls else {}
                ),
                referencing_api_ref2url=(
                    type2key2urls["apis"] if "apis" in type2key2urls else {}
                ),
                referencing_template2url=(
                    type2key2urls["templates"] if "templates" in type2key2urls else {}
                ),
                referencing_cookbook2url=(
                    type2key2urls["cookbooks"] if "cookbooks" in type2key2urls else {}
                ),
            )
            for result, type2key2urls in zip(results, arxiv_id2type2key2urls.values())
        ]
        return papers


def _format_doc_url(doc_path: str) -> str:
    return f"https://{LANGCHAIN_PYTHON_URL}/{doc_path}"


def _format_api_ref_url(doc_path: str, compact: bool = False) -> str:
    # agents/langchain_core.agents.AgentAction.html#langchain_core.agents.AgentAction
    return f"https://api.{LANGCHAIN_PYTHON_URL}/en/latest/{doc_path.split('langchain.com/')[-1]}"


def _format_template_url(template_name: str) -> str:
    return (
        f"https://github.com/langchain-ai/langchain/blob/v0.2/templates/{template_name}"
    )


def _format_cookbook_url(cookbook_name: str) -> str:
    return f"https://github.com/langchain-ai/langchain/blob/master/cookbook/{cookbook_name}.ipynb"


def _compact_module_full_name(doc_path: str) -> str:
    # agents/langchain_core.agents.AgentAction.html#langchain_core.agents.AgentAction
    module = doc_path.split("#")[1].replace("module-", "")
    if module.count(".") > 2:
        # langchain_community.llms.oci_data_science_model_deployment_endpoint.OCIModelDeploymentTGI
        # -> langchain_community...OCIModelDeploymentTGI
        module_parts = module.split(".")
        module = f"{module_parts[0]}...{module_parts[-1]}"
    return module


def compound_urls(
    arxiv_id2file_names: dict[str, set[str]],
    arxiv_id2code_urls: dict[str, set[str]],
    arxiv_id2templates: dict[str, set[str]],
    arxiv_id2cookbooks: dict[str, set[str]],
) -> dict[str, dict[str, set[str]]]:
    # format urls and verify that the urls are correct
    arxiv_id2file_names_new = {}
    for arxiv_id, file_names in arxiv_id2file_names.items():
        key2urls = {
            key: _format_doc_url(key)
            for key in file_names
            if _is_url_ok(_format_doc_url(key))
        }
        if key2urls:
            arxiv_id2file_names_new[arxiv_id] = key2urls

    arxiv_id2code_urls_new = {}
    for arxiv_id, code_urls in arxiv_id2code_urls.items():
        key2urls = {
            key: _format_api_ref_url(key)
            for key in code_urls
            if _is_url_ok(_format_api_ref_url(key))
        }
        if key2urls:
            arxiv_id2code_urls_new[arxiv_id] = key2urls

    arxiv_id2templates_new = {}
    for arxiv_id, templates in arxiv_id2templates.items():
        key2urls = {
            key: _format_template_url(key)
            for key in templates
            if _is_url_ok(_format_template_url(key))
        }
        if key2urls:
            arxiv_id2templates_new[arxiv_id] = key2urls

    arxiv_id2cookbooks_new = {}
    for arxiv_id, cookbooks in arxiv_id2cookbooks.items():
        key2urls = {
            key: _format_cookbook_url(key)
            for key in cookbooks
            if _is_url_ok(_format_cookbook_url(key))
        }
        if key2urls:
            arxiv_id2cookbooks_new[arxiv_id] = key2urls

    arxiv_id2type2key2urls = dict.fromkeys(
        arxiv_id2file_names_new
        | arxiv_id2code_urls_new
        | arxiv_id2templates_new
        | arxiv_id2cookbooks_new
    )
    arxiv_id2type2key2urls = {k: {} for k in arxiv_id2type2key2urls}
    for arxiv_id, key2urls in arxiv_id2file_names_new.items():
        arxiv_id2type2key2urls[arxiv_id]["docs"] = key2urls
    for arxiv_id, key2urls in arxiv_id2code_urls_new.items():
        arxiv_id2type2key2urls[arxiv_id]["apis"] = key2urls
    for arxiv_id, key2urls in arxiv_id2templates_new.items():
        arxiv_id2type2key2urls[arxiv_id]["templates"] = key2urls
    for arxiv_id, key2urls in arxiv_id2cookbooks_new.items():
        arxiv_id2type2key2urls[arxiv_id]["cookbooks"] = key2urls

    # reverse sort by the arxiv_id (the newest papers first)
    ret = dict(
        sorted(arxiv_id2type2key2urls.items(), key=lambda item: item[0], reverse=True)
    )
    return ret


def log_results(arxiv_id2type2key2urls):
    arxiv_ids = arxiv_id2type2key2urls.keys()
    doc_number, api_number, templates_number, cookbooks_number = 0, 0, 0, 0
    for type2key2url in arxiv_id2type2key2urls.values():
        if "docs" in type2key2url:
            doc_number += len(type2key2url["docs"])
        if "apis" in type2key2url:
            api_number += len(type2key2url["apis"])
        if "templates" in type2key2url:
            templates_number += len(type2key2url["templates"])
        if "cookbooks" in type2key2url:
            cookbooks_number += len(type2key2url["cookbooks"])
    logger.warning(
        f"Found {len(arxiv_ids)} arXiv references in the {doc_number} docs, {api_number} API Refs,"
        f" {templates_number} Templates, and {cookbooks_number} Cookbooks."
    )


def generate_arxiv_references_page(file_name: Path, papers: list[ArxivPaper]) -> None:
    with open(file_name, "w") as f:
        # Write the table headers
        f.write(
            """# arXiv
            
LangChain implements the latest research in the field of Natural Language Processing.
This page contains `arXiv` papers referenced in the LangChain Documentation, API Reference,
 Templates, and Cookbooks.

From the opposite direction, scientists use `LangChain` in research and reference it in the research papers. 

`arXiv` papers with references to:
 [LangChain](https://arxiv.org/search/?query=langchain&searchtype=all&source=header) | [LangGraph](https://arxiv.org/search/?query=langgraph&searchtype=all&source=header) | [LangSmith](https://arxiv.org/search/?query=langsmith&searchtype=all&source=header)

## Summary

| arXiv id / Title | Authors | Published date 🔻 | LangChain Documentation|
|------------------|---------|-------------------|------------------------|
"""
        )
        for paper in papers:
            refs = []
            if paper.referencing_doc2url:
                refs += [
                    "`Docs:` "
                    + ", ".join(
                        f"[{key}]({url})"
                        for key, url in paper.referencing_doc2url.items()
                    )
                ]
            if paper.referencing_api_ref2url:
                refs += [
                    "`API:` "
                    + ", ".join(
                        f"[{_compact_module_full_name(key)}]({url})"
                        for key, url in paper.referencing_api_ref2url.items()
                    )
                ]
            if paper.referencing_template2url:
                refs += [
                    "`Template:` "
                    + ", ".join(
                        f"[{key}]({url})"
                        for key, url in paper.referencing_template2url.items()
                    )
                ]
            if paper.referencing_cookbook2url:
                refs += [
                    "`Cookbook:` "
                    + ", ".join(
                        f"[{str(key).replace('_', ' ').title()}]({url})"
                        for key, url in paper.referencing_cookbook2url.items()
                    )
                ]
            refs_str = ", ".join(refs)

            title_link = f"[{paper.title}]({paper.url})"
            f.write(
                f"| {' | '.join([f'`{paper.arxiv_id}` {title_link}', ', '.join(paper.authors), paper.published_date.replace('-', '&#8209;'), refs_str])}\n"
            )

        for paper in papers:
            docs_refs = (
                f"   - **Documentation:** {', '.join(f'[{key}]({url})' for key, url in paper.referencing_doc2url.items())}"
                if paper.referencing_doc2url
                else ""
            )
            api_ref_refs = (
                f"   - **API Reference:** {', '.join(f'[{_compact_module_full_name(key)}]({url})' for key, url in paper.referencing_api_ref2url.items())}"
                if paper.referencing_api_ref2url
                else ""
            )
            template_refs = (
                f"   - **Template:** {', '.join(f'[{key}]({url})' for key, url in paper.referencing_template2url.items())}"
                if paper.referencing_template2url
                else ""
            )
            cookbook_refs = (
                f"   - **Cookbook:** {', '.join(f'[{key}]({url})' for key, url in paper.referencing_cookbook2url.items())}"
                if paper.referencing_cookbook2url
                else ""
            )
            refs = "\n".join(
                [
                    el
                    for el in [docs_refs, api_ref_refs, template_refs, cookbook_refs]
                    if el
                ]
            )
            f.write(
                f"""
## {paper.title}

- **Authors:** {', '.join(paper.authors)}
- **arXiv id:** [{paper.arxiv_id}]({paper.url})  **Published Date:** {paper.published_date}
- **LangChain:**

{refs}

**Abstract:** {paper.abstract}
                """
            )

    logger.warning(f"Created the {file_name} file with {len(papers)} arXiv references.")


def main():
    # search the documentation and the API Reference for arXiv references:
    arxiv_id2module_name_and_members = search_code_for_arxiv_references(CODE_DIR)
    arxiv_id2code_urls = convert_module_name_and_members_to_urls(
        arxiv_id2module_name_and_members
    )
    arxiv_id2file_names = search_documentation_for_arxiv_references(DOCS_DIR)
    arxiv_id2templates = search_templates_for_arxiv_references(TEMPLATES_DIR)
    arxiv_id2cookbooks = search_cookbooks_for_arxiv_references(COOKBOOKS_DIR)
    arxiv_id2type2key2urls = compound_urls(
        arxiv_id2file_names, arxiv_id2code_urls, arxiv_id2templates, arxiv_id2cookbooks
    )
    log_results(arxiv_id2type2key2urls)

    # get the arXiv paper information
    papers = ArxivAPIWrapper().get_papers(arxiv_id2type2key2urls)

    # generate the arXiv references page
    output_file = DOCS_DIR / "additional_resources" / "arxiv_references.mdx"
    generate_arxiv_references_page(output_file, papers)


if __name__ == "__main__":
    main()
