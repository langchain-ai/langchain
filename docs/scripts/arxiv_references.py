"""Parse arXiv references from the documentation.
Generate a page with a table of the arXiv references with links to the documentation pages.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

from pydantic.v1 import BaseModel, root_validator

# TODO parse docstrings for arXiv references
# TODO Generate a page with a table of the references with correspondent modules/classes/functions.

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(os.path.abspath(__file__)).parents[2]
DOCS_DIR = _ROOT_DIR / "docs" / "docs"
CODE_DIR = _ROOT_DIR / "libs"
ARXIV_ID_PATTERN = r"https://arxiv\.org/(abs|pdf)/(\d+\.\d+)"


@dataclass
class ArxivPaper:
    """ArXiv paper information."""

    arxiv_id: str
    referencing_docs: list[str]  # TODO: Add the referencing docs
    referencing_api_refs: list[str]  # TODO: Add the referencing docs
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

    # loop all the files (ipynb, mdx, md) in the docs folder
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


def convert_module_name_and_members_to_urls(
    arxiv_id2module_name_and_members: dict[str, set[str]],
) -> dict[str, set[str]]:
    arxiv_id2urls = {}
    for arxiv_id, module_name_and_members in arxiv_id2module_name_and_members.items():
        urls = set()
        for module_name_and_member in module_name_and_members:
            module_name, type_and_member = module_name_and_member.split(":")
            if "$" in type_and_member:
                type, member = type_and_member.split("$")
            else:
                type = type_and_member
                member = ""
            _namespace_parts = module_name.split(".")
            if type == "module":
                first_namespace_part = _namespace_parts[0]
                if first_namespace_part.startswith("langchain_"):
                    first_namespace_part = first_namespace_part.replace(
                        "langchain_", ""
                    )
                url = f"{first_namespace_part}_api_reference.html#module-{module_name}"
            elif type in ["class", "function"]:
                second_namespace_part = _namespace_parts[1]
                url = f"{second_namespace_part}/{module_name}.{member}.html#{module_name}.{member}"
            else:
                raise ValueError(
                    f"Unknown type: {type} in the {module_name_and_member}."
                )
            urls.add(url)
        arxiv_id2urls[arxiv_id] = urls
    return arxiv_id2urls


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
            arxiv_id2module_name_and_members_reduced[arxiv_id] = (
                module_name_and_member_reduced
            )
        if removed_modules:
            logger.warning(
                f"{arxiv_id}: Removed the following modules with 2+ -part namespaces: {removed_modules}."
            )
    return arxiv_id2module_name_and_members_reduced


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


def compound_urls(
    arxiv_id2file_names: dict[str, set[str]], arxiv_id2code_urls: dict[str, set[str]]
) -> dict[str, dict[str, set[str]]]:
    arxiv_id2urls = dict()
    for arxiv_id, code_urls in arxiv_id2code_urls.items():
        arxiv_id2urls[arxiv_id] = {"api": code_urls}
        # intersection of the two sets
        if arxiv_id in arxiv_id2file_names:
            arxiv_id2urls[arxiv_id]["docs"] = arxiv_id2file_names[arxiv_id]
    for arxiv_id, file_names in arxiv_id2file_names.items():
        if arxiv_id not in arxiv_id2code_urls:
            arxiv_id2urls[arxiv_id] = {"docs": file_names}
    # reverse sort by the arxiv_id (the newest papers first)
    ret = dict(sorted(arxiv_id2urls.items(), key=lambda item: item[0], reverse=True))
    return ret


def _format_doc_link(doc_paths: list[str]) -> list[str]:
    return [
        f"[{doc_path}](https://python.langchain.com/{doc_path})"
        for doc_path in doc_paths
    ]


def _format_api_ref_link(
    doc_paths: list[str], compact: bool = False
) -> list[str]:  # TODO
    # agents/langchain_core.agents.AgentAction.html#langchain_core.agents.AgentAction
    ret = []
    for doc_path in doc_paths:
        module = doc_path.split("#")[1].replace("module-", "")
        if compact and module.count(".") > 2:
            # langchain_community.llms.oci_data_science_model_deployment_endpoint.OCIModelDeploymentTGI
            # -> langchain_community.llms...OCIModelDeploymentTGI
            module_parts = module.split(".")
            module = f"{module_parts[0]}.{module_parts[1]}...{module_parts[-1]}"
        ret.append(
            f"[{module}](https://api.python.langchain.com/en/latest/{doc_path.split('langchain.com/')[-1]})"
        )
    return ret


def log_results(arxiv_id2urls):
    arxiv_ids = arxiv_id2urls.keys()
    doc_number, api_number = 0, 0
    for urls in arxiv_id2urls.values():
        if "docs" in urls:
            doc_number += len(urls["docs"])
        if "api" in urls:
            api_number += len(urls["api"])
    logger.info(
        f"Found {len(arxiv_ids)} arXiv references in the {doc_number} docs and in {api_number} API Refs."
    )


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
        self, arxiv_id2urls: dict[str, dict[str, set[str]]]
    ) -> list[ArxivPaper]:
        """
        Performs an arxiv search and returns information about the papers found.

        If an error occurs or no documents found, error text
        is returned instead.
        Args:
            arxiv_id2urls: Dictionary with arxiv_id as key and dictionary
             with sets of doc file names and API Ref urls.

        Returns:
            List of ArxivPaper objects.
        """  # noqa: E501

        def cut_authors(authors: list) -> list[str]:
            if len(authors) > 3:
                return [str(a) for a in authors[:3]] + [" et al."]
            else:
                return [str(a) for a in authors]

        if not arxiv_id2urls:
            return []
        try:
            arxiv_ids = list(arxiv_id2urls.keys())
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
                referencing_docs=urls["docs"] if "docs" in urls else [],
                referencing_api_refs=urls["api"] if "api" in urls else [],
            )
            for result, urls in zip(results, arxiv_id2urls.values())
        ]
        return papers


def generate_arxiv_references_page(file_name: str, papers: list[ArxivPaper]) -> None:
    with open(file_name, "w") as f:
        # Write the table headers
        f.write("""# arXiv
            
LangChain implements the latest research in the field of Natural Language Processing.
This page contains `arXiv` papers referenced in the LangChain Documentation and API Reference.

## Summary

| arXiv id / Title | Authors | Published date 🔻 | LangChain Documentation and API Reference |
|------------------|---------|-------------------|-------------------------|
""")
        for paper in papers:
            refs = []
            if paper.referencing_docs:
                refs += [
                    "`Docs:` " + ", ".join(_format_doc_link(paper.referencing_docs))
                ]
            if paper.referencing_api_refs:
                refs += [
                    "`API:` "
                    + ", ".join(
                        _format_api_ref_link(paper.referencing_api_refs, compact=True)
                    )
                ]
            refs_str = ", ".join(refs)

            title_link = f"[{paper.title}]({paper.url})"
            f.write(
                f"| {' | '.join([f'`{paper.arxiv_id}` {title_link}', ', '.join(paper.authors), paper.published_date, refs_str])}\n"
            )

        for paper in papers:
            docs_refs = (
                f"- **LangChain Documentation:** {', '.join(_format_doc_link(paper.referencing_docs))}"
                if paper.referencing_docs
                else ""
            )
            api_ref_refs = (
                f"- **LangChain API Reference:** {', '.join(_format_api_ref_link(paper.referencing_api_refs))}"
                if paper.referencing_api_refs
                else ""
            )
            f.write(f"""
## {paper.title}

- **arXiv id:** {paper.arxiv_id}
- **Title:** {paper.title}
- **Authors:** {', '.join(paper.authors)}
- **Published Date:** {paper.published_date}
- **URL:** {paper.url}
{docs_refs}
{api_ref_refs}

**Abstract:** {paper.abstract}
                """)

    logger.info(f"Created the {file_name} file with {len(papers)} arXiv references.")


def main():
    # search the documentation and the API Reference for arXiv references:
    arxiv_id2module_name_and_members = search_code_for_arxiv_references(CODE_DIR)
    arxiv_id2code_urls = convert_module_name_and_members_to_urls(
        arxiv_id2module_name_and_members
    )
    arxiv_id2file_names = search_documentation_for_arxiv_references(DOCS_DIR)
    arxiv_id2urls = compound_urls(arxiv_id2file_names, arxiv_id2code_urls)
    log_results(arxiv_id2urls)

    # get the arXiv paper information
    papers = ArxivAPIWrapper().get_papers(arxiv_id2urls)

    # generate the arXiv references page
    output_file = str(DOCS_DIR / "additional_resources" / "arxiv_references.mdx")
    generate_arxiv_references_page(output_file, papers)


if __name__ == "__main__":
    main()
