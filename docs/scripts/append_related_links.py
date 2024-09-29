import itertools
import multiprocessing
import re
import sys
from pathlib import Path


def _generate_related_links_section(integration_type: str, notebook_name: str):
    concept_display_name = None
    concept_heading = None
    if integration_type == "chat":
        concept_display_name = "Chat model"
        concept_heading = "chat-models"
    elif integration_type == "llms":
        concept_display_name = "LLM"
        concept_heading = "llms"
    elif integration_type == "text_embedding":
        concept_display_name = "Embedding model"
        concept_heading = "embedding-models"
    elif integration_type == "document_loaders":
        concept_display_name = "Document loader"
        concept_heading = "document-loaders"
    elif integration_type == "vectorstores":
        concept_display_name = "Vector store"
        concept_heading = "vector-stores"
    elif integration_type == "retrievers":
        concept_display_name = "Retriever"
        concept_heading = "retrievers"
    elif integration_type == "tools":
        concept_display_name = "Tool"
        concept_heading = "tools"
    elif integration_type == "stores":
        concept_display_name = "Key-value store"
        concept_heading = "key-value-stores"
        # Special case because there are no key-value store how-tos yet
        return f"""## Related

- [{concept_display_name} conceptual guide](/docs/concepts/#{concept_heading})
"""
    else:
        return None
    return f"""## Related

- {concept_display_name} [conceptual guide](/docs/concepts/#{concept_heading})
- {concept_display_name} [how-to guides](/docs/how_to/#{concept_heading})
"""


def _process_path(doc_path: Path):
    content = doc_path.read_text()
    pattern = r"/docs/integrations/([^/]+)/([^/]+).mdx?"
    match = re.search(pattern, str(doc_path))
    if match and match.group(2) != "index":
        integration_type = match.group(1)
        notebook_name = match.group(2)
        related_links_section = _generate_related_links_section(
            integration_type, notebook_name
        )
        if related_links_section:
            content = content + "\n\n" + related_links_section
            doc_path.write_text(content)


if __name__ == "__main__":
    output_docs_dir = Path(sys.argv[1])

    mds = output_docs_dir.rglob("integrations/**/*.md")
    mdxs = output_docs_dir.rglob("integrations/**/*.mdx")
    paths = itertools.chain(mds, mdxs)
    # modify all md files in place
    with multiprocessing.Pool() as pool:
        pool.map(_process_path, paths)
