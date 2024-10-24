import itertools
import multiprocessing
import re
import sys
from pathlib import Path
from typing import Optional

# List of 4-tuples (integration_name, display_name, concept_page, how_to_fragment)
INTEGRATION_INFO = [
    ("chat", "Chat model", "chat_models", "chat-models"),
    ("llms", "LLM", "text_llms", "llms"),
    ("text_embedding", "Embedding model", "embedding_models", "embedding-models"),
    ("document_loaders", "Document loader", "document_loaders", "document-loaders"),
    ("vectorstores", "Vector store", "vectorstores", "vector-stores"),
    ("retrievers", "Retriever", "retrievers", "retrievers"),
    ("tools", "Tool", "tools", "tools"),
    # stores is a special case because there are no key-value store how-tos yet
    # this is a placeholder for when we do have them
    # for now the related links section will only contain the conceptual guide.
    ("stores", "Key-value store", "key_value_stores", "key-value-stores"),
]

# Create a dictionary with key being the first element (integration_name) and value being the rest of the tuple
INTEGRATION_INFO_DICT = {
    integration_name: rest for integration_name, *rest in INTEGRATION_INFO
}

RELATED_LINKS_SECTION = """## Related
- {concept_display_name} [conceptual guide](/docs/concepts/{concept_page})
- {concept_display_name} [how-to guides](/docs/how_to/#{how_to_fragment})
"""

RELATED_LINKS_WITHOUT_HOW_TO_SECTION = """## Related
- {concept_display_name} [conceptual guide](/docs/concepts/{concept_page})
"""


def _generate_related_links_section(
    integration_type: str, notebook_name: str
) -> Optional[str]:
    if integration_type not in INTEGRATION_INFO_DICT:
        return None
    concept_display_name, concept_page, how_to_fragment = INTEGRATION_INFO_DICT[
        integration_type
    ]

    # Special case because there are no key-value store how-tos yet
    if integration_type == "stores":
        return RELATED_LINKS_WITHOUT_HOW_TO_SECTION.format(
            concept_display_name=concept_display_name,
            concept_page=concept_page,
        )

    return RELATED_LINKS_SECTION.format(
        concept_display_name=concept_display_name,
        concept_page=concept_page,
        how_to_fragment=how_to_fragment,
    )


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
