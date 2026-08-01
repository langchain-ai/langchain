# langchain-docuweave

LangChain integration for [DocuWeave](https://github.com/venkateswararao18/docuweave) — a PDF loader that reconstructs the document's heading hierarchy before chunking, so each chunk knows which section it came from.

## Installation

```bash
pip install -U langchain-docuweave
```

## Usage

```python
from langchain_docuweave import DocuWeaveLoader

loader = DocuWeaveLoader("paper.pdf", max_tokens=512)
docs = loader.load()

print(docs[0].metadata["section_path"])
# "3 Methods > 3.2 Experimental Setup"

# or stream one chunk at a time
for doc in loader.lazy_load():
    print(doc.page_content[:80])
    print(doc.metadata["section_path"])
```

## Metadata

Each returned `Document` includes:

| Field | Description |
|---|---|
| `source` | Path to the PDF |
| `section_title` | Heading text of the section |
| `section_path` | Full breadcrumb, e.g. `"3 Methods > 3.2 Setup"` |
| `section_level` | Nesting depth (0 = top-level) |
| `page_start` / `page_end` | 1-based page numbers |
| `previous_chunk_id` / `next_chunk_id` | Linked-list pointers for context expansion |
| `hierarchy_confidence` | `[0, 1]` score; below 0.3 means scanned/image-heavy PDF |

## Why not PyMuPDFLoader?

`PyMuPDFLoader` extracts text per page. `DocuWeaveLoader` builds a section tree from heading signals across the full document and cuts chunks at section boundaries. In a benchmark of 390 PDFs / 3,927 QA pairs, this improved Recall@1 by **+23.4%** over recursive character splitting (p<0.01, bge-base-en-v1.5).
