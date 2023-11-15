# PubMed

# PubMed

>[PubMed®](https://pubmed.ncbi.nlm.nih.gov/) by `The National Center for Biotechnology Information, National Library of Medicine` 
> comprises more than 35 million citations for biomedical literature from `MEDLINE`, life science journals, and online books. 
> Citations may include links to full text content from `PubMed Central` and publisher web sites.

## Setup
You need to install a python package.

```bash
pip install xmltodict
```

### Retriever

See a [usage example](/docs/integrations/retrievers/pubmed).

```python
from langchain.retrievers import PubMedRetriever
```

### Document Loader

See a [usage example](/docs/integrations/document_loaders/pubmed).

```python
from langchain.document_loaders import PubMedLoader
```
