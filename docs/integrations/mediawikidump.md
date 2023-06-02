# MediaWikiDump

>[MediaWiki XML Dumps](https://www.mediawiki.org/wiki/Manual:Importing_XML_dumps) contain the content of a wiki 
> (wiki pages with all their revisions), without the site-related data. A XML dump does not create a full backup 
> of the wiki database, the dump does not contain user accounts, images, edit logs, etc.


## Installation and Setup

We need to install several python packages.

The `mediawiki-utilities` supports XML schema 0.11 in unmerged branches.
```bash
pip install -qU git+https://github.com/mediawiki-utilities/python-mwtypes@updates_schema_0.11
```

The `mediawiki-utilities mwxml` has a bug, fix PR pending.

```bash
pip install -qU git+https://github.com/gdedrouas/python-mwxml@xml_format_0.11
pip install -qU mwparserfromhell
```

## Document Loader

See a [usage example](../modules/indexes/document_loaders/examples/mediawikidump.ipynb).


```python
from langchain.document_loaders import MWDumpLoader
```
