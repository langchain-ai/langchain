Document Loaders
==========================

.. note::
   `Conceptual Guide <https://docs.langchain.com/docs/components/indexing/document-loaders>`_


Combining language models with your own text data is a powerful way to differentiate them.
The first step in doing this is to load the data into "Documents" - a fancy way of say some pieces of text.
The document loader is aimed at making this easy.


The following document loaders are provided:


Transformer Loaders
------------------------------

These **transformer** loaders transform data from a specific format into the Document format.
For example, there are **transformers** for CSV and SQL.
Mostly, these loaders input data from files but sometime from URLs.

A primary driver of a lot of these transformers is the `Unstructured <https://github.com/Unstructured-IO/unstructured>`_ python package.
This package transforms many types of files - text, powerpoint, images, html, pdf, etc - into text data.

For detailed instructions on how to get set up with Unstructured, see installation guidelines `here <https://github.com/Unstructured-IO/unstructured#coffee-getting-started>`_.


.. toctree::
   :maxdepth: 1
   :glob:

   ./document_loaders/examples/examples/CoNLL-U.ipynb
   ./document_loaders/examples/examples/copypaste.ipynb
   ./document_loaders/examples/examples/csv.ipynb
   ./document_loaders/examples/examples/dataframe.ipynb
   ./document_loaders/examples/examples/directory_loader.ipynb
   ./document_loaders/examples/examples/email.ipynb
   ./document_loaders/examples/examples/epub.ipynb
   ./document_loaders/examples/examples/image.ipynb
   ./document_loaders/examples/examples/html.ipynb
   ./document_loaders/examples/examples/evernote.ipynb
   ./document_loaders/examples/examples/facebook_chat.ipynb
   ./document_loaders/examples/examples/markdown.ipynb
   ./document_loaders/examples/examples/notebook.ipynb
   ./document_loaders/examples/examples/pdf.ipynb
   ./document_loaders/examples/examples/powerpoint.ipynb
   ./document_loaders/examples/examples/sitemap.ipynb
   ./document_loaders/examples/examples/srt.ipynb
   ./document_loaders/examples/examples/telegram.ipynb
   ./document_loaders/examples/examples/toml.ipynb
   ./document_loaders/examples/examples/unstructured_file.ipynb
   ./document_loaders/examples/examples/url.ipynb
   ./document_loaders/examples/examples/web_base.ipynb
   ./document_loaders/examples/examples/whatsapp_chat.ipynb
   ./document_loaders/examples/examples/word_document.ipynb


Public dataset or service loaders
----------------------------------
These datasets and sources are created for public domain and we use queries to search there
and download necessary documents.
For example, **Hacker News** service.
You don't need any access permissions to these datasets and services.


.. toctree::
   :maxdepth: 1
   :glob:

   ./document_loaders/examples/examples/arxiv.ipynb
   ./document_loaders/examples/examples/azlyrics.ipynb
   ./document_loaders/examples/examples/bilibili.ipynb
   ./document_loaders/examples/examples/college_confidential.ipynb
   ./document_loaders/examples/examples/gutenberg.ipynb
   ./document_loaders/examples/examples/hn.ipynb
   ./document_loaders/examples/examples/hugging_face_dataset.ipynb
   ./document_loaders/examples/examples/ifixit.ipynb
   ./document_loaders/examples/examples/imsdb.ipynb
   ./document_loaders/examples/examples/mediawikidump.ipynb
   ./document_loaders/examples/examples/youtube.ipynb


Proprietary dataset or service loaders
------------------------------
These datasets and services are not from the public domain.
These loaders mostly transform data from specific formats of applications or cloud services,
for example the **Google Drive**.
We need access tokens and sometime other parameters to get access to these datasets and services.


.. toctree::
   :maxdepth: 1
   :glob:

   ./document_loaders/examples/examples/airbyte_json.ipynb
   ./document_loaders/examples/examples/apify_dataset.ipynb
   ./document_loaders/examples/examples/azure_blob_storage_container.ipynb
   ./document_loaders/examples/examples/azure_blob_storage_file.ipynb
   ./document_loaders/examples/examples/bigquery.ipynb
   ./document_loaders/examples/examples/blackboard.ipynb
   ./document_loaders/examples/examples/blockchain.ipynb
   ./document_loaders/examples/examples/chatgpt_loader.ipynb
   ./document_loaders/examples/examples/confluence.ipynb
   ./document_loaders/examples/examples/diffbot.ipynb
   ./document_loaders/examples/examples/discord_loader.ipynb
   ./document_loaders/examples/examples/duckdb.ipynb
   ./document_loaders/examples/examples/figma.ipynb
   ./document_loaders/examples/examples/gcs_directory.ipynb
   ./document_loaders/examples/examples/gcs_file.ipynb
   ./document_loaders/examples/examples/gitbook.ipynb
   ./document_loaders/examples/examples/git.ipynb
   ./document_loaders/examples/examples/googledrive.ipynb
   ./document_loaders/examples/examples/image_captions.ipynb
   ./document_loaders/examples/examples/modern_treasury.ipynb
   ./document_loaders/examples/examples/notiondb.ipynb
   ./document_loaders/examples/examples/notion.ipynb
   ./document_loaders/examples/examples/obsidian.ipynb
   ./document_loaders/examples/examples/onedrive.ipynb
   ./document_loaders/examples/examples/readthedocs_documentation.ipynb
   ./document_loaders/examples/examples/reddit.ipynb
   ./document_loaders/examples/examples/roam.ipynb
   ./document_loaders/examples/examples/s3_directory.ipynb
   ./document_loaders/examples/examples/s3_file.ipynb
   ./document_loaders/examples/examples/slack_directory.ipynb
   ./document_loaders/examples/examples/spreedly.ipynb
   ./document_loaders/examples/examples/stripe.ipynb
   ./document_loaders/examples/examples/twitter.ipynb
