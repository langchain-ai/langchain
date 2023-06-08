Document Loaders
==========================

.. note::
   `Conceptual Guide <https://docs.langchain.com/docs/components/indexing/document-loaders>`_


Combining language models with your own text data is a powerful way to differentiate them.
The first step in doing this is to load the data into "Documents" - a fancy way of say some pieces of text.
The document loader is aimed at making this easy.


The following document loaders are provided:


Transform loaders
------------------------------

These **transform** loaders transform data from a specific format into the Document format.
For example, there are **transformers** for CSV and SQL.
Mostly, these loaders input data from files but sometime from URLs.

A primary driver of a lot of these transformers is the `Unstructured <https://github.com/Unstructured-IO/unstructured>`_ python package.
This package transforms many types of files - text, powerpoint, images, html, pdf, etc - into text data.

For detailed instructions on how to get set up with Unstructured, see installation guidelines `here <https://github.com/Unstructured-IO/unstructured#coffee-getting-started>`_.


.. toctree::
   :maxdepth: 1
   :glob:

   ./document_loaders/examples/audio.ipynb
   ./document_loaders/examples/conll-u.ipynb
   ./document_loaders/examples/copypaste.ipynb
   ./document_loaders/examples/csv.ipynb
   ./document_loaders/examples/email.ipynb
   ./document_loaders/examples/epub.ipynb
   ./document_loaders/examples/evernote.ipynb
   ./document_loaders/examples/excel.ipynb
   ./document_loaders/examples/facebook_chat.ipynb
   ./document_loaders/examples/file_directory.ipynb
   ./document_loaders/examples/html.ipynb
   ./document_loaders/examples/image.ipynb
   ./document_loaders/examples/jupyter_notebook.ipynb
   ./document_loaders/examples/json.ipynb
   ./document_loaders/examples/markdown.ipynb
   ./document_loaders/examples/microsoft_powerpoint.ipynb
   ./document_loaders/examples/microsoft_word.ipynb
   ./document_loaders/examples/odt.ipynb
   ./document_loaders/examples/pandas_dataframe.ipynb
   ./document_loaders/examples/pdf.ipynb
   ./document_loaders/examples/sitemap.ipynb
   ./document_loaders/examples/subtitle.ipynb
   ./document_loaders/examples/telegram.ipynb
   ./document_loaders/examples/toml.ipynb
   ./document_loaders/examples/unstructured_file.ipynb
   ./document_loaders/examples/url.ipynb
   ./document_loaders/examples/web_base.ipynb
   ./document_loaders/examples/weather.ipynb
   ./document_loaders/examples/whatsapp_chat.ipynb



Public dataset or service loaders
----------------------------------
These datasets and sources are created for public domain and we use queries to search there
and download necessary documents.
For example, **Hacker News** service.

We don't need any access permissions to these datasets and services.


.. toctree::
   :maxdepth: 1
   :glob:

   ./document_loaders/examples/arxiv.ipynb
   ./document_loaders/examples/azlyrics.ipynb
   ./document_loaders/examples/bilibili.ipynb
   ./document_loaders/examples/college_confidential.ipynb
   ./document_loaders/examples/gutenberg.ipynb
   ./document_loaders/examples/hacker_news.ipynb
   ./document_loaders/examples/hugging_face_dataset.ipynb
   ./document_loaders/examples/ifixit.ipynb
   ./document_loaders/examples/imsdb.ipynb
   ./document_loaders/examples/mediawikidump.ipynb
   ./document_loaders/examples/wikipedia.ipynb
   ./document_loaders/examples/youtube_transcript.ipynb


Proprietary dataset or service loaders
--------------------------------------
These datasets and services are not from the public domain.
These loaders mostly transform data from specific formats of applications or cloud services,
for example **Google Drive**.

We need access tokens and sometime other parameters to get access to these datasets and services.


.. toctree::
   :maxdepth: 1
   :glob:

   ./document_loaders/examples/airbyte_json.ipynb
   ./document_loaders/examples/apify_dataset.ipynb
   ./document_loaders/examples/aws_s3_directory.ipynb
   ./document_loaders/examples/aws_s3_file.ipynb
   ./document_loaders/examples/azure_blob_storage_container.ipynb
   ./document_loaders/examples/azure_blob_storage_file.ipynb
   ./document_loaders/examples/blackboard.ipynb
   ./document_loaders/examples/blockchain.ipynb
   ./document_loaders/examples/chatgpt_loader.ipynb
   ./document_loaders/examples/confluence.ipynb
   ./document_loaders/examples/diffbot.ipynb
   ./document_loaders/examples/discord_loader.ipynb
   ./document_loaders/examples/docugami.ipynb
   ./document_loaders/examples/duckdb.ipynb
   ./document_loaders/examples/figma.ipynb
   ./document_loaders/examples/gitbook.ipynb
   ./document_loaders/examples/git.ipynb
   ./document_loaders/examples/google_bigquery.ipynb
   ./document_loaders/examples/google_cloud_storage_directory.ipynb
   ./document_loaders/examples/google_cloud_storage_file.ipynb
   ./document_loaders/examples/google_drive.ipynb
   ./document_loaders/examples/image_captions.ipynb
   ./document_loaders/examples/iugu.ipynb
   ./document_loaders/examples/joplin.ipynb
   ./document_loaders/examples/microsoft_onedrive.ipynb
   ./document_loaders/examples/modern_treasury.ipynb
   ./document_loaders/examples/notiondb.ipynb
   ./document_loaders/examples/notion.ipynb
   ./document_loaders/examples/obsidian.ipynb
   ./document_loaders/examples/psychic.ipynb
   ./document_loaders/examples/pyspark_dataframe.ipynb
   ./document_loaders/examples/readthedocs_documentation.ipynb
   ./document_loaders/examples/reddit.ipynb
   ./document_loaders/examples/roam.ipynb
   ./document_loaders/examples/slack.ipynb
   ./document_loaders/examples/spreedly.ipynb
   ./document_loaders/examples/stripe.ipynb
   ./document_loaders/examples/tomarkdown.ipynb
   ./document_loaders/examples/twitter.ipynb
