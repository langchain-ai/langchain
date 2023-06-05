Document Loaders
==========================

Combining language models with your own text data is a powerful way to differentiate them.
The first step in doing this is to load the data into "Documents" - a fancy way to say some pieces of text.
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

   ./document_loaders/examples/conll-u.html
   ./document_loaders/examples/copypaste.html
   ./document_loaders/examples/csv.html
   ./document_loaders/examples/email.html
   ./document_loaders/examples/epub.html
   ./document_loaders/examples/evernote.html
   ./document_loaders/examples/facebook_chat.html
   ./document_loaders/examples/file_directory.html
   ./document_loaders/examples/html.html
   ./document_loaders/examples/image.html
   ./document_loaders/examples/jupyter_notebook.html
   ./document_loaders/examples/json.html
   ./document_loaders/examples/markdown.html
   ./document_loaders/examples/microsoft_powerpoint.html
   ./document_loaders/examples/microsoft_word.html
   ./document_loaders/examples/odt.html
   ./document_loaders/examples/pandas_dataframe.html
   ./document_loaders/examples/pdf.html
   ./document_loaders/examples/sitemap.html
   ./document_loaders/examples/subtitle.html
   ./document_loaders/examples/telegram.html
   ./document_loaders/examples/toml.html
   ./document_loaders/examples/unstructured_file.html
   ./document_loaders/examples/url.html
   ./document_loaders/examples/web_base.html
   ./document_loaders/examples/weather.html
   ./document_loaders/examples/whatsapp_chat.html



Public dataset or service loaders
----------------------------------
These datasets and sources are created for public domain and we use queries to search there
and download necessary documents.
For example, **Hacker News** service.

We don't need any access permissions to these datasets and services.


.. toctree::
   :maxdepth: 1
   :glob:

   ./document_loaders/examples/arxiv.html
   ./document_loaders/examples/azlyrics.html
   ./document_loaders/examples/bilibili.html
   ./document_loaders/examples/college_confidential.html
   ./document_loaders/examples/gutenberg.html
   ./document_loaders/examples/hacker_news.html
   ./document_loaders/examples/hugging_face_dataset.html
   ./document_loaders/examples/ifixit.html
   ./document_loaders/examples/imsdb.html
   ./document_loaders/examples/mediawikidump.html
   ./document_loaders/examples/wikipedia.html
   ./document_loaders/examples/youtube_transcript.html


Proprietary dataset or service loaders
--------------------------------------
These datasets and services are not from the public domain.
These loaders mostly transform data from specific formats of applications or cloud services,
for example **Google Drive**.

We need access tokens and sometime other parameters to get access to these datasets and services.


.. toctree::
   :maxdepth: 1
   :glob:

   ./document_loaders/examples/airbyte_json.html
   ./document_loaders/examples/apify_dataset.html
   ./document_loaders/examples/aws_s3_directory.html
   ./document_loaders/examples/aws_s3_file.html
   ./document_loaders/examples/azure_blob_storage_container.html
   ./document_loaders/examples/azure_blob_storage_file.html
   ./document_loaders/examples/blackboard.html
   ./document_loaders/examples/blockchain.html
   ./document_loaders/examples/chatgpt_loader.html
   ./document_loaders/examples/confluence.html
   ./document_loaders/examples/diffbot.html
   ./document_loaders/examples/discord_loader.html
   ./document_loaders/examples/docugami.html
   ./document_loaders/examples/duckdb.html
   ./document_loaders/examples/figma.html
   ./document_loaders/examples/gitbook.html
   ./document_loaders/examples/git.html
   ./document_loaders/examples/google_bigquery.html
   ./document_loaders/examples/google_cloud_storage_directory.html
   ./document_loaders/examples/google_cloud_storage_file.html
   ./document_loaders/examples/google_drive.html
   ./document_loaders/examples/image_captions.html
   ./document_loaders/examples/iugu.html
   ./document_loaders/examples/joplin.html
   ./document_loaders/examples/microsoft_onedrive.html
   ./document_loaders/examples/modern_treasury.html
   ./document_loaders/examples/notiondb.html
   ./document_loaders/examples/notion.html
   ./document_loaders/examples/obsidian.html
   ./document_loaders/examples/psychic.html
   ./document_loaders/examples/pyspark_dataframe.html
   ./document_loaders/examples/readthedocs_documentation.html
   ./document_loaders/examples/reddit.html
   ./document_loaders/examples/roam.html
   ./document_loaders/examples/slack.html
   ./document_loaders/examples/spreedly.html
   ./document_loaders/examples/stripe.html
   ./document_loaders/examples/tomarkdown.html
   ./document_loaders/examples/twitter.html
