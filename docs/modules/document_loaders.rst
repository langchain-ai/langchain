Document Loaders
==========================

Combining language models with your own text data is a powerful way to differentiate them.
The first step in doing this is to load the data into "documents" - a fancy way of say some pieces of text.
This module is aimed at making this easy.

A primary driver of a lot of this is the `Unstructured <https://github.com/Unstructured-IO/unstructured>`_ python package.
This package is a great way to transform all types of files - text, powerpoint, images, html, pdf, etc - into text data.

For detailed instructions on how to get set up with Unstructured, see installation guidelines `here <https://github.com/Unstructured-IO/unstructured#coffee-getting-started>`_.

The following sections of documentation are provided:

- `Key Concepts <./document_loaders/key_concepts.html>`_: A conceptual guide going over the various concepts related to loading documents.

- `How-To Guides <./document_loaders/how_to_guides.html>`_: A collection of how-to guides. These highlight different types of loaders.




.. toctree::
   :maxdepth: 1
   :caption: Document Loaders
   :name: Document Loaders
   :hidden:

   ./document_loaders/key_concepts.md
   ./document_loaders/how_to_guides.rst