Text Splitters
==========================

.. note::
   `Conceptual Guide <https://docs.langchain.com/docs/components/indexing/text-splitters>`_


When you want to deal with long pieces of text, it is necessary to split up that text into chunks.
As simple as this sounds, there is a lot of potential complexity here. Ideally, you want to keep the semantically related pieces of text together. What "semantically related" means could depend on the type of text.
This notebook showcases several ways to do that.

At a high level, text splitters work as following:

1. Split the text up into small, semantically meaningful chunks (often sentences).
2. Start combining these small chunks into a larger chunk until you reach a certain size (as measured by some function).
3. Once you reach that size, make that chunk its own piece of text and then start creating a new chunk of text with some overlap (to keep context between chunks).

That means there are two different axes along which you can customize your text splitter:

1. How the text is split
2. How the chunk size is measured

For an introduction to the default text splitter and generic functionality see:


.. toctree::
   :maxdepth: 1
   :glob:

   ./text_splitters/getting_started.ipynb


Usage examples for the text splitters:

- `Character <./text_splitters/examples/character_text_splitter.html>`_
- `Code (including HTML, Markdown, Latex, Python, etc) <./text_splitters/examples/code_splitter.html>`_
- `NLTK <./text_splitters/examples/nltk.html>`_
- `Recursive Character <./text_splitters/examples/recursive_text_splitter.html>`_
- `spaCy <./text_splitters/examples/spacy.html>`_
- `tiktoken (OpenAI) <./text_splitters/examples/tiktoken_splitter.html>`_


.. toctree::
   :maxdepth: 1
   :caption: Text Splitters
   :name: text_splitters
   :hidden:

   ./text_splitters/examples/character_text_splitter.ipynb
   ./text_splitters/examples/code_splitter.ipynb
   ./text_splitters/examples/nltk.ipynb
   ./text_splitters/examples/recursive_text_splitter.ipynb
   ./text_splitters/examples/spacy.ipynb
   ./text_splitters/examples/tiktoken_splitter.ipynb


Most LLMs are constrained by the number of tokens that you can pass in, which is not the same as the number of characters.
In order to get a more accurate estimate, we can use tokenizers to count the number of tokens in the text.
We use this number inside the `..TextSplitter` classes.
This implemented as the `from_<tokenizer>` methods of the `..TextSplitter` classes:

- `Hugging Face tokenizer <./text_splitters/examples/huggingface_length_function.html>`_
- `tiktoken (OpenAI) tokenizer <./text_splitters/examples/tiktoken.html>`_

.. toctree::
   :maxdepth: 1
   :caption: Text Splitters with Tokens
   :name: text_splitter_with_tokens
   :hidden:

   ./text_splitters/examples/huggingface_length_function.ipynb
   ./text_splitters/examples/tiktoken.ipynb