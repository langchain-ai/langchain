# langchain-upstage

This package contains the LangChain integrations for [Upstage](https://upstage.ai) through their [APIs](https://developers.upstage.ai/docs/getting-started/models).

## Installation and Setup

- Install the LangChain partner package
```bash
pip install -U langchain-upstage
```

- Get an Upstage api key from [Upstage Console](https://console.upstage.ai/home) and set it as an environment variable (`UPSTAGE_API_KEY`)

## Chat Models

This package contains the `ChatUpstage` class, which is the recommended way to interface with Upstage models.

See a [usage example](https://python.langchain.com/docs/integrations/chat/upstage)

## Embeddings

See a [usage example](https://python.langchain.com/docs/integrations/text_embedding/upstage)

Use `solar-embedding-1-large` model for embeddings. Do not add suffixes such as `-query` or `-passage` to the model name.
`UpstageEmbeddings` will automatically add the suffixes based on the method called.

## LayoutAnalysis Loader

See a [usage example](https://python.langchain.com/v0.1/docs/integrations/document_loaders/upstage/)

To load an image using UpstageLayoutAnalysisLoader, OCR must be enabled by setting `use_ocr=True` when constructing the loader, as shown in the example below:

```python
from langchain_upstage import UpstageLayoutAnalysisLoader

file_path = "/PATH/TO/YOUR/FILE.image"
layzer = UpstageLayoutAnalysisLoader(file_path, split="page", use_ocr=True)

# For improved memory efficiency, consider using the lazy_load method to load documents page by page.
docs = layzer.load()  # or layzer.lazy_load()

for doc in docs[:3]:
    print(doc)
```

If you are a Windows user, please ensure that the [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) is installed before using the loader.
