import tiktoken
from unstructured.nlp.tokenize import download_nltk_packages


def download_tiktoken_data():
    # This will trigger the download and caching of the necessary files
    _ = tiktoken.encoding_for_model("gpt2")
    _ = tiktoken.encoding_for_model("gpt-3.5-turbo")
    _ = tiktoken.encoding_for_model("gpt-4o-mini")


if __name__ == "__main__":
    download_tiktoken_data()
    download_nltk_packages()
