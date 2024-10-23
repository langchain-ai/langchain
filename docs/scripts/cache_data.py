import nltk
import tiktoken


def download_tiktoken_data():
    # This will trigger the download and caching of the necessary files
    _ = tiktoken.encoding_for_model("gpt-3.5-turbo")


def download_nltk_data():
    nltk.download("punkt")


if __name__ == "__main__":
    download_tiktoken_data()
    download_nltk_data()
