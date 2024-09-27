import tiktoken

# This will trigger the download and caching of the necessary files
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
