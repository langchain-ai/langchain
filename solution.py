# suppress the noisy “Failed to use model_dump” debug messages from langchain_core

import logging
import os

# 1️⃣  Set the global log level to INFO (or WARNING) before importing LangChain
os.environ["LC_LOG_LEVEL"] = "INFO"          # LangChain respects this env var
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# 2️⃣  (Optional) add a filter that silently drops the specific message
class SuppressModelDumpFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # return False to drop the log
        return "Failed to use model_dump" not in record.getMessage()

logging.getLogger("langchain_core").addFilter(SuppressModelDumpFilter())

# 3️⃣  Your normal LangChain code goes below
# ------------------------------------------------------------------
#   from langchain import SomeClass
#   ...
# ------------------------------------------------------------------

# Example: a simple call that previously had the noisy log
if __name__ == "__main__":
    from langchain_community.tools import GoogleSearchResults
    tool = GoogleSearchResults()
    print(tool.run("Python logging best practices"))