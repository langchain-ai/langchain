from typing import List

from langchain.text_splitter import TextSplitter


class AI21TextSplitter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        pass