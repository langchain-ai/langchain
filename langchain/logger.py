from abc import ABC, abstractmethod
from typing import Optional, Any
from langchain.printing import print_text
from pathlib import Path

CONTEXT_KEY = "__context__"


class Logger(ABC):

    @abstractmethod
    def log_start_of_chain(self, inputs):
        """"""

    @abstractmethod
    def log_end_of_chain(self, outputs):
        """"""

    @abstractmethod
    def log(self, text: str, context: dict, **kwargs):
        """"""


class PrintLogger(Logger):
    def log_start_of_chain(self, inputs):
        """"""
        print("\n\n\033[1m> Entering new chain...\033[0m")

    def log_end_of_chain(self, outputs):
        """"""
        print("\n\033[1m> Finished chain.\033[0m")

    def log(self, text: str, context: dict, title: Optional[str ] =None ,**kwargs:Any):
        """"""
        if title is not None:
            print(title)
        print_text(text, **kwargs)

import json
class JSONLogger(Logger):

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def log_start_of_chain(self, inputs):
        """"""
        fname = self.log_dir / f"{inputs[CONTEXT_KEY]['id']}.json"
        if not fname.exists():
            with open(fname, 'w') as f:
                json.dump([], f)

    def log_end_of_chain(self, outputs):
        """"""
        fname = self.log_dir / f"{outputs[CONTEXT_KEY]['id']}.json"
        with open(fname) as f:
            logs = json.load(f)
        logs.append(outputs)
        with open(fname, 'w') as f:
            json.dump(logs, f)

    def log(self, text: str, context: dict, title: Optional[str ] =None ,**kwargs:Any):
        """"""
        fname = self.log_dir / f"{context['id']}.json"
        with open(fname) as f:
            logs = json.load(f)
        logs.append({"text": text, "title": title})
        with open(fname, 'w') as f:
            json.dump(logs, f)

