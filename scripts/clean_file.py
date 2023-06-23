import argparse
from typing import Optional, Sequence

from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatAnthropic, ChatOpenAI


def parse_args(src: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="Apply an LLM to a code file to proofread docstrings and edit grammar.",
    )
    parser.add_argument("file", type=str, help="File to proofread.")
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        help="Model to use.",
        choices=["anthropic", "openai", "auto"],
    )
    return parser.parse_args(src)


def select_model(text: str, model: str) -> BaseLanguageModel:
    if model == "anthropic":
        return ChatAnthropic(model="claude-v1-100k", temperature=0)
    elif model == "openai":
        return ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    elif model == "auto":
        import tiktoken

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
        num_tokens = len(encoding.encode(text))
        if num_tokens > 15800:
            return ChatAnthropic(model="claude-v1-100k", temperature=0)
        else:
            return ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    else:
        raise ValueError(f"Invalid model {model}")


def main(file: str, model: str) -> str:
    """Run the llm."""
    with open(file, "r") as f:
        text = f.read()
    model_ = select_model(text, model)
    template = """Please review the following code and improve the docstrings to make our documentation clean. Update the docstrings and descriptions as needed in the format of scikit-learn. Provide examples if necessary.
    ```
    {code}
    ```"""
    chain = LLMChain.from_string(llm=model_, template=template)
    cleaned = chain(text, return_only_outputs=True)["text"]
    return cleaned.strip().strip("`")


if __name__ == "__main__":
    args = parse_args()
    cleaned = main(args.file, args.model)
    print(cleaned)
