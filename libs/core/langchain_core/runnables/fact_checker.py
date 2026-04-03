from typing import Any, Dict, List
import re

from langchain_core.messages import AIMessage
from langchain_core.runnables.base import Runnable


class RunnableFactChecker(Runnable[Dict[str, Any], AIMessage]):
    """Runnable that verifies LLM outputs against context."""

    def __init__(self, llm, strict_mode: bool = False):
        self.llm = llm
        self.strict_mode = strict_mode

    def invoke(self, input: Dict[str, Any], config: Any = None) -> AIMessage:
        context_docs = input.get("context", [])
        answer = input.get("answer") or input.get("content")

        if not answer:
            return AIMessage(content="", response_metadata={"confidence_score": 0.0})

        sentences = self._split_sentences(answer)

        context_text = "\n\n".join(
            getattr(doc, "page_content", str(doc)) for doc in context_docs
        )

        supported = []

        for sentence in sentences:
            verdict = self._verify(sentence, context_text)
            if verdict.strip().upper() == "TRUE":
                supported.append(sentence)

        score = len(supported) / len(sentences) if sentences else 0.0

        final_text = (
            " ".join(supported) if self.strict_mode else " ".join(sentences)
        )

        return AIMessage(
            content=final_text,
            response_metadata={"confidence_score": score},
        )

    async def ainvoke(self, input: Dict[str, Any], config: Any = None) -> AIMessage:
        return self.invoke(input, config)

    def _verify(self, sentence: str, context: str) -> str:
        prompt = f"""Context:
{context}

Statement:
{sentence}

Is this supported? Answer TRUE or FALSE."""
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response))

    def _split_sentences(self, text: str) -> List[str]:
        return re.split(r"(?<=[.!?])\s+", text.strip())
