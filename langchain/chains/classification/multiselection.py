"""Perform classification / selection using language models."""
from __future__ import annotations

import csv
from io import StringIO
from itertools import islice
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TypeVar,
    cast,
)

from bs4 import BeautifulSoup

from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.schema import BaseOutputParser

MULTI_SELECT_TEMPLATE = """\
Here is a table in CSV format:

{records}

---

question:

{question}

---

Output IDs of rows that answer the question or match the question.

For example, if row id 132 and id 133 are relevant, output: <ids>132,133</ids>

---

Begin:"""


def _extract_content_from_tag(html: str, tag: str) -> List[str]:
    """Extract content from the given tag."""
    soup = BeautifulSoup(html, "lxml")
    queries = []
    for query in soup.find_all(tag):
        queries.append(query.text)
    return queries


class IDParser(BaseOutputParser[List[int]]):
    """An output parser that extracts all IDs from the output."""

    def parse(self, text: str) -> List[int]:
        """Parse the text and return a list of IDs"""
        tags = _extract_content_from_tag(text, "ids")

        if not tags:
            return []

        if len(tags) > 1:
            # Fail if more than 1 tag group is identified
            return []

        tag = tags[0]
        ids = tag.split(",")

        finalized_ids = []
        for idx in ids:
            if idx.isdigit():
                finalized_ids.append(int(idx))
        return finalized_ids


def _write_records_to_string(
    records: Sequence[Mapping[str, Any]],
    *,
    columns: Optional[Sequence[str]] = None,
    delimiter: str = "|",
) -> str:
    """Write records to a CSV string.

    Args:
        records: a list of records, assumes that all records have all keys
        columns: a list of columns to include in the CSV
        delimiter: the delimiter to use

    Returns:
        a CSV string
    """
    buffer = StringIO()
    if columns is None:
        existing_columns: Set[str] = set()
        for record in records:
            existing_columns.update(record.keys())
        _columns: Sequence[str] = sorted(existing_columns)
    else:
        _columns = columns

    # Make sure the id column is always first
    _columns_with_id_first = list(_columns)

    if "id" in _columns_with_id_first:
        _columns_with_id_first.remove("id")

    # Make sure the `id` column is always first
    _columns_with_id_first.insert(0, "id")

    writer = csv.DictWriter(
        buffer,
        fieldnames=_columns_with_id_first,
        delimiter=delimiter,
    )
    writer.writeheader()
    writer.writerows(records)
    buffer.seek(0)
    return buffer.getvalue()


T = TypeVar("T")


def _batch(iterable: Iterable[T], size: int) -> Iterator[List[T]]:
    """Batch an iterable into chunks of size `size`.

    Args:
        iterable: the iterable to batch
        size: the size of each batch

    Returns:
        iterator over batches of size `size` except for last batch which will be up
        to size `size`
    """
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


class MultiSelectChain(Chain):
    """A chain that performs multi-selection from a list of choices."""

    llm_chain: LLMChain

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return ["question", "choices"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return ["selected"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run the chain."""
        choices = inputs["choices"]
        question = inputs["question"]
        columns = inputs.get("columns", None)

        selected: List[Mapping[str, Any]] = []
        # TODO(): Balance choices into equal batches with constraint dependent
        # on context window and prompt
        max_choices = 30

        for choice_batch in _batch(choices, max_choices):
            records_with_ids = [
                {**record, "id": idx} for idx, record in enumerate(choice_batch)
            ]
            records_str = _write_records_to_string(
                records_with_ids, columns=columns, delimiter="|"
            )

            indexes = cast(
                List[int],
                self.llm_chain.predict_and_parse(
                    records=records_str,
                    question=question,
                    callbacks=run_manager.get_child() if run_manager else None,
                ),
            )
            valid_indexes = [idx for idx in indexes if 0 <= idx < len(choice_batch)]
            selected.extend(choice_batch[i] for i in valid_indexes)

        return {
            "selected": selected,
        }

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        choices = inputs["choices"]
        question = inputs["question"]
        columns = inputs.get("columns", None)

        selected: List[Mapping[str, Any]] = []
        # TODO(): Balance choices into equal batches with constraint dependent
        # on context window and prompt
        max_choices = 30

        for choice_batch in _batch(choices, max_choices):
            records_with_ids = [
                {**record, "id": idx} for idx, record in enumerate(choice_batch)
            ]
            records_str = _write_records_to_string(
                records_with_ids, columns=columns, delimiter="|"
            )

            indexes = cast(
                List[int],
                await self.llm_chain.apredict_and_parse(
                    records=records_str,
                    question=question,
                    callbacks=run_manager.get_child() if run_manager else None,
                ),
            )
            valid_indexes = [idx for idx in indexes if 0 <= idx < len(choice_batch)]
            selected.extend(choice_batch[i] for i in valid_indexes)

        return {
            "selected": selected,
        }

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "multilabel_binary_classifier"

    @classmethod
    def from_default(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: str = MULTI_SELECT_TEMPLATE,
        parser: BaseOutputParser = IDParser(),
    ) -> MultiSelectChain:
        """Provide a multilabel binary classifier."""
        prompt_template = PromptTemplate.from_template(prompt, output_parser=parser)
        if set(prompt_template.input_variables) != {"question", "records"}:
            raise ValueError("Prompt must contain only {question} and {records}")

        return cls(
            llm_chain=LLMChain(
                llm=llm,
                prompt=prompt_template,
            )
        )
