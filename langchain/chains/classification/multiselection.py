"""Perform classification / selection using language models."""
from __future__ import annotations

import csv
from bs4 import BeautifulSoup
from io import StringIO
from typing import Sequence, Mapping, Any, Optional, Dict, List, cast, Set

from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.schema import BaseOutputParser
from typing import TypedDict


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


class MultiSelectionInput(TypedDict):
    """Input for the multi-selection chain."""

    question: str
    records: Sequence[Mapping[str, Any]]
    delimiter: str
    columns: Optional[Sequence[str]]


class MultiSelectionOutput(TypedDict):
    """Output for the multi-selection chain."""

    records: Sequence[Mapping[str, Any]]


from itertools import islice


def batch(iterable, size):
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
    ) -> MultiSelectionOutput:
        """Run the chain."""
        choices = inputs["choices"]
        question = inputs["question"]
        columns = inputs.get("columns", None)

        selected = []
        max_choices = 30

        for choice_batch in batch(choices, max_choices):
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
                    callbacks=run_manager.get_child(),
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
