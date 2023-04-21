""""""
import random
import re
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from pydantic import Field

from langchain.document_loaders import TextLoader
from langchain.experimental.retriever_eval.base import (
    ExpectedSubstringsTestCase,
    RetrieverTestCase,
)
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter


class ManyDocsTestCase(ExpectedSubstringsTestCase):
    """"""

    @classmethod
    def from_config(
        cls, retrieve: int = 5, total: int = 100, seed: int = 0, **kwargs: Any
    ) -> "ManyDocsTestCase":
        """"""
        random.seed(seed)
        name = f"Many docs ({retrieve=}, {total=})"
        text_template = "On {date} the peak temperature was {temp} degrees"
        dates = pd.date_range(start="01-01-2023", freq="D", periods=total).astype(str)
        temps = [str(random.randint(50, 80)) for _ in range(len(dates))]
        texts = [text_template.format(date=d, temp=t) for d, t in zip(dates, temps)]
        docs = [Document(page_content=t) for t in texts]

        sample_idxs = random.choices(range(len(dates)), k=retrieve)
        expected_dates = [dates[i] for i in sample_idxs]
        query = f"What were the peak temperatures on {', '.join(expected_dates)}?"
        return cls(
            name=name, query=query, docs=docs, expected_substrings=expected_dates
        )


class RedundantDocsTestCase(ExpectedSubstringsTestCase):
    """"""

    @classmethod
    def from_config(cls, **kwargs: Any) -> "RedundantDocsTestCase":
        """"""
        name = "Redundant docs"
        texts = [
            "OpenAI announces the release of GPT-5",
            "GPT-5 released by OpenAI",
            "The next-generation OpenAI GPT model is here",
            "GPT-5: OpenAI's next model is the biggest yet",
            "Sam Altman's OpenAI comes out with new GPT-5 model",
            "GPT-5 is here. What you need to know about the OpenAI model",
            "OpenAI announces ChatGPT successor GPT-5",
            "5 jaw-dropping things OpenAI's GPT-5 can do that ChatGPT couldn't",
            "OpenAI's GPT-5 Is Exciting and Scary",
            "OpenAI announces GPT-5, the new generation of AI",
            "OpenAI says new model GPT-5 is more creative and less",
            "Meta open sources new AI model, largest yet",
        ]
        docs = [Document(page_content=t) for t in texts]
        query = "What companies have recently released new models?"
        expected_substrings = ["OpenAI", "Meta"]
        return cls(
            name=name, docs=docs, query=query, expected_substrings=expected_substrings
        )


class EntityLinkingTestCase(ExpectedSubstringsTestCase):
    """"""

    @classmethod
    def from_config(
        cls, filler_texts: Optional[List[str]] = None, **kwargs: Any
    ) -> "EntityLinkingTestCase":
        """"""
        if filler_texts is None:
            filler_docs = TextLoader(
                "../docs/modules/state_of_the_union.txt"
            ).load_and_split()
            filler_texts = [d.page_content for d in filler_docs]
        name = f"Entity linking (num_filler={len(filler_texts)})"
        texts = [
            "The founder of ReallyCoolAICompany LLC is from Louisville, Kentucky.",
            "Melissa Harkins, founder of ReallyCoolAICompany LLC, said in a recent interview that she will be stepping down as CEO.",
        ]
        texts = texts + filler_texts
        docs = [Document(page_content=t) for t in texts]
        query = "Where is Melissa Harkins from?"
        expected_substrings = ["Harkins", "Louisville"]
        return cls(
            name=name,
            docs=docs,
            query=query,
            expected_substrings=expected_substrings,
            can_edit_docs=False,
        )


class TemporalQueryTestCase(RetrieverTestCase):
    """"""

    correct_date: str

    def check_retrieved_docs(self, retrieved_docs: List[Document]) -> bool:
        """"""
        return any(d.metadata["date"] == self.correct_date for d in retrieved_docs)

    @classmethod
    def from_config(
        cls,
        options: Optional[List[str]] = None,
        phrasings: Optional[List[str]] = None,
        num_docs: int = 200,
        seed: int = 0,
        **kwargs: Any,
    ) -> "TemporalQueryTestCase":
        """"""
        random.seed(seed)
        if options is None:
            options = [
                "happy",
                "sad",
                "confused",
                "angry",
                "disgusted",
                "scared",
                "thankful",
                "astonished",
                "calm",
            ]
        if phrasings is None:
            phrasings = [
                "Today I felt {option}",
                "I felt {option} today",
                "I was really {option} today",
                "My primary emotion is {option}",
                "Everybody says I seemed so {option}",
            ]
        name = f"Temporal query ({num_docs=})"
        options_sample = random.choices(options, k=num_docs - 1) + [options[0]]
        texts = [
            phrase.format(option=option)
            for phrase, option in zip(
                random.choices(phrasings, k=num_docs), options_sample
            )
        ]
        dates = pd.date_range(start="01-01-2023", freq="D", periods=num_docs).astype(
            str
        )
        metadatas = [{"date": d} for d in dates]
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        query = f"When was the first time I mentioned being {options[0]}"
        correct_date = dates[options_sample.index(options[0])]
        return cls(name=name, docs=docs, query=query, correct_date=correct_date)


class RevisedStatementTestCase(ExpectedSubstringsTestCase):
    """"""

    @classmethod
    def from_config(
        cls, filler_text: Optional[str] = None, **kwargs
    ) -> "RevisedStatementTestCase":
        """"""
        if filler_text is None:
            filler_text = (
                TextLoader("../docs/modules/state_of_the_union.txt")
                .load()[0]
                .page_content
            )
        texts = CharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_text(
            filler_text
        )
        docs = [Document(page_content=t) for t in texts]
        updates = [
            "We are receiving reports of a magnitude 10 earthquake in Japan",
            "The latest reports are that the earthquake that has hit Japan is actually of magnitude 8.2",
            "Now the earthquake in Japan has been downgraded to magnitude 7",
            "Looks like the earthquake is back up to an 8",
            "The latest news is that the earthquake was of magnitude 3",
            "No no it's a magnitude 4",
            "I heard the earthquake is 6.3",
            "Or did they say the earthquake in Japan was a magnitude 6.2",
            "The Japanese arthquake is actually being recorded as a magnitude 12",
            "Sorry correction, my Japanese was poor, the magnitude of the earthquake is 2",
            "The Japanese earthquake is now being recorded as magnitude 5",
        ]
        for update, doc in zip(updates, docs):
            doc.page_content += " " + update + "."
        query = "What is the latest reported magnitude of the earthquake in Japan?"
        num_revisions = len(updates)
        name = f"Revised statement ({num_revisions=})"
        expected_substrings = ["5"]
        return cls(
            name=name, docs=docs, query=query, expected_substrings=expected_substrings
        )


class LongTextOneFactTestCase(ExpectedSubstringsTestCase):
    """"""

    @classmethod
    def from_config(
        cls, filler_text: Optional[str] = None, **kwargs
    ) -> "LongTextOneFactTestCase":
        if filler_text is None:
            filler_text = (
                TextLoader("../docs/modules/state_of_the_union.txt")
                .load()[0]
                .page_content
            )
        fact = (
            "We've just received reports of a purple monkey invading the White House."
        )
        filler_split = filler_text.split(". ")
        all_text = ". ".join(
            filler_split[: len(filler_split) // 2]
            + [fact]
            + filler_split[len(filler_split) // 2 :]
        )
        doc = Document(page_content=all_text)
        text_len = len(all_text)
        name = f"Fact in long text ({text_len=})"
        query = "What color was the animal that was mentioned?"
        expected_substrings = ["purple"]
        return cls(
            name=name, docs=[doc], query=query, expected_substrings=expected_substrings
        )


def load_transcript() -> List[Document]:
    interview = (
        TextLoader(
            "../../Ian_Goodfellow--Generative_Adversarial_Networks_(GANs)-Artificial_Intelligence_(AI)_Podcast-April_18_2019.md"
        )
        .load()[0]
        .page_content
    )
    speaker_tmpl = "\*\*\[{name}\]\*\*"
    splits = re.split(speaker_tmpl.format(name="(.*)"), interview.strip())
    # Madeup times
    times = np.cumsum([len(splits[i].split()) for i in range(2, len(splits), 2)]) / 2.5
    docs = [
        Document(
            page_content=splits[i + 1].strip(),
            metadata={
                "speaker": splits[i],
                "statement_index": i // 2,
                "time": times[i // 2],
            },
        )
        for i in range(1, len(splits), 2)
    ]
    return docs


class PodcastTestCase(RetrieverTestCase):
    docs: List[Document] = Field(default_factory=load_transcript)


class FirstMentionTestCase(PodcastTestCase, ExpectedSubstringsTestCase):
    name: str = "Podcast First Mention"
    query: str = "What was the first mention of deep learning?"
    expected_substrings: List[str] = Field(
        default_factory=lambda: ['"Deep Learning" book']
    )


class SpeakerTestCase(PodcastTestCase, ExpectedSubstringsTestCase):
    name: str = "Podcast Reference to Speaker"
    query: str = "What did Ian say about how he came up with the idea for GANs?"
    expected_substrings: List[str] = Field(
        default_factory=lambda: ["drinking helped a little bit"]
    )
