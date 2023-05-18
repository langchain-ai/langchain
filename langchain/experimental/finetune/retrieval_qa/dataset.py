import glob
import json
import os
import pickle
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "src/langchain"))

from json import JSONDecodeError
from typing import Dict, List

from tqdm import tqdm

from langchain.chains import QAGenerationChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever


def _timestamp() -> str:
    from datetime import datetime

    from pytz import timezone

    return datetime.now(timezone("US/Pacific")).strftime("%Y-%m-%d-%H-%M")


def _extract_corpus_from_transcripts(
    transcripts_glob="./transcripts/vtt/*large.vtt",
) -> str:
    """
    Splice out timestamps from Video Text Track transcripts and
    concat, yielding a continuous string of text (eg not even speaker roles delimited)
    """
    fps = glob.glob(transcripts_glob)
    transcripts = []
    for fp in fps:
        with open(fp) as f:
            transcript = f.read()
            transcript = re.sub(
                r"\d+:?\d+:\d+\.\d+ --> \d+:?\d+:\d+\.\d+.*\n", "", transcript
            )
            transcript = transcript.replace("\n\n", "")
            transcript = re.sub(r"^WEBVTT", "", transcript)
            transcripts.append(transcript)

    return " ".join(transcripts)


def generate_retrieval_qa_dataset(
    dataset_size: int,
    chunks: List[str],
    qa_gen_chain: QAGenerationChain,
    retriever: VectorStoreRetriever,
    checkpoint: bool = True,
) -> List[Dict[str, str]]:
    """Query LM to generate question, answer pairs for a given context.
    Reject any question that cannot be used to retrieve its generating context.

    The goal is to finetune the QA component of a RetrievalQAChain. Without rejection sampling like so,
    when we run the RetrievalQAChain to build prompt, completion pairs, an "imprecise" question will retrieve
    context that's not related at all to the answer, in effect encouraging hallucination.

    Args:
        dataset_size: desired number of question, answer pairs
        chunks:       contexts for question, answer pairs
        retriever:    retriever over chunks
        checkpoint:   save every so often, useful for OpenAI chat competions which cannot currently be processed in parallel

    Returns:
        dicts with "question" and "answer" keys.
    """
    start_time = _timestamp()
    accepted, rejected = [], []

    tic = time.time()
    while len(accepted) < dataset_size:
        if len(accepted) % 25 == 0:
            toc = time.time()
            print(f"{len(accepted)=} {len(rejected)=} after {(toc-tic)/60:.2f} min")
            if checkpoint:
                if not os.path.exist("./tmp"):
                    os.makedirs("./tmp")
                pickle.dump(accepted, open(f"./tmp/accepted-{start_time}.pkl", "wb"))
                pickle.dump(rejected, open(f"./tmp/rejected-{start_time}.pkl", "wb"))

        chunk = random.choice(chunks)
        try:
            qa = qa_gen_chain.run([chunk])[0]
        except JSONDecodeError:
            continue

        docs = retriever.get_relevant_documents(qa["question"])

        if any(doc.page_content == chunk for doc in docs):
            accepted.append((qa, docs))
        else:
            rejected.append((qa, chunk, docs))

    return [qa[0] for qa in accepted]


def generate_retrieval_qa_t2t_dataset(
    qa_dataset: List[Dict[str, str]], qa_chain: RetrievalQA
):
    """Run retrieval step and generate prompt, completion pairs for fine-tuning.

    Returns:
        dicts with "question", "answer", and "question_with_context" keys.
    """
    # https://github.com/hwchase17/langchain/commit/d1b92537b00db5a1eb09bcaa448652781b679b5a
    # I have to hack to expose the prompt that's been "stuffed" with the result of retrieval.
    qa_chain.combine_documents_chain.llm_chain.populate_prompt_cache = True
    qa_chain.combine_documents_chain.llm_chain.prompt_template_cache_key = "question"

    for example in tqdm(qa_dataset):
        qa_chain(example["question"])

    qa_chain.combine_documents_chain.llm_chain.populate_prompt_cache = (
        False  # Unset to un-shunt generations.
    )

    out = []
    for example in tqdm(qa_dataset):
        question = example["question"]
        raw_prompt = qa_chain.combine_documents_chain.llm_chain.prompt_cache[question]
        out.append({**example, "question_with_context": raw_prompt})
    return out


def main(dataset_size: int, context_size: int, skip_datagen: bool = False):
    """
    Args:
        dataset_size: # of QA pairs to synthesize
        context_size: # of tokens accepted by LM we'll finetune on QA pairs
                      e.g. google/flan-t5-* is only 512 tokens, so the retrieval chunk size will also be small.
        skip_datagen: already synthesized the datasets, want to re-construct the corpus + retriever for evaluation
    """

    corpus = _extract_corpus_from_transcripts()
    # TODO: temporarily 37e6 -> 10e6
    corpus = corpus[: int(10e6)]

    # Also TODO: this is a weak approximation, having to drop samples when actually tokenizing
    chars = context_size * 4
    retrieval_chars = chars - 250
    retrieval_k = 3
    chunk_size = round(retrieval_chars // retrieval_k, -1)

    chunker = CharacterTextSplitter(
        separator=" ", chunk_size=chunk_size, chunk_overlap=chunk_size // 10
    )
    chunks = chunker.split_text(corpus)
    vectorstore = FAISS.from_texts(chunks, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(k=retrieval_k)

    if skip_datagen:
        return retriever

    llm = OpenAI(temperature=0)
    chat_llm = ChatOpenAI(temperature=0)

    # QAGenerationChain default to chat model.
    qa_gen_chain = QAGenerationChain.from_llm(chat_llm)

    # But QAChain fine-tuning non-chat model.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="question",
        return_source_documents=False,
    )

    qa_ds = generate_retrieval_qa_dataset(dataset_size, chunks, qa_gen_chain, retriever)
    t2t_qa_ds = generate_retrieval_qa_t2t_dataset(qa_ds, qa_chain)
    return t2t_qa_ds


if __name__ == "__main__":
    dataset_size = 1000
    context_size = 512  # small experiment to start: google/flan-t5-base

    # Synthesize dataset. This takes awhile with chat completions api atm.
    # t2t_qa_ds = main(dataset_size, context_size)
    # with open("./tmp/t2t_qa_ds_2023_05_17.json", "w") as f:
    #     json.dump(t2t_qa_ds, f)

    # Baseline accuracy with QAEvalChain. How well does gpt-3.5-turbo (used for datagen + eval) do on the task?
    retriever = main(dataset_size, context_size, skip_datagen=True)
    with open("./tmp/t2t_qa_ds_2023_05_17.json") as f:
        t2t_qa_ds = json.load(f)

    from langchain.evaluation.qa import QAEvalChain

    chat_llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="question",
        return_source_documents=False,
    )
    qa_eval_chain = QAEvalChain.from_llm(chat_llm)

    t2t_qa_ds_subsampled = random.sample(t2t_qa_ds, k=100)
    results = []
    for example in tqdm(t2t_qa_ds_subsampled):
        answer = qa_chain(example["question"])
        ground_truth = {"question": example["question"], "answer": example["answer"]}
        prediction = {"result": answer}
        res = qa_eval_chain.evaluate(
            [ground_truth], [prediction], question_key="question"
        )[0]
        results.append({**ground_truth, **prediction, "evaluation": res["text"]})

    results_correct = sum(
        result["evaluation"].lower() == "correct" for result in results
    )
    print(f"{results_correct=} {len(results)=}")
