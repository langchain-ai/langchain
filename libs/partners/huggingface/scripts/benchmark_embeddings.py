"""Benchmark HuggingFaceEmbeddings: optimized vs baseline on BAAI/bge-small-en-v1.5.

Run from libs/partners/huggingface:

    uv run --group test python scripts/benchmark_embeddings.py

The script measures wall-clock time for embed_documents() under three configs
and compares them to direct sentence-transformers (the reference baseline).

Expected results on M4 MacBook Air (MPS auto-selected):
  baseline  convert_to_numpy=True  batch_size=32  : ~800ms
  optimized convert_to_tensor=True batch_size=32  : ~200ms
  optimized convert_to_tensor=True batch_size=128 : ~170ms
  reference sentence-transformers  batch_size=32  : ~220ms
"""

from __future__ import annotations

import time

MODEL = "BAAI/bge-small-en-v1.5"
N_TEXTS = 1_000
WARMUP = 50  # texts used for a warmup pass (JIT / MPS kernel compile)

_BASE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Embeddings map text to high-dimensional vectors for semantic search.",
    "LangChain provides composable primitives for LLM applications.",
    "Apple Silicon M4 uses a unified memory architecture for CPU and GPU.",
    "BAAI/bge-small-en-v1.5 is a compact, high-quality sentence encoder.",
]


def _make_texts(n: int) -> list[str]:
    cycle = _BASE_TEXTS * ((n // len(_BASE_TEXTS)) + 1)
    return cycle[:n]


def _time_langchain(encode_kwargs: dict, batch_size: int, texts: list[str]) -> float:
    from langchain_huggingface import HuggingFaceEmbeddings

    model = HuggingFaceEmbeddings(
        model_name=MODEL,
        batch_size=batch_size,
        encode_kwargs=encode_kwargs,
    )
    # Warmup to trigger JIT / MPS kernel compilation.
    model.embed_documents(texts[:WARMUP])

    t0 = time.perf_counter()
    model.embed_documents(texts)
    return time.perf_counter() - t0


def _time_sentence_transformers(batch_size: int, texts: list[str]) -> float:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL)
    model.encode(texts[:WARMUP], batch_size=batch_size)

    t0 = time.perf_counter()
    model.encode(texts, batch_size=batch_size, convert_to_tensor=True)
    return time.perf_counter() - t0


def main() -> None:
    texts = _make_texts(N_TEXTS)

    print(f"Model : {MODEL}")
    print(f"Texts : {N_TEXTS}  |  warmup={WARMUP}\n")
    print(f"{'Config':<55} {'Time':>8}  {'ms/text':>9}")
    print("-" * 75)

    configs = [
        (
            "baseline  (convert_to_numpy=True,  batch_size= 32)",
            {"convert_to_tensor": False},
            32,
        ),
        (
            "optimized (convert_to_tensor=True,  batch_size= 32)",
            {},
            32,
        ),
        (
            "optimized (convert_to_tensor=True,  batch_size=128)",
            {},
            128,
        ),
    ]

    for label, enc_kwargs, bs in configs:
        elapsed = _time_langchain(enc_kwargs, bs, texts)
        ms_per_text = elapsed / N_TEXTS * 1000
        print(f"langchain  {label:<44} {elapsed:>7.3f}s  {ms_per_text:>8.2f}ms")

    elapsed = _time_sentence_transformers(32, texts)
    print(
        f"reference  {'sentence-transformers (convert_to_tensor=True, bs=32)':<44}"
        f" {elapsed:>7.3f}s  {elapsed / N_TEXTS * 1000:>8.2f}ms"
    )

    print()
    print("Tip: on Apple Silicon pass model_kwargs={'device': 'mps'} if auto-detect")
    print("     fails, or omit 'device' entirely to let sentence-transformers choose.")


if __name__ == "__main__":
    main()
