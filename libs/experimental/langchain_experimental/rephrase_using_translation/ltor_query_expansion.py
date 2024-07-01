from typing import Dict, List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def default_preprocessing_func(text: str) -> List[str]:
    return text.split()


class LTORQueryExpansion:
    """
    LTORQueryExpansion class provides functionalities for query expansion
    using translation.

    Attributes:
    - model_name_fwd (str): Name of the forward translation model.
                            Default is 'Helsinki-NLP/opus-mt-en-es'.

    - model_name_reverse (str): Name of the reverse translation model.
                                Default is 'Helsinki-NLP/opus-mt-es-en'.

    Methods:
    - translate_pass(texts: List[str], pass_fwd_or_rev: str) -> List[str]:
                                                        Performs translation
                                                        using the
                                                        specified translation
                                                        direction
                                                        ('fwd' or 'rev').
    - rephrase_using_translation(query: List[str],
                                            iteration: int) -> List[List[str]]:
                                                    Rephrases the query
                                                    using translation iteratively.
    - get_expanded_queries_ltor(query: List[str],
                                iteration: int = 1) -> List[List[str]]: Returns
                                                                        expanded
                                                                        queries
                                                                        using LTOR
                                                                        approach.
    - get_relevant_documents(
        retriever: Document,
        queries: List[List[str]]) -> List[List[Document]]: Retrieves relevant
                                                documents for each query.

    - _remove_duplicates(
        relevant_documents: List[Document]) -> List[Document]: Removes
                                                               duplicate documents.

    """

    def __init__(
        self,
        model_name_fwd: str = "Helsinki-NLP/opus-mt-en-es",
        model_name_reverse: str = "Helsinki-NLP/opus-mt-es-en",
    ) -> None:
        """
        Initializes the LTORQueryExpansion object.

        Args:
        - model_name_fwd (str): Name of the forward translation model.
                                Default is 'Helsinki-NLP/opus-mt-en-es'.
        - model_name_reverse (str): Name of the reverse translation model.
                                    Default is 'Helsinki-NLP/opus-mt-es-en'.
        """

        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            raise ImportError(
                "Could not import transformers, "
                "please install with `pip install "
                "transformers`."
            )
        try:
            import torch
        except ImportError:
            raise ImportError("Could not import torch, please install torch ")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_fwd = MarianMTModel.from_pretrained(model_name_fwd)
        self.tokenizer_fwd = MarianTokenizer.from_pretrained(model_name_fwd)
        self.model_reverse = MarianMTModel.from_pretrained(model_name_reverse)
        self.tokenizer_reverse = MarianTokenizer.from_pretrained(model_name_reverse)

    def translate_pass(self, texts: List[str], pass_fwd_or_rev: str) -> List[str]:
        """
        Performs translation using the specified
        translation direction ('fwd' or 'rev').

        Args:
        - texts (List[str]): List of texts to be translated.
        - pass_fwd_or_rev (str): Translation direction. Either
                                'fwd' for forward translation or
                                'rev' for reverse translation.

        Returns:
        - translated_texts (List[str]): List of translated texts.
        """
        if pass_fwd_or_rev == "fwd":
            tokenizer = self.tokenizer_fwd
            model = self.model_fwd.to(self.device)
        else:
            tokenizer = self.tokenizer_reverse
            model = self.model_reverse.to(self.device)
        # Tokenize input texts
        input_ids = tokenizer.batch_encode_plus(
            texts, return_tensors="pt", padding=True, truncation=True
        )["input_ids"]

        input_ids = input_ids.to(self.device)
        # Perform translation
        output_ids = model.generate(input_ids)

        # Decode the translated texts
        translated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        input_ids.to("cpu")
        return translated_texts

    def rephrase_using_translation(
        self, query: List[str], iteration: int
    ) -> List[List[str]]:
        """
        Rephrases the query using translation iteratively.

        Args:
        - query (List[str]): Input query to be rephrased.
        - iteration (int): Number of iterations for query rephrasing.

        Returns:
        - queries (List[List[str]]): List of rephrased queries.
        """
        queries = [query]
        current_query = queries[-1]
        for _ in range(iteration):
            fwd_pass_text = self.translate_pass(current_query, pass_fwd_or_rev="fwd")
            rev_pass_text = self.translate_pass(fwd_pass_text, pass_fwd_or_rev="rev")
            current_query = rev_pass_text
            queries.append(rev_pass_text)
        return queries

    def get_expanded_queries_ltor(
        self, query: List[str], iteration: int = 1
    ) -> List[List[str]]:
        """
        Returns expanded queries using LTOR approach.

        Args:
        - query (List[str]): Input query.
        - iteration (int): Number of iterations for query expansion. Default is 1.

        Returns:
        - list_of_queries (List[List[str]]): List of expanded queries.
        """
        expanded_queries = self.rephrase_using_translation(query, iteration)
        list_of_queries: Dict[int, List[str]] = {}
        for i in expanded_queries:
            for count, j in enumerate(i):
                list_of_queries[count] = list_of_queries.get(count, []) + [j.lower()]

        return list(list_of_queries.values())

    def get_relevant_documents(
        self, retriever: BM25Retriever, queries: List[List[str]]
    ) -> List[List[Document]]:
        """
        Retrieves relevant documents for each query.

        Args:
        - retriever: BM25Retriever retriever object.
        - queries (List[List[str]]): List of queries.

        Returns:
        - relevant_documents (List[List[Document]]): List of relevant
                                                    documents for each query.
        """
        relevant_documents = []
        for i in queries:
            relevant_documents_per_qry = []
            for ex_qry in i:
                relevant_documents_per_qry.extend(
                    retriever.get_relevant_documents(ex_qry)
                )

            relevant_documents_per_qry = self._remove_duplicates(
                relevant_documents_per_qry
            )

            relevant_documents.append(relevant_documents_per_qry)

        return relevant_documents

    @staticmethod
    def _remove_duplicates(relevant_documents: List[Document]) -> List[Document]:
        """
        Removes duplicate documents.

        Args:
        - relevant_documents (List[Document]): List of relevant documents.

        Returns:
        - relevant_documents_wo_duplicates (List[Document]): List of relevant
        documents without duplicates.
        """
        page_contents = []
        relevant_documents_wo_duplicates = []
        for i in relevant_documents:
            if i.page_content not in page_contents:
                page_contents.append(i.page_content)
                relevant_documents_wo_duplicates.append(i)

        return relevant_documents_wo_duplicates
