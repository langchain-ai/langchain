from langchain_core.documents import Document
from typing import List

def default_preprocessing_func(text: str) -> List[str]:
    return text.split()

class LTORQueryExpansion():

    def __init__(self, model_name_fwd='Helsinki-NLP/opus-mt-en-es',
                 model_name_reverse='Helsinki-NLP/opus-mt-es-en'):

        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            raise ImportError(
                "Could not import transformers, please install with `pip install "
                "transformers`."
            )
        try:
            import torch
        except ImportError:
            raise ImportError(
                "Could not import torch, please install torch "
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_fwd = MarianMTModel.from_pretrained(model_name_fwd)
        self.tokenizer_fwd = MarianTokenizer.from_pretrained(model_name_fwd)
        self.model_reverse = MarianMTModel.from_pretrained(model_name_reverse)
        self.tokenizer_reverse = MarianTokenizer.from_pretrained(model_name_reverse)

    def translate_pass(self, texts, pass_fwd_or_rev):
        if pass_fwd_or_rev == 'fwd':
            tokenizer = self.tokenizer_fwd
            model = self.model_fwd.to(self.device)
        else:
            tokenizer = self.tokenizer_reverse
            model = self.model_reverse.to(self.device)
        # Tokenize input texts
        input_ids = tokenizer.batch_encode_plus(texts, return_tensors='pt',
                                                padding=True, truncation=True)['input_ids']

        input_ids = input_ids.to(self.device)
        # Perform translation
        output_ids = model.generate(input_ids)

        # Decode the translated texts
        translated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        input_ids.to('cpu')
        return translated_texts

    def rephrase_using_translation(self, query, iteration):
        queries = [query]
        current_query = query
        for _ in range(iteration):
            fwd_pass_text = self.translate_pass(current_query, pass_fwd_or_rev='fwd')
            rev_pass_text = self.translate_pass(fwd_pass_text, pass_fwd_or_rev='rev')
            current_query = rev_pass_text
            queries.append(rev_pass_text)
        return queries

    def _get_expanded_queries_ltor(self, query, iteration=1
                                   )-> List[Document]:
        expanded_queries = self.rephrase_using_translation(query, iteration)
        list_of_queries = {}
        for _, i in enumerate(expanded_queries):
            for count, j in enumerate(i):
                if count not in list_of_queries:
                    list_of_queries[count] = []
                if j.lower() not in list_of_queries[count]:
                    list_of_queries[count].append(j.lower())

        list_of_queries = list(list_of_queries.values())
        return list_of_queries

    def _get_relevant_documents(self, retriever, queries):
        relevant_documents = []
        for qry_id, i in enumerate(queries):
            relevant_documents_per_qry = []
            for ex_qry in i:
                relevant_documents_per_qry.extend(retriever.get_relevant_documents(ex_qry))

            relevant_documents_per_qry = self._remove_duplicates(relevant_documents_per_qry)

            relevant_documents.append(relevant_documents_per_qry)

        return relevant_documents

    def _remove_duplicates(self, relevant_documents):
        page_contents = []
        relevant_documents_wo_duplicates = []
        for i in relevant_documents:
            if i.page_content not in page_contents:
                page_contents.append(i.page_content)
                relevant_documents_wo_duplicates.append(i)

        return relevant_documents_wo_duplicates



