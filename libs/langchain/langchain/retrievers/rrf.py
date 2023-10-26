from typing import List, Optional

from langchain.schema import Document


def weighted_reciprocal_rank_fusion(
    doc_lists: List[List[Document]],
    weights: Optional[List[float]] = None,
    c: int = 60,
) -> List[Document]:
    """
    Perform weighted Reciprocal Rank Fusion on multiple rank lists.
        You can find more details about RRF here:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.
            weights: A list of weights corresponding to the queries.
                Defaults to equal weighting for all queries.
            c: A constant added to the rank, controlling the balance between
                the importance of high-ranked items and the consideration given to
                lower-ranked items. Default is 60.

        Returns:
            list: The final aggregated list of items sorted by their weighted RRF
                    scores in descending order.

    """
    if weights is None:
        weights = [1.0] * len(doc_lists)

    if len(doc_lists) != len(weights):
        raise ValueError("Number of rank lists must be equal to the number of weights.")

    # Create a union of all unique documents in the input doc_lists
    all_documents = set()
    for doc_list in doc_lists:
        for doc in doc_list:
            all_documents.add(doc.page_content)

    # Initialize the RRF score dictionary for each document
    rrf_score_dic = {doc: 0.0 for doc in all_documents}

    # Calculate RRF scores for each document
    for doc_list, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(doc_list, start=1):
            rrf_score = weight * (1 / (rank + c))
            rrf_score_dic[doc.page_content] += rrf_score

    # Sort documents by their RRF scores in descending order
    sorted_documents = sorted(
        rrf_score_dic.keys(), key=lambda x: rrf_score_dic[x], reverse=True
    )

    # Map the sorted page_content back to the original document objects
    page_content_to_doc_map = {
        doc.page_content: doc for doc_list in doc_lists for doc in doc_list
    }
    sorted_docs = [
        page_content_to_doc_map[page_content] for page_content in sorted_documents
    ]

    return sorted_docs
