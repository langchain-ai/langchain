# File changes (if any)
def similarity_search_with_relevance_scores(self, query_vector, score_threshold):
# Existing code...
if self.distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
docs_and_similarities = [
(doc, similarity) for doc, similarity in docs_and_similarities if similarity >= score_threshold
]
else:
docs_and_similarities = [
(doc, 1.0 - similarity) for doc, similarity in docs_and_similarities if (1.0 - similarity) >= score_threshold
]
# Continue with the rest of the function...