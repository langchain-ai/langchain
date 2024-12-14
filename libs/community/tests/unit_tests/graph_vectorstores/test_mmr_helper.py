from __future__ import annotations

import math

from langchain_community.graph_vectorstores.mmr_helper import MmrHelper

IDS = {
    "-1",
    "-2",
    "-3",
    "-4",
    "-5",
    "+1",
    "+2",
    "+3",
    "+4",
    "+5",
}


class TestMmrHelper:
    def test_mmr_helper_functional(self) -> None:
        helper = MmrHelper(k=3, query_embedding=[6, 5], lambda_mult=0.5)

        assert len(list(helper.candidate_ids())) == 0

        helper.add_candidates({"-1": [3, 5]})
        helper.add_candidates({"-2": [3, 5]})
        helper.add_candidates({"-3": [2, 6]})
        helper.add_candidates({"-4": [1, 6]})
        helper.add_candidates({"-5": [0, 6]})

        assert len(list(helper.candidate_ids())) == 5

        helper.add_candidates({"+1": [5, 3]})
        helper.add_candidates({"+2": [5, 3]})
        helper.add_candidates({"+3": [6, 2]})
        helper.add_candidates({"+4": [6, 1]})
        helper.add_candidates({"+5": [6, 0]})

        assert len(list(helper.candidate_ids())) == 10

        for idx in range(3):
            best_id = helper.pop_best()
            assert best_id in IDS
            assert len(list(helper.candidate_ids())) == 9 - idx
            assert best_id not in helper.candidate_ids()

    def test_mmr_helper_max_diversity(self) -> None:
        helper = MmrHelper(k=2, query_embedding=[6, 5], lambda_mult=0)
        helper.add_candidates({"-1": [3, 5]})
        helper.add_candidates({"-2": [3, 5]})
        helper.add_candidates({"-3": [2, 6]})
        helper.add_candidates({"-4": [1, 6]})
        helper.add_candidates({"-5": [0, 6]})

        best = {helper.pop_best(), helper.pop_best()}
        assert best == {"-1", "-5"}

    def test_mmr_helper_max_similarity(self) -> None:
        helper = MmrHelper(k=2, query_embedding=[6, 5], lambda_mult=1)
        helper.add_candidates({"-1": [3, 5]})
        helper.add_candidates({"-2": [3, 5]})
        helper.add_candidates({"-3": [2, 6]})
        helper.add_candidates({"-4": [1, 6]})
        helper.add_candidates({"-5": [0, 6]})

        best = {helper.pop_best(), helper.pop_best()}
        assert best == {"-1", "-2"}

    def test_mmr_helper_add_candidate(self) -> None:
        helper = MmrHelper(5, [0.0, 1.0])
        helper.add_candidates(
            {
                "a": [0.0, 1.0],
                "b": [1.0, 0.0],
            }
        )
        assert helper.best_id == "a"

    def test_mmr_helper_pop_best(self) -> None:
        helper = MmrHelper(5, [0.0, 1.0])
        helper.add_candidates(
            {
                "a": [0.0, 1.0],
                "b": [1.0, 0.0],
            }
        )
        assert helper.pop_best() == "a"
        assert helper.pop_best() == "b"
        assert helper.pop_best() is None

    def angular_embedding(self, angle: float) -> list[float]:
        return [math.cos(angle * math.pi), math.sin(angle * math.pi)]

    def test_mmr_helper_added_documents(self) -> None:
        """Test end to end construction and MMR search.
        The embedding function used here ensures `texts` become
        the following vectors on a circle (numbered v0 through v3):

               ______ v2
              /      \
             /        |  v1
        v3  |     .    | query
             |        /  v0
              |______/                 (N.B. very crude drawing)


        With fetch_k==2 and k==2, when query is at 0.0, (1, ),
        one expects that v2 and v0 are returned (in some order)
        because v1 is "too close" to v0 (and v0 is closer than v1)).

        Both v2 and v3 are discovered after v0.
        """
        helper = MmrHelper(5, self.angular_embedding(0.0))

        # Fetching the 2 nearest neighbors to 0.0
        helper.add_candidates(
            {
                "v0": self.angular_embedding(-0.124),
                "v1": self.angular_embedding(+0.127),
            }
        )
        assert helper.pop_best() == "v0"

        # After v0 is selected, new nodes are discovered.
        # v2 is closer than v3. v1 is "too similar" to "v0" so it's not included.
        helper.add_candidates(
            {
                "v2": self.angular_embedding(+0.25),
                "v3": self.angular_embedding(+1.0),
            }
        )
        assert helper.pop_best() == "v2"

        assert math.isclose(
            helper.selected_similarity_scores[0], 0.9251, abs_tol=0.0001
        )
        assert math.isclose(
            helper.selected_similarity_scores[1], 0.7071, abs_tol=0.0001
        )
        assert math.isclose(helper.selected_mmr_scores[0], 0.4625, abs_tol=0.0001)
        assert math.isclose(helper.selected_mmr_scores[1], 0.1608, abs_tol=0.0001)
