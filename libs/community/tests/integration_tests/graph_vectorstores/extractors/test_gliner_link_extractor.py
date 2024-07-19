import pytest
from langchain_core.graph_vectorstores.links import Link

from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor

PAGE_1 = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃ'tjɐnu
ʁɔ'naldu]; born 5 February 1985) is a Portuguese professional footballer who
plays as a forward for and captains both Saudi Pro League club Al Nassr and the
Portugal national team. Widely regarded as one of the greatest players of all
time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's
Player of the Year Awards, and four European Golden Shoes, the most by a
European player. He has won 33 trophies in his career, including seven league
titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA
Nations League. Ronaldo holds the records for most appearances (183), goals
(140) and assists (42) in the Champions League, goals in the European
Championship (14), international goals (128) and international appearances
(205). He is one of the few players to have made over 1,200 professional career
appearances, the most by an outfield player, and has scored over 850 official
senior career goals for club and country, making him the top goalscorer of all
time.
"""


@pytest.mark.requires("gliner")
def test_one_from_keywords() -> None:
    extractor = GLiNERLinkExtractor(
        labels=["Person", "Award", "Date", "Competitions", "Teams"]
    )

    results = extractor.extract_one(PAGE_1)
    assert results == {
        Link.bidir(kind="entity:Person", tag="Cristiano Ronaldo dos Santos Aveiro"),
        Link.bidir(kind="entity:Award", tag="European Golden Shoes"),
        Link.bidir(kind="entity:Competitions", tag="European\nChampionship"),
        Link.bidir(kind="entity:Award", tag="UEFA Men's\nPlayer of the Year Awards"),
        Link.bidir(kind="entity:Date", tag="5 February 1985"),
        Link.bidir(kind="entity:Competitions", tag="UEFA Champions Leagues"),
        Link.bidir(kind="entity:Teams", tag="Portugal national team"),
        Link.bidir(kind="entity:Competitions", tag="UEFA European Championship"),
        Link.bidir(kind="entity:Competitions", tag="UEFA\nNations League"),
        Link.bidir(kind="entity:Award", tag="Ballon d'Or"),
    }
