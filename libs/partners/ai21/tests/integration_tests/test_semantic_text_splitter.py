from ai21 import AI21Client
from langchain_core.documents import Document

from langchain_ai21 import AI21SemanticTextSplitter

TEXT = (
    "The original full name of the franchise is Pocket Monsters (ポケットモンスター, "
    "Poketto Monsutā), which was abbreviated to "
    "Pokemon during development of the original games.\n"
    "When the franchise was released internationally, the short form of the title was "
    "used, with an acute accent (´) "
    "over the e to aid in pronunciation.\n"
    "Pokémon refers to both the franchise itself and the creatures within its "
    "fictional universe.\n"
    "As a noun, it is identical in both the singular and plural, as is every "
    "individual species name;[10] it is "
    'grammatically correct to say "one Pokémon" and "many Pokémon", as well '
    'as "one Pikachu" and "many Pikachu".\n'
    "In English, Pokémon may be pronounced either /'powkɛmon/ (poe-keh-mon) or "
    "/'powkɪmon/ (poe-key-mon).\n"
    "The Pokémon franchise is set in a world in which humans coexist with creatures "
    "known as Pokémon.\n"
    "Pokémon Red and Blue contain 151 Pokémon species, with new ones being introduced "
    "in subsequent games; as of December 2023, 1,025 Pokémon species have been "
    "introduced.\n[b] Most Pokémon are inspired by real-world animals;[12] for example,"
    "Pikachu are a yellow mouse-like species[13] with lightning bolt-shaped tails[14] "
    "that possess electrical abilities.[15]\nThe player character takes the role of a "
    "Pokémon Trainer.\nThe Trainer has three primary goals: travel and explore the "
    "Pokémon world; discover and catch each Pokémon species in order to complete their"
    "Pokédex; and train a team of up to six Pokémon at a time and have them engage "
    "in battles.\nMost Pokémon can be caught with spherical devices known as Poké "
    "Balls.\nOnce the opposing Pokémon is sufficiently weakened, the Trainer throws "
    "the Poké Ball against the Pokémon, which is then transformed into a form of "
    "energy and transported into the device.\nOnce the catch is successful, "
    "the Pokémon is tamed and is under the Trainer's command from then on.\n"
    "If the Poké Ball is thrown again, the Pokémon re-materializes into its "
    "original state.\nThe Trainer's Pokémon can engage in battles against opposing "
    "Pokémon, including those in the wild or owned by other Trainers.\nBecause the "
    "franchise is aimed at children, these battles are never presented as overtly "
    "violent and contain no blood or gore.[I]\nPokémon never die in battle, instead "
    "fainting upon being defeated.[20][21][22]\nAfter a Pokémon wins a battle, it "
    "gains experience and becomes stronger.[23] After gaining a certain amount of "
    "experience points, its level increases, as well as one or more of its "
    "statistics.\nAs its level increases, the Pokémon can learn new offensive "
    "and defensive moves to use in battle.[24][25] Furthermore, many species can "
    "undergo a form of spontaneous metamorphosis called Pokémon evolution, and "
    "transform into stronger forms.[26] Most Pokémon will evolve at a certain level, "
    "while others evolve through different means, such as exposure to a certain "
    "item.[27]\n"
)


def test_split_text_to_document() -> None:
    segmentation = AI21SemanticTextSplitter()
    segments = segmentation.split_text_to_documents(source=TEXT)
    assert len(segments) > 0
    for segment in segments:
        assert segment.page_content is not None
        assert segment.metadata is not None


def test_split_text() -> None:
    segmentation = AI21SemanticTextSplitter()
    segments = segmentation.split_text(source=TEXT)
    assert len(segments) > 0


def test_split_text__when_chunk_size_is_large__should_merge_segments() -> None:
    segmentation_no_merge = AI21SemanticTextSplitter()
    segments_no_merge = segmentation_no_merge.split_text(source=TEXT)
    segmentation_merge = AI21SemanticTextSplitter(chunk_size=1000)
    segments_merge = segmentation_merge.split_text(source=TEXT)
    # Assert that a merge did happen
    assert len(segments_no_merge) > len(segments_merge)
    reconstructed_text_merged = "".join(segments_merge)
    reconstructed_text_non_merged = "".join(segments_no_merge)
    # Assert that the merge did not change the content
    assert reconstructed_text_merged == reconstructed_text_non_merged


def test_split_text__chunk_size_is_too_small__should_return_non_merged_segments() -> (
    None
):
    segmentation_no_merge = AI21SemanticTextSplitter()
    segments_no_merge = segmentation_no_merge.split_text(source=TEXT)
    segmentation_merge = AI21SemanticTextSplitter(chunk_size=10)
    segments_merge = segmentation_merge.split_text(source=TEXT)
    # Assert that a merge did happen
    assert len(segments_no_merge) == len(segments_merge)
    reconstructed_text_merged = "".join(segments_merge)
    reconstructed_text_non_merged = "".join(segments_no_merge)
    # Assert that the merge did not change the content
    assert reconstructed_text_merged == reconstructed_text_non_merged


def test_split_text__when_chunk_size_set_with_ai21_tokenizer() -> None:
    segmentation_no_merge = AI21SemanticTextSplitter(
        length_function=AI21Client().count_tokens
    )
    segments_no_merge = segmentation_no_merge.split_text(source=TEXT)
    segmentation_merge = AI21SemanticTextSplitter(
        chunk_size=1000, length_function=AI21Client().count_tokens
    )
    segments_merge = segmentation_merge.split_text(source=TEXT)
    # Assert that a merge did happen
    assert len(segments_no_merge) > len(segments_merge)
    reconstructed_text_merged = "".join(segments_merge)
    reconstructed_text_non_merged = "".join(segments_no_merge)
    # Assert that the merge did not change the content
    assert reconstructed_text_merged == reconstructed_text_non_merged


def test_create_documents() -> None:
    texts = [TEXT]
    segmentation = AI21SemanticTextSplitter()
    documents = segmentation.create_documents(texts=texts)
    assert len(documents) > 0
    for document in documents:
        assert document.page_content is not None
        assert document.metadata is not None


def test_split_documents() -> None:
    documents = [Document(page_content=TEXT, metadata={"foo": "bar"})]
    segmentation = AI21SemanticTextSplitter()
    segments = segmentation.split_documents(documents=documents)
    assert len(segments) > 0
    for segment in segments:
        assert segment.page_content is not None
        assert segment.metadata is not None
