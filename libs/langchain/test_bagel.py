from langchain.vectorstores import Bagel
from bagel.config import Settings


def test_create_bagel() -> Bagel:
    setting = Settings(
        bagel_api_impl="rest",
        bagel_server_host="api.bageldb.ai",
    )
    return Bagel(client_settings=setting)


def test_add_only_texts(bagel: Bagel) -> None:
    bagel.add_texts(texts=["hello bagel", "hello langchain"])
    print(">> add_texts with only text")


def test_similarity_search(bagel: Bagel) -> None:
    result = bagel.similarity_search(query="bagel", k=1)
    print(f">> {result}")


def main():
    """Bagel intigaration test"""
    bagel_instance = test_create_bagel()
    test_add_only_texts(bagel_instance)
    test_similarity_search(bagel_instance)


if __name__ == "__main__":
    main()
