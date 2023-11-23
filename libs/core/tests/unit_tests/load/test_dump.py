from typing import Any, Dict

from langchain_core.load.dump import dumps
from langchain_core.load.serializable import Serializable


class Person(Serializable):
    secret: str

    you_can_see_me: str = "hello"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"secret": "SECRET"}

    @property
    def lc_attributes(self) -> Dict[str, str]:
        return {"you_can_see_me": self.you_can_see_me}


class SpecialPerson(Person):
    another_secret: str

    another_visible: str = "bye"

    # Gets merged with parent class's secrets
    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"another_secret": "ANOTHER_SECRET"}

    # Gets merged with parent class's attributes
    @property
    def lc_attributes(self) -> Dict[str, str]:
        return {"another_visible": self.another_visible}


def test_person(snapshot: Any) -> None:
    p = Person(secret="hello")
    assert dumps(p, pretty=True) == snapshot
    sp = SpecialPerson(another_secret="Wooo", secret="Hmm")
    assert dumps(sp, pretty=True) == snapshot
    assert Person.lc_id() == ["tests", "unit_tests", "load", "test_dump", "Person"]
