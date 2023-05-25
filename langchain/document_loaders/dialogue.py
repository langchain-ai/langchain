import json
from abc import ABC
from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age


class Dialogue:
    """
    Build an abstract dialogue model using classes and methods to represent different dialogue elements.
    This class serves as a fundamental framework for constructing dialogue models.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.turns = []

    def add_turn(self, turn):
        """
        Create an instance of a conversation participant
        :param turn:
        :return:
        """
        self.turns.append(turn)

    def parse_dialogue(self):
        """
        The parse_dialogue function reads the specified dialogue file and parses each dialogue turn line by line.
        For each turn, the function extracts the name of the speaker and the message content from the text,
        creating a Turn instance. If the speaker is not already present in the participants dictionary,
        a new Person instance is created. Finally, the parsed Turn instance is added to the Dialogue object.

        Please note that this sample code assumes that each line in the file follows a specific format:
        <speaker>:\r\n<message>\r\n\r\n. If your file has a different format or includes other metadata,
         you may need to adjust the parsing logic accordingly.
        """
        participants = {}
        speaker_name = None
        message = None

        with open(self.file_path, encoding='utf-8') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                if speaker_name is None:
                    speaker_name, _ = line.split(':', 1)
                elif message is None:
                    message = line
                    if speaker_name not in participants:
                        participants[speaker_name] = Person(speaker_name, None)

                    speaker = participants[speaker_name]
                    turn = Turn(speaker, message)
                    self.add_turn(turn)

                    # Reset speaker_name and message for the next turn
                    speaker_name = None
                    message = None

    def display(self):
        for turn in self.turns:
            print(f"{turn.speaker.name}: {turn.message}")

    def export_to_file(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            for turn in self.turns:
                file.write(f"{turn.speaker.name}: {turn.message}\n")

    def to_dict(self):
        dialogue_dict = {"turns": []}
        for turn in self.turns:
            turn_dict = {
                "speaker": turn.speaker.name,
                "message": turn.message
            }
            dialogue_dict["turns"].append(turn_dict)
        return dialogue_dict

    def to_json(self):
        dialogue_dict = self.to_dict()
        return json.dumps(dialogue_dict, ensure_ascii=False, indent=2)

    def participants_to_export(self):
        """
        participants_to_export
        :return:
        """
        participants = set()
        for turn in self.turns:
            participants.add(turn.speaker.name)
        return ', '.join(participants)


class Turn:
    def __init__(self, speaker, message):
        self.speaker = speaker
        self.message = message


class DialogueLoader(BaseLoader, ABC):
    """Load dialogue."""

    def __init__(self, file_path: str):
        """Initialize with dialogue."""
        self.file_path = file_path
        dialogue = Dialogue(file_path=file_path)
        dialogue.parse_dialogue()
        self.dialogue = dialogue

    def load(self) -> List[Document]:
        """Load from dialogue."""
        documents = []
        participants = self.dialogue.participants_to_export()

        for turn in self.dialogue.turns:
            metadata = {"source": f"Dialogue File：{self.dialogue.file_path},"
                                  f"speaker：{turn.speaker.name}，"
                                  f"participant：{participants}"}
            turn_document = Document(page_content=turn.message, metadata=metadata.copy())
            documents.append(turn_document)

        return documents
