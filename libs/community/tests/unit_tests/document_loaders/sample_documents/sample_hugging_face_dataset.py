from typing import Any, Generator, List, Tuple

import datasets


class SampleHuggingface(datasets.GeneratorBasedBuilder):
    """Sample huggingface dataset with two different versions for testing."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="v1",
            version=datasets.Version("1.0.0"),
            description="Sample v1 description",
        ),
        datasets.BuilderConfig(
            name="v2",
            version=datasets.Version("1.0.0"),
            description="Sample v2 description",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        """This function defines the structure of the dataset"""
        return datasets.DatasetInfo(
            description="Sample Huggingface dataset",
            features=datasets.Features(
                {
                    "split": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "list": datasets.features.Sequence(datasets.Value("string")),
                    "dict": datasets.features.Sequence(
                        {
                            "dict_text": datasets.Value("string"),
                            "dict_int": datasets.Value("int32"),
                        }
                    ),
                }
            ),
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """
        This function defines how the dataset's splits will be generated.
        Args:
            dl_manager (`DownloadManager`):
                Helper for downloading datasets from files and online sources.
                This is not being used for this test file.
        """
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": "train", "name": self.config.name},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": "test", "name": self.config.name},
            ),
        ]

    def _generate_examples(
        self, split: str, name: str
    ) -> Generator[Tuple[int, object], Any, None]:
        """This function returns the examples.
        Args:
            split (`string`):
                Split to process
            name (`string`):
                Name of dataset, as defined in the BuilderConfig
        """
        if name == "v1":
            yield (
                1,
                {
                    "split": split,
                    "text": "This is text in version 1",
                    "list": ["List item 1", "List item 2", "List item 3"],
                    "dict": [
                        {
                            "dict_text": "Object text 1",
                            "dict_int": "1",
                        },
                        {
                            "dict_text": "Object text 2",
                            "dict_int": str(000),
                        },
                    ],
                },
            )
        elif name == "v2":
            yield (
                2,
                {
                    "split": split,
                    "text": "This is text in version 2",
                    "list": ["Hello", "Bonjour", "Hola"],
                    "dict": [
                        {
                            "dict_text": "Hello world!",
                            "dict_int": "2",
                        },
                        {
                            "dict_text": "langchain is cool",
                            "dict_int": str(123),
                        },
                    ],
                },
            )
