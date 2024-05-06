import json
import re
import zipfile
from abc import ABC
from pathlib import Path
from typing import Iterator, List, Set, Tuple

from langchain_community.docstore.document import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob


class VsdxParser(BaseBlobParser, ABC):
    """Parser for vsdx files."""

    def parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[override]
        """Parse a vsdx file."""
        return self.lazy_parse(blob)

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Retrieve the contents of pages from a .vsdx file
        and insert them into documents, one document per page."""

        with blob.as_bytes_io() as pdf_file_obj:
            with zipfile.ZipFile(pdf_file_obj, "r") as zfile:
                pages = self.get_pages_content(zfile, blob.source)  # type: ignore[arg-type]

        yield from [
            Document(
                page_content=page_content,
                metadata={
                    "source": blob.source,
                    "page": page_number,
                    "page_name": page_name,
                },
            )
            for page_number, page_name, page_content in pages
        ]

    def get_pages_content(
        self, zfile: zipfile.ZipFile, source: str
    ) -> List[Tuple[int, str, str]]:
        """Get the content of the pages of a vsdx file.

        Attributes:
            zfile (zipfile.ZipFile): The vsdx file under zip format.
            source (str): The path of the vsdx file.

        Returns:
            list[tuple[int, str, str]]: A list of tuples containing the page number,
            the name of the page and the content of the page
            for each page of the vsdx file.
        """

        try:
            import xmltodict
        except ImportError:
            raise ImportError(
                "The xmltodict library is required to parse vsdx files. "
                "Please install it with `pip install xmltodict`."
            )

        if "visio/pages/pages.xml" not in zfile.namelist():
            print("WARNING - No pages.xml file found in {}".format(source))  # noqa: T201
            return  # type: ignore[return-value]
        if "visio/pages/_rels/pages.xml.rels" not in zfile.namelist():
            print("WARNING - No pages.xml.rels file found in {}".format(source))  # noqa: T201
            return  # type: ignore[return-value]
        if "docProps/app.xml" not in zfile.namelist():
            print("WARNING - No app.xml file found in {}".format(source))  # noqa: T201
            return  # type: ignore[return-value]

        pagesxml_content: dict = xmltodict.parse(zfile.read("visio/pages/pages.xml"))
        appxml_content: dict = xmltodict.parse(zfile.read("docProps/app.xml"))
        pagesxmlrels_content: dict = xmltodict.parse(
            zfile.read("visio/pages/_rels/pages.xml.rels")
        )

        if isinstance(pagesxml_content["Pages"]["Page"], list):
            disordered_names: List[str] = [
                rel["@Name"].strip() for rel in pagesxml_content["Pages"]["Page"]
            ]
        else:
            disordered_names: List[str] = [  # type: ignore[no-redef]
                pagesxml_content["Pages"]["Page"]["@Name"].strip()
            ]
        if isinstance(pagesxmlrels_content["Relationships"]["Relationship"], list):
            disordered_paths: List[str] = [
                "visio/pages/" + rel["@Target"]
                for rel in pagesxmlrels_content["Relationships"]["Relationship"]
            ]
        else:
            disordered_paths: List[str] = [  # type: ignore[no-redef]
                "visio/pages/"
                + pagesxmlrels_content["Relationships"]["Relationship"]["@Target"]
            ]
        ordered_names: List[str] = appxml_content["Properties"]["TitlesOfParts"][
            "vt:vector"
        ]["vt:lpstr"][: len(disordered_names)]
        ordered_names = [name.strip() for name in ordered_names]
        ordered_paths = [
            disordered_paths[disordered_names.index(name.strip())]
            for name in ordered_names
        ]

        # Pages out of order and without content of their relationships
        disordered_pages = []
        for path in ordered_paths:
            content = zfile.read(path)
            string_content = json.dumps(xmltodict.parse(content))

            samples = re.findall(
                r'"#text"\s*:\s*"([^\\"]*(?:\\.[^\\"]*)*)"', string_content
            )
            if len(samples) > 0:
                page_content = "\n".join(samples)
                map_symboles = {
                    "\\n": "\n",
                    "\\t": "\t",
                    "\\u2013": "-",
                    "\\u2019": "'",
                    "\\u00e9r": "é",
                    "\\u00f4me": "ô",
                }
                for key, value in map_symboles.items():
                    page_content = page_content.replace(key, value)

                disordered_pages.append({"page": path, "page_content": page_content})

        # Direct relationships of each page in a dict format
        pagexml_rels = [
            {
                "path": page_path,
                "content": xmltodict.parse(
                    zfile.read(f"visio/pages/_rels/{Path(page_path).stem}.xml.rels")
                ),
            }
            for page_path in ordered_paths
            if f"visio/pages/_rels/{Path(page_path).stem}.xml.rels" in zfile.namelist()
        ]

        # Pages in order and with content of their relationships (direct and indirect)
        ordered_pages: List[Tuple[int, str, str]] = []
        for page_number, (path, page_name) in enumerate(
            zip(ordered_paths, ordered_names)
        ):
            relationships = self.get_relationships(
                path, zfile, ordered_paths, pagexml_rels
            )
            page_content = "\n".join(
                [
                    page_["page_content"]
                    for page_ in disordered_pages
                    if page_["page"] in relationships
                ]
                + [
                    page_["page_content"]
                    for page_ in disordered_pages
                    if page_["page"] == path
                ]
            )
            ordered_pages.append((page_number, page_name, page_content))

        return ordered_pages

    def get_relationships(
        self,
        page: str,
        zfile: zipfile.ZipFile,
        filelist: List[str],
        pagexml_rels: List[dict],
    ) -> Set[str]:
        """Get the relationships of a page and the relationships of its relationships,
        etc... recursively.
        Pages are based on other pages (ex: background page),
        so we need to get all the relationships to get all the content of a single page.
        """

        name_path = Path(page).name
        parent_path = Path(page).parent
        rels_path = parent_path / f"_rels/{name_path}.rels"

        if str(rels_path) not in zfile.namelist():
            return set()

        pagexml_rels_content = next(
            page_["content"] for page_ in pagexml_rels if page_["path"] == page
        )

        if isinstance(pagexml_rels_content["Relationships"]["Relationship"], list):
            targets = [
                rel["@Target"]
                for rel in pagexml_rels_content["Relationships"]["Relationship"]
            ]
        else:
            targets = [pagexml_rels_content["Relationships"]["Relationship"]["@Target"]]

        relationships = set(
            [str(parent_path / target) for target in targets]
        ).intersection(filelist)

        for rel in relationships:
            relationships = relationships | self.get_relationships(
                rel, zfile, filelist, pagexml_rels
            )

        return relationships
