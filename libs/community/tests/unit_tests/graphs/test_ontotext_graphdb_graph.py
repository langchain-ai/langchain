import os
import tempfile
import unittest

import pytest


class TestOntotextGraphDBGraph(unittest.TestCase):
    def test_import(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph  # noqa: F401

    @pytest.mark.requires("rdflib", "SPARQLWrapper")
    def test_validate_user_query_wrong_type(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph

        with self.assertRaises(TypeError) as e:
            OntotextGraphDBGraph._validate_user_query(
                [  # type: ignore[arg-type]
                    "PREFIX starwars: <https://swapi.co/ontology/> "
                    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
                    "DESCRIBE starwars: ?term "
                    "WHERE {?term rdfs:isDefinedBy starwars: }"
                ]
            )
        self.assertEqual("Ontology query must be provided as string.", str(e.exception))

    @pytest.mark.requires("rdflib", "SPARQLWrapper")
    def test_validate_user_query_invalid_sparql_syntax(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph

        with self.assertRaises(ValueError) as e:
            OntotextGraphDBGraph._validate_user_query(
                "CONSTRUCT {?s ?p ?o} FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o"
            )
        self.assertEqual(
            "('Ontology query is not a valid SPARQL query.', "
            "Expected ConstructQuery, "
            "found end of text  (at char 70), (line:1, col:71))",
            str(e.exception),
        )

    @pytest.mark.requires("rdflib", "SPARQLWrapper")
    def test_validate_user_query_invalid_query_type_select(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph

        with self.assertRaises(ValueError) as e:
            OntotextGraphDBGraph._validate_user_query("SELECT * { ?s ?p ?o }")
        self.assertEqual(
            "Invalid query type. Only CONSTRUCT queries are supported.",
            str(e.exception),
        )

    @pytest.mark.requires("rdflib", "SPARQLWrapper")
    def test_validate_user_query_invalid_query_type_ask(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph

        with self.assertRaises(ValueError) as e:
            OntotextGraphDBGraph._validate_user_query("ASK { ?s ?p ?o }")
        self.assertEqual(
            "Invalid query type. Only CONSTRUCT queries are supported.",
            str(e.exception),
        )

    @pytest.mark.requires("rdflib", "SPARQLWrapper")
    def test_validate_user_query_invalid_query_type_describe(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph

        with self.assertRaises(ValueError) as e:
            OntotextGraphDBGraph._validate_user_query(
                "PREFIX swapi: <https://swapi.co/ontology/> "
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
                "DESCRIBE ?term WHERE { ?term rdfs:isDefinedBy swapi: }"
            )
        self.assertEqual(
            "Invalid query type. Only CONSTRUCT queries are supported.",
            str(e.exception),
        )

    @pytest.mark.requires("rdflib", "SPARQLWrapper")
    def test_validate_user_query_construct(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph

        OntotextGraphDBGraph._validate_user_query(
            "CONSTRUCT {?s ?p ?o} FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}"
        )

    @pytest.mark.requires("rdflib", "SPARQLWrapper")
    def test_check_connectivity(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph

        with self.assertRaises(ValueError) as e:
            OntotextGraphDBGraph(
                query_endpoint="http://localhost:7200/repositories/non-existing-repository",
                query_ontology="PREFIX swapi: <https://swapi.co/ontology/> "
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
                "DESCRIBE ?term WHERE {?term rdfs:isDefinedBy swapi: }",
            )
        self.assertEqual(
            "Could not query the provided endpoint. "
            "Please, check, if the value of the provided "
            "query_endpoint points to the right repository.",
            str(e.exception),
        )

    @pytest.mark.requires("rdflib", "SPARQLWrapper")
    def test_local_file_does_not_exist(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph

        non_existing_file = os.path.join("non", "existing", "path", "to", "file.ttl")
        with self.assertRaises(FileNotFoundError) as e:
            OntotextGraphDBGraph._load_ontology_schema_from_file(non_existing_file)
        self.assertEqual(f"File {non_existing_file} does not exist.", str(e.exception))

    @pytest.mark.requires("rdflib", "SPARQLWrapper")
    def test_local_file_no_access(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph

        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file_name = tmp_file.name

            # Set file permissions to write and execute only
            os.chmod(tmp_file_name, 0o300)

            with self.assertRaises(PermissionError) as e:
                OntotextGraphDBGraph._load_ontology_schema_from_file(tmp_file_name)

            self.assertEqual(
                f"Read permission for {tmp_file_name} is restricted", str(e.exception)
            )

    @pytest.mark.requires("rdflib", "SPARQLWrapper")
    def test_local_file_bad_syntax(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph

        with tempfile.TemporaryDirectory() as tempdir:
            tmp_file_path = os.path.join(tempdir, "starwars-ontology.trig")
            with open(tmp_file_path, "w") as tmp_file:
                tmp_file.write("invalid trig")

            with self.assertRaises(ValueError) as e:
                OntotextGraphDBGraph._load_ontology_schema_from_file(tmp_file_path)
            self.assertEqual(
                f"('Invalid file format for {tmp_file_path} : '"
                ", BadSyntax('', 0, 'invalid trig', 0, "
                "'expected directive or statement'))",
                str(e.exception),
            )

    @pytest.mark.requires("rdflib", "SPARQLWrapper")
    def test_both_query_and_local_file_provided(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph

        with self.assertRaises(ValueError) as e:
            OntotextGraphDBGraph(
                query_endpoint="http://localhost:7200/repositories/non-existing-repository",
                query_ontology="CONSTRUCT {?s ?p ?o}"
                "FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}",
                local_file="starwars-ontology-wrong.trig",
            )
        self.assertEqual(
            "Both file and query provided. Only one is allowed.", str(e.exception)
        )

    @pytest.mark.requires("rdflib", "SPARQLWrapper")
    def test_nor_query_nor_local_file_provided(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph

        with self.assertRaises(ValueError) as e:
            OntotextGraphDBGraph(
                query_endpoint="http://localhost:7200/repositories/non-existing-repository",
            )
        self.assertEqual(
            "Neither file nor query provided. One is required.", str(e.exception)
        )
