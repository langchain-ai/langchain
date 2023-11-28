from langchain.chains.graph_qa.dgraph_chain import DGraphQAChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.graphs.dgraph_graph import DGraph


def populate_dgraph_database() -> None:
    schema = """
  <age>: int .
  <cast>: [uid] @reverse .
  <name>: string @index(hash) .
  <year>: string @index(exact) .
  type <Movie> {
    cast
    name
    year
  }
  type <Actor> {
    name
    age
  }
  """

    actorsRDF = """
  _:tom <name> "Tom Hanks" .
  _:tom <age> "63" .
  _:tom <dgraph.type> "Actor" .

  _:jennifer <name> "Jennifer Lawrence" .
  _:jennifer <age> "33" .
  _:jennifer <dgraph.type> "Actor" .


  _:porom <name> "Porom Kamal" .
  _:porom <age> "21" .
  _:porom <dgraph.type> "Actor" .

  _:forrest <name> "Forrest Gump" .
  _:forrest <year> "1994" .
  _:forrest <dgraph.type> "Movie" .
  _:forrest <cast> _:tom .

  _:terminal <name> "The Terminal" .
  _:terminal <year> "2004" .
  _:terminal <dgraph.type> "Movie" .
  _:terminal <cast> _:tom .
  _:terminal <cast> _:porom .

  _:hunger <name> "The Hunger Games" .
  _:hunger <year> "2012" .
  _:hunger <dgraph.type> "Movie" .
  _:hunger <cast> _:jennifer .
  """
    dgraph = DGraph(clientUrl="localhost:9080")
    dgraph.add_schema(schema)
    dgraph.add_node_rdf(actorsRDF)


def test_connect_dgraph() -> None:
    """Test that the DGraph database is correctly instantiated and connected."""
    dgraph = DGraph(clientUrl="localhost:9080")

    sample_dql_result = dgraph.query("{ me(func: has(name)) { name } }")
    assert ["me"] == sample_dql_result


def test_empty_schema() -> None:
    """Test that the schema is empty for an empty DGraph Database"""
    dgraph = DGraph(clientUrl="localhost:9080")

    assert dgraph.get_schema() == {}


def test_dql_generation() -> None:
    """Test that the DQL query is correctly generated for the given user input."""
    populate_dgraph_database()
    dgraph = DGraph(clientUrl="localhost:9080")
    chain = DGraphQAChain.from_llm(ChatOpenAI(temperature=0), graph=dgraph)
    output = chain("What is the UID of Tom Hanks?")
    assert output["dql_result"] is not None

    output = chain("How old is Tom Hanks?")
    assert output[chain.output_key] is not None
    assert "63" in output[chain.output_key]
