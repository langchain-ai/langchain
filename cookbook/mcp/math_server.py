from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
import sys

mcp = FastMCP("Math")

class Query(BaseModel):
    query: str


class Person(BaseModel):
    name: str
    age: int


@mcp.tool()
def find_preson(name: Query) -> Person:
    """Find an info about some person by name"""
    print(f">>> Calling find_person({name})")
    return Person(name="John Doe", age=30)


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers"""
    print(f">>> Calling add({a}, {b})")
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    print(f">>> Calling multiply({a}, {b})")
    return a * b


if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
