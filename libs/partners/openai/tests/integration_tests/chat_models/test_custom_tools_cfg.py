"""Integration tests for CFG-enabled custom tools."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.cfg_grammar import (
    validate_cfg_format,
    validate_custom_tool_output,
)


@tool(custom=True)
def execute_math_expression(expression: str) -> str:
    """Execute a mathematical expression with CFG validation.

    Args:
        expression: A mathematical expression to evaluate.

    Returns:
        The result of the mathematical expression or an error message.
    """
    # Define grammar for arithmetic expressions
    math_grammar = """
    start: expr
    expr: term (("+" | "-") term)*
    term: factor (("*" | "/") factor)*
    factor: NUMBER | "(" expr ")"
    NUMBER: /[0-9]+(\\.[0-9]+)?/
    %import common.WS
    %ignore WS
    """

    # Create CFG validator
    tool_format = {"type": "grammar", "grammar": math_grammar}
    validator = validate_cfg_format(tool_format)

    # Validate input against grammar
    if not validate_custom_tool_output(expression, validator):
        return f"Grammar validation failed for expression: {expression}"

    # If valid, evaluate safely (in practice, use a safer evaluator)
    try:
        # Simple evaluation - in production, use ast.literal_eval or similar
        # for safety!
        result = eval(expression)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Execution error: {e}"


@tool(custom=True)
def generate_sql_query(query: str) -> str:
    """Generate and validate SQL SELECT queries with CFG validation.

    Args:
        query: A SQL SELECT query to validate.

    Returns:
        Validation result and mock execution.
    """
    # Define grammar for simple SELECT queries
    sql_grammar = """
    start: query
    query: "SELECT" field_list "FROM" table_name where_clause?
    field_list: field ("," field)*
    field: IDENTIFIER | "*"
    table_name: IDENTIFIER
    where_clause: "WHERE" condition
    condition: IDENTIFIER "=" STRING
    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
    STRING: /"[^"]*"/ | /'[^']*'/
    %import common.WS
    %ignore WS
    """

    # Create CFG validator
    tool_format = {"type": "grammar", "grammar": sql_grammar}
    validator = validate_cfg_format(tool_format)

    # Validate input against grammar
    if not validate_custom_tool_output(query, validator):
        return f"SQL grammar validation failed for query: {query}"

    # Mock execution of valid query
    return f"Valid SQL query: {query}\nMock result: [Sample data rows]"


class TestCFGCustomToolsIntegration:
    """Integration tests for CFG-enabled custom tools."""

    @pytest.mark.scheduled
    def test_cfg_math_tool_with_valid_expressions(self) -> None:
        """Test CFG math tool with valid mathematical expressions."""
        # Test the tool directly with valid expressions
        result1 = execute_math_expression("5 + 3")
        assert "Result: 8" in result1

        result2 = execute_math_expression("(10 - 2) * 3")
        assert "Result: 24" in result2

        result3 = execute_math_expression("15 / 3")
        assert "Result: 5" in result3

    @pytest.mark.scheduled
    def test_cfg_math_tool_with_invalid_expressions(self) -> None:
        """Test CFG math tool rejects invalid expressions."""
        # Test with invalid expressions
        result1 = execute_math_expression("hello world")
        assert "Grammar validation failed" in result1

        result2 = execute_math_expression("5 + + 3")
        assert "Grammar validation failed" in result2

        result3 = execute_math_expression("print('hello')")
        assert "Grammar validation failed" in result3

    @pytest.mark.scheduled
    def test_cfg_sql_tool_with_valid_queries(self) -> None:
        """Test CFG SQL tool with valid SELECT queries."""
        # Test with valid SQL queries
        result1 = generate_sql_query("SELECT id FROM users")
        assert "Valid SQL query" in result1
        assert "Mock result" in result1

        result2 = generate_sql_query("SELECT name, email FROM customers")
        assert "Valid SQL query" in result2

        result3 = generate_sql_query(
            'SELECT * FROM products WHERE category = "electronics"'
        )
        assert "Valid SQL query" in result3

    @pytest.mark.scheduled
    def test_cfg_sql_tool_with_invalid_queries(self) -> None:
        """Test CFG SQL tool rejects invalid queries."""
        # Test with invalid SQL queries
        result1 = generate_sql_query("DELETE FROM users")
        assert "SQL grammar validation failed" in result1

        result2 = generate_sql_query("UPDATE users SET name = 'test'")
        assert "SQL grammar validation failed" in result2

    @pytest.mark.scheduled
    def test_cfg_validator_error_handling(self) -> None:
        """Test CFG validator handles edge cases properly."""
        # Test empty input
        result1 = execute_math_expression("")
        assert "Grammar validation failed" in result1

        # Test whitespace-only input
        result2 = execute_math_expression("   ")
        assert "Grammar validation failed" in result2

        # Test special characters
        result3 = execute_math_expression("5 & 3")
        assert "Grammar validation failed" in result3

    @pytest.mark.scheduled
    def test_cfg_validator_with_complex_valid_expressions(self) -> None:
        """Test CFG validator with complex but valid expressions."""
        # Test nested parentheses
        result1 = execute_math_expression("((5 + 3) * 2) - 1")
        assert "Result: 15" in result1

        # Test decimal numbers
        result2 = execute_math_expression("3.14 * 2")
        assert "Result: 6.28" in result2

        # Test multiple operations
        result3 = execute_math_expression("10 + 5 - 3 * 2")
        assert "Result: 9" in result3  # Should follow operator precedence

    @pytest.mark.scheduled
    def test_cfg_integration_performance(self) -> None:
        """Test that CFG validation doesn't significantly impact performance."""
        import time

        expressions = [
            "1 + 1",
            "2 * 3",
            "10 / 2",
            "(4 + 6) * 2",
            "100 - 50",
            "3.14 * 2",
            "25 / 5",
            "7 + 8 - 3",
        ]

        start_time = time.time()

        for expr in expressions:
            result = execute_math_expression(expr)
            assert "Result:" in result  # All should be valid

        end_time = time.time()
        duration = end_time - start_time

        # Should complete all validations in reasonable time (< 1 second)
        assert duration < 1.0, f"CFG validation took too long: {duration}s"

    @pytest.mark.scheduled
    def test_cfg_grammar_reusability(self) -> None:
        """Test that CFG grammars can be reused efficiently."""
        math_grammar = """
        start: expr
        expr: term (("+" | "-") term)*
        term: factor (("*" | "/") factor)*
        factor: NUMBER | "(" expr ")"
        NUMBER: /[0-9]+(\\.[0-9]+)?/
        %import common.WS
        %ignore WS
        """

        tool_format = {"type": "grammar", "grammar": math_grammar}
        validator = validate_cfg_format(tool_format)

        # Reuse validator multiple times
        test_cases = [
            ("5 + 3", True),
            ("hello", False),
            ("10 * 2", True),
            ("invalid syntax", False),
            ("(1 + 2) * 3", True),
        ]

        for expression, should_be_valid in test_cases:
            result = validate_custom_tool_output(expression, validator)
            assert result == should_be_valid, f"Failed for expression: {expression}"


# Note: The following tests would require actual model integration
# which isn't fully implemented yet.


class TestCFGModelIntegration:
    """Integration tests for CFG validation with actual OpenAI models."""

    @pytest.mark.skip(reason="CFG model integration not yet implemented")
    @pytest.mark.scheduled
    def test_model_with_cfg_tools_valid_output(self) -> None:
        """Test that model generates valid CFG-compliant outputs."""
        # This would test the full integration once implemented
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # This syntax doesn't exist yet - placeholder for future implementation
        llm_with_cfg_tools = llm.bind_tools(
            [execute_math_expression],
            tool_format={
                "type": "grammar",
                "grammar": "start: expr\nexpr: NUMBER ('+' | '-' | '*' | '/') NUMBER\nNUMBER: /[0-9]+/",  # noqa: E501
            },
        )

        response = llm_with_cfg_tools.invoke(
            "Calculate 5 + 3 using the math tool. Make sure to output a valid mathematical expression."  # noqa: E501
        )

        assert isinstance(response, AIMessage)
        assert response.tool_calls
        # Would verify the tool call output matches the CFG grammar

    @pytest.mark.skip(reason="CFG model integration not yet implemented")
    @pytest.mark.scheduled
    def test_model_cfg_validation_rejection(self) -> None:
        """Test that model tool calls are rejected if they don't match CFG."""
        # This would test rejection of invalid outputs once implemented
        pass
