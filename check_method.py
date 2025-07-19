#!/usr/bin/env python3

from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
import inspect

parser = JsonOutputKeyToolsParser(key_name='func', first_tool_only=True, return_id=True)
print('Method source:')
print(inspect.getsource(parser.parse_result))
